import argparse
import json
from pathlib import Path
import torch
import numpy as np
import librosa
from tqdm import tqdm

from chatterbox.mtl_tts import ChatterboxMultilingualTTS, punc_norm
from chatterbox.models.t3.t3 import T3
from chatterbox.models.t3.modules.t3_config import T3Config
from chatterbox.models.s3tokenizer import S3_SR

from finetune_mtl import patched_from_local  # same dir as this script


def load_lines(metadata_file: str):
    data = []
    meta_path = Path(metadata_file)
    root = meta_path.parent
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) != 2:
                print("Skipping malformed line:", line)
                continue
            rel_audio, text = parts
            audio_path = (root / rel_audio).resolve()
            if not audio_path.exists():
                print("Audio file does not exist:", audio_path)
                continue
            data.append({"audio": str(audio_path), "text": text})
    return data


def main():
    # ---- CLI args ----
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata_file",
        type=str,
        required=True,
        help="Path to metadata file with 'rel/path.wav|text' lines.",
    )
    parser.add_argument(
        "--local_model_dir",
        type=str,
        required=True,
        help="Directory with ve.pt, t3_*.safetensors, s3gen.pt, tokenizer, etc.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Where to save the precomputed .pt file.",
    )
    parser.add_argument(
        "--language_id",
        type=str,
        default="sv",
        help="Language ID for text tokenizer (e.g. 'sv').",
    )
    args = parser.parse_args()

    metadata_file = args.metadata_file
    local_model_dir = args.local_model_dir
    language_id = args.language_id
    out_path = args.output_path

    print(f"Metadata file: {metadata_file}")
    print(f"Local model dir: {local_model_dir}")
    print(f"Output path: {out_path}")
    print(f"Language ID: {language_id}")

    # 1) Load model once
    mtl_model = patched_from_local(local_model_dir, device="cpu")
    t3_config = mtl_model.t3.hp
    text_tokenizer = mtl_model.tokenizer
    speech_tokenizer = mtl_model.s3gen.tokenizer
    voice_encoder = mtl_model.ve

    s3_sr = S3_SR
    enc_cond_audio_len_samples = int(
        3.0 * s3_sr
    )  # 3 seconds, match audio_prompt_duration_s

    # 2) Load metadata
    items = load_lines(metadata_file)
    print(f"Loaded {len(items)} lines from metadata")

    precomputed = []

    for i, item in enumerate(tqdm(items, desc="Precomputing T3 features")):
        audio_path = item["audio"]
        text = item["text"]

        try:
            # ---- audio: load + resample ----
            wav, orig_sr = librosa.load(audio_path, sr=None, mono=True)
            if orig_sr != s3_sr:
                wav = librosa.resample(wav, orig_sr=orig_sr, target_sr=s3_sr)
            if wav.ndim > 1:
                wav = librosa.to_mono(wav)
            wav = wav.astype(np.float32)

            if len(wav) == 0:
                print("Empty audio, skipping:", audio_path)
                continue

            # ---- speaker embedding ----
            spk_emb_np = voice_encoder.embeds_from_wavs([wav], sample_rate=s3_sr)
            spk_emb = spk_emb_np[0].astype("float32")

            # ---- text tokens ----
            norm_text = punc_norm(text)
            raw_text_tokens = text_tokenizer.text_to_tokens(
                norm_text, language_id=language_id
            ).squeeze(0)

            text_tokens = torch.nn.functional.pad(
                raw_text_tokens, (1, 0), value=t3_config.start_text_token
            )
            text_tokens = torch.nn.functional.pad(
                text_tokens, (0, 1), value=t3_config.stop_text_token
            )

            # match data_args.max_text_len = 256
            if len(text_tokens) > 256:
                text_tokens = text_tokens[:255]
                text_tokens = torch.cat(
                    [
                        text_tokens,
                        torch.tensor(
                            [t3_config.stop_text_token], dtype=text_tokens.dtype
                        ),
                    ]
                )
            text_tokens = text_tokens.to(torch.int32).numpy()

            # ---- full speech tokens ----
            raw_speech_tokens_batch, speech_token_lengths_batch = (
                speech_tokenizer.forward([wav])
            )
            if raw_speech_tokens_batch is None or speech_token_lengths_batch is None:
                print("S3 tokenizer returned None for:", audio_path)
                continue

            length = speech_token_lengths_batch.squeeze(0).item()
            raw_speech_tokens = raw_speech_tokens_batch.squeeze(0)[:length]

            speech_tokens = torch.nn.functional.pad(
                raw_speech_tokens, (1, 0), value=t3_config.start_speech_token
            )
            speech_tokens = torch.nn.functional.pad(
                speech_tokens, (0, 1), value=t3_config.stop_speech_token
            )

            # match data_args.max_speech_len = 800
            if len(speech_tokens) > 800:
                speech_tokens = speech_tokens[:799]
                speech_tokens = torch.cat(
                    [
                        speech_tokens,
                        torch.tensor(
                            [t3_config.stop_speech_token], dtype=speech_tokens.dtype
                        ),
                    ]
                )
            speech_tokens = speech_tokens.to(torch.int32).numpy()

            # ---- cond prompt speech tokens (first N seconds) ----
            cond_wav = wav[:enc_cond_audio_len_samples]
            if len(cond_wav) == 0:
                cond_prompt = np.zeros(
                    (t3_config.speech_cond_prompt_len,), dtype="int32"
                )
            else:
                cond_prompt_batch, _ = speech_tokenizer.forward(
                    [cond_wav], max_len=t3_config.speech_cond_prompt_len
                )
                if cond_prompt_batch is None:
                    cond_prompt = np.zeros(
                        (t3_config.speech_cond_prompt_len,), dtype="int32"
                    )
                else:
                    cond_prompt_t = cond_prompt_batch.squeeze(0)
                    # pad / crop to exact length
                    if cond_prompt_t.shape[0] > t3_config.speech_cond_prompt_len:
                        cond_prompt_t = cond_prompt_t[
                            : t3_config.speech_cond_prompt_len
                        ]
                    elif cond_prompt_t.shape[0] < t3_config.speech_cond_prompt_len:
                        pad_len = (
                            t3_config.speech_cond_prompt_len - cond_prompt_t.shape[0]
                        )
                        cond_prompt_t = torch.nn.functional.pad(
                            cond_prompt_t, (0, pad_len), value=0
                        )
                    cond_prompt = cond_prompt_t.to(torch.int32).numpy()

            emotion_adv_scalar = 0.5

            precomputed.append(
                {
                    "text_tokens": text_tokens,
                    "speech_tokens": speech_tokens,
                    "t3_cond_speaker_emb": spk_emb,
                    "t3_cond_prompt_speech_tokens": cond_prompt,
                    "emotion_adv_scalar": float(emotion_adv_scalar),
                }
            )

        except Exception as e:
            print(f"Error on {audio_path}: {e}")
            continue

    torch.save(precomputed, out_path)
    print("Saved", len(precomputed), "examples to", out_path)


if __name__ == "__main__":
    main()
