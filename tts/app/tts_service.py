from dataclasses import dataclass, field
from typing import List
import contextlib
import os
import re
import time
import subprocess
import shutil
from pathlib import Path
from typing import Optional
import torch
import torchaudio as ta
import numpy as np
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file
from chatterbox.models.voice_encoder import VoiceEncoder
from chatterbox.mtl_tts import ChatterboxMultilingualTTS, Conditionals
import whisper
from jiwer import cer as jiwer_cer, wer as jiwer_wer
from dataclasses import dataclass


@dataclass()
class FinetuningProgress:
    progress: int
    time_left: float

@dataclass
class EvaluationSentenceResult:
    # The result for individual sentence.
    index: int
    ref_text: str
    base_transcribed: str
    ft_transcribed: str
    base_cer: float
    ft_cer: float
    base_wer: float
    ft_wer: float

@dataclass
class EvaluationResult:
    # The overall evaluation result.
    sentence_results: List[EvaluationSentenceResult] = field(default_factory=list)
    base_cer_avg: float = 0.0
    ft_cer_avg: float = 0.0
    base_wer_avg: float = 0.0
    ft_wer_avg: float = 0.0

class TTSService:
    
    def __init__(self, repo_id="ResembleAI/chatterbox", data_root="/app/data", device=None):
        self.repo_id = repo_id
        self.data_root = Path(data_root)
        self.base_dir = self.data_root / "models" / "base" / "chatterbox"
        self.ft_dir = self.data_root / "models" / "checkpoints" / "chatterbox_finetuned_swedish" / "merged_for_infer"
        self.device, self.fp16, self.no_cuda = self._detect_device(device)
        self.base_model = None
        self.ft_model = None
        self.whisper_model = None

    def _detect_device(self, device):
        has_cuda = torch.cuda.is_available()
        has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if device:
            return device, False, device == "cpu"
        if has_mps:
            return "mps", False, True
        elif has_cuda:
            return "cuda", True, False
        else:
            return "cpu", False, True

    def download_base_model(self, cache_folder=None):
        models_folder = self.data_root / "models"
        base_folder = self.base_dir
        checkpoints_folder = models_folder / "checkpoints"
        cache_folder = Path(cache_folder) if cache_folder else (models_folder / "hf_cache")
        for p in (base_folder, checkpoints_folder, cache_folder):
            p.mkdir(parents=True, exist_ok=True)
        allow = [
            "ve.pt", "ve.safetensors", "s3gen.pt", "conds.pt", "Cangjie5_TC.json",
            "mtl_tokenizer.json", "tokenizer.json", "grapheme_mtl_merged_expanded_v1.json",
            "t3_23lang.safetensors", "t3_mtl23ls_v2.safetensors",
        ]
        ckpt_dir = Path(snapshot_download(
            repo_id=self.repo_id,
            repo_type="model",
            revision="main",
            allow_patterns=allow,
            token=os.getenv("HF_TOKEN"),
            cache_dir=str(cache_folder),
            local_dir=str(base_folder),
            local_dir_use_symlinks=False
        ))
        return ckpt_dir

    def finetune(
        self,
        output_dir: str,
        metadata_file: str,
        dataset_dir: str,
        local_model_dir: Optional[str] = None,
        train_split_name: str = "train",
        eval_split_size: float = 0.1,
        num_train_epochs: int = 10,
        per_device_train_batch_size: int = 2,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 1e-5,
        warmup_steps: int = 200,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        logging_steps: int = 10,
        evaluation_strategy: str = "epoch",
        save_strategy: str = "epoch",
        save_total_limit: int = 5,
        report_to: str = "tensorboard",
        do_train: bool = True,
        do_eval: bool = True,
        dataloader_pin_memory: bool = False,
        eval_on_start: bool = True,
        label_names: str = "labels_speech",
        text_column_name: str = "text_scribe",
        language_id: str = "sv",
        dataloader_num_workers: int = 0,
        seed: int = 42,
        script_path: str = "/app/chatterbox/finetune_mtl.py",
        progress_file: Optional[str] = None # Will be set to default if None
    ):
        """
        Run finetuning with all configurable parameters.
        """
        # Use /app/data for output_dir if not absolute
        output_dir = str(self.data_root / output_dir) if not os.path.isabs(output_dir) else output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        if local_model_dir is None:
            local_model_dir = str(self.base_dir)
        if progress_file is None:
            progress_file = str(self.data_root / "output" / "ft_progress.json")
        cmd = [
            "python3", script_path,
            "--output_dir", output_dir,
            "--local_model_dir", local_model_dir,
            "--metadata_file", metadata_file,
            "--dataset_dir", dataset_dir,
            "--train_split_name", train_split_name,
            "--eval_split_size", str(eval_split_size),
            "--num_train_epochs", str(num_train_epochs),
            "--per_device_train_batch_size", str(per_device_train_batch_size),
            "--gradient_accumulation_steps", str(gradient_accumulation_steps),
            "--learning_rate", str(learning_rate),
            "--warmup_steps", str(warmup_steps),
            "--weight_decay", str(weight_decay),
            "--max_grad_norm", str(max_grad_norm),
            "--logging_steps", str(logging_steps),
            "--evaluation_strategy", evaluation_strategy,
            "--save_strategy", save_strategy,
            "--save_total_limit", str(save_total_limit),
            "--report_to", report_to,
        ]
        if do_train:
            cmd.append("--do_train")
        if do_eval:
            cmd.append("--do_eval")
        cmd += [
            "--dataloader_pin_memory", str(dataloader_pin_memory),
            "--eval_on_start", str(eval_on_start),
            "--label_names", label_names,
            "--text_column_name", text_column_name,
            "--language_id", language_id,
            "--dataloader_num_workers", str(dataloader_num_workers),
            "--seed", str(seed),
        ]

         # Write progress to a file.
        cmd.extend(["--progress_file", progress_file])
        # Also, delete the file if it already exists, since it is invalid.
        progress_path = Path(progress_file)
        if progress_path.is_file():
            progress_path.unlink()

        env = os.environ.copy()
        if self.device == "mps":
            env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        if self.no_cuda:
            cmd.append("--no_cuda")
        
        print(f"dataset_dir: {dataset_dir}")
        print(f"metadata_file: {metadata_file}")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in proc.stdout: # type: ignore
                print(line, end="")  # replace with logger if available
            proc.wait()
            if proc.returncode != 0:
                raise RuntimeError(f"Finetune script exited with code {proc.returncode}")
        except Exception:
            self.is_processing = False
            try:
                proc.kill()
                self.finetune_proc = None
            except Exception:
                pass
            raise

        # try:
        #     # capture output so we can log it if the subprocess fails
        #     completed = subprocess.run(cmd, env=env, check=True, capture_output=True, text=True)
        # except subprocess.CalledProcessError as e:
        #     try:
        #         print(f"Finetuning subprocess failed with return code {e.returncode}")
        #         print("stdout:", getattr(e, "stdout", None))
        #         print("stderr:", getattr(e, "stderr", None))
        #     except Exception:
        #         pass
        #     raise RuntimeError(f"Finetuning failed (returncode={getattr(e, 'returncode', None)}). See logs for details.") from e
        # except OSError as e:
        #     try:
        #         print(f"Failed to start finetuning subprocess: {e}")
        #     except Exception:
        #         pass
        #     raise RuntimeError("Failed to start finetuning subprocess") from e
        return output_dir
    
    def get_finetuning_progress(self, progress_file: Optional[str] = None) -> FinetuningProgress:
        # Read progress from file progress_file.

        # Note: this method will return (progress=-1, time_left=-1) if the file does not exist.
        # However, the there is a startup time between the call to finetune and to the time the 
        # model has started training. This means that during this time, there won't exist a progress 
        # file although finetuning has begun. The frontend may have to handle this special case.
        if progress_file is None:
            progress_file = str(self.data_root / "output" / "ft_progress.json")
        try:
            with open(progress_file, 'r') as f:
                import json
                data = json.load(f)
                progress = data.get("percent", -1)
                time_left = data.get("eta_seconds", -1)
                return FinetuningProgress(progress=progress, time_left=time_left)
        except Exception as e:
            # Log the error and fall back to default progress values.
            try:
                print(f"Failed to read finetuning progress from {progress_file}: {e}")
            except Exception:
                # If logging fails for any reason, ignore and continue to return defaults.
                pass
        return FinetuningProgress(progress=-1, time_left=-1)

    @contextlib.contextmanager
    def _force_cpu_load(self):
        from torch.serialization import load as _orig_load
        def _cpu_load(*args, **kwargs):
            kwargs["map_location"] = torch.device("cpu")
            return _orig_load(*args, **kwargs)
        old = torch.load
        torch.load = _cpu_load
        yield
        torch.load = old

    def merge_finetuned_checkpoint(self, ft_root=None):
        ft_root = Path(ft_root or (self.data_root / "models" / "checkpoints" / "chatterbox_finetuned_swedish"))
        ckpt_dirs = sorted(
            [p for p in ft_root.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")],
            key=lambda p: int(p.name.split("-")[1])
        )
        ft_ckpt = ckpt_dirs[-1] if ckpt_dirs else ft_root
        src_model = ft_ckpt / "model.safetensors"
        assert src_model.exists(), f"Missing checkpoint file: {src_model}"
        merged_dir = ft_root / "merged_for_infer"
        if merged_dir.exists():
            shutil.rmtree(merged_dir)
        merged_dir.mkdir(parents=True, exist_ok=True)
        copies = [
            ("ve.pt", "ve.pt"), ("s3gen.pt", "s3gen.pt"), ("conds.pt", "conds.pt"),
            ("Cangjie5_TC.json", "Cangjie5_TC.json"), ("mtl_tokenizer.json", "mtl_tokenizer.json"),
            ("tokenizer.json", "mtl_tokenizer.json"), ("grapheme_mtl_merged_expanded_v1.json", "mtl_tokenizer.json"),
            ("t3_mtl23ls_v2.safetensors", "t3_mtl23ls_v2.safetensors"), ("t3_23lang.safetensors", "t3_23lang.safetensors"),
        ]
        for src_name, dst_name in copies:
            src = self.base_dir / src_name
            if src.exists():
                shutil.copy2(src, merged_dir / dst_name)
        assert (merged_dir / "mtl_tokenizer.json").exists(), f"Tokenizer not found in base_dir: {self.base_dir}"
        t3_primary = None
        for cand in ("t3_mtl23ls_v2.safetensors", "t3_23lang.safetensors"):
            if (merged_dir / cand).exists():
                t3_primary = cand
                break
        assert t3_primary is not None, f"No base T3 file found in {self.base_dir}"
        sd = load_file(str(src_model))
        t3_sd = {k.split("t3.", 1)[1]: v for k, v in sd.items() if k.startswith("t3.")} if any(k.startswith("t3.") for k in sd) else dict(sd)
        assert any(k.startswith(("tfmr.layers.", "tfmr.embed_tokens")) for k in t3_sd), "Converted state dict doesn't look like T3 weights."
        save_file(t3_sd, str(merged_dir / t3_primary))
        alias = "t3_23lang.safetensors" if t3_primary == "t3_mtl23ls_v2.safetensors" else "t3_mtl23ls_v2.safetensors"
        save_file(t3_sd, str(merged_dir / alias))
        return merged_dir

    def load_models(self):
        self.base_model = self._load_chatterbox_model(str(self.base_dir))
        self.ft_model = self._load_chatterbox_model(str(self.ft_dir))
        return self.base_model, self.ft_model
    
    def unload_models(self):
        """Unload TTS models from memory to free VRAM."""
        print("Unloading TTS models...")
        if self.base_model is not None:
            del self.base_model
            self.base_model = None
        if self.ft_model is not None:
            del self.ft_model
            self.ft_model = None
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("TTS models unloaded and VRAM freed")

    def _load_chatterbox_model(self, model_folder, device=None):
        model_folder = Path(model_folder)
        device = device or self.device
        print(f"Loading model from {model_folder} on device={device}…")
        with self._force_cpu_load():
            tts = ChatterboxMultilingualTTS.from_local(model_folder, device)
        if device in ("cuda", "mps"):
            tts.t3.to(device)
            tts.s3gen.to(device)
            tts.ve.to(device)
            tts.device = device
        print("Model loaded.")
        return tts

    def create_voice_profile_from_wav(self, tts, wav_path, output_dir=None, name=None, exaggeration=0.5):
        wav_path = Path(wav_path)
        if not wav_path.exists():
            raise FileNotFoundError(f"Reference audio not found: {wav_path}")
        output_dir = Path(output_dir) if output_dir else (self.data_root / "models" / "voices")
        output_dir.mkdir(parents=True, exist_ok=True)
        name = name or wav_path.stem
        original_device = getattr(tts, "device", "cpu")
        if original_device == "mps":
            tts.t3.to("cpu")
            tts.s3gen.to("cpu")
            tts.ve.to("cpu")
            tts.device = "cpu"
        tts.prepare_conditionals(str(wav_path), exaggeration=exaggeration)
        conds = tts.conds
        if conds is None:
            if original_device == "mps":
                tts.t3.to(original_device)
                tts.s3gen.to(original_device)
                tts.ve.to(original_device)
                tts.device = original_device
            raise RuntimeError("prepare_conditionals() returned None — extraction failed.")
        out_path = output_dir / f"{name}.pt"
        conds.save(out_path)
        print(f"[voice] Saved voice profile to: {out_path}")
        if original_device == "mps":
            tts.t3.to("mps")
            tts.s3gen.to("mps")
            tts.ve.to("mps")
            tts.device = "mps"
        return out_path

    def apply_voice_profile_from_path(self, tts, conds_path):
        conds_path = Path(conds_path)
        if not conds_path.exists():
            raise FileNotFoundError(f"Voice profile not found: {conds_path}")
        device = getattr(tts, "device", "cpu")
        conds = Conditionals.load(str(conds_path), map_location=device).to(device)
        tts.conds = conds
        print(f"[voice] Applied voice profile from {conds_path} → device={device}")
        return tts

    def synthesize_with_model(self, tts, text, output_dir=None, language="sv", exaggeration=0.5, cfg_weight=0.5, filename=""):
        output_dir = Path(output_dir) if output_dir else (self.data_root / "output")
        output_dir.mkdir(parents=True, exist_ok=True)
        wav = self._generate_chunked(tts, text, language_id=language, exaggeration=exaggeration, cfg_weight=cfg_weight)
        if not filename:
            timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"{timestamp}_{language.lower()}.wav"
        else:
            if not filename.lower().endswith(".wav"):
                filename += ".wav"
        out_path = output_dir / filename
        sr = getattr(tts, "sr", getattr(tts, "sample_rate", 24000))
        ta.save(str(out_path), wav.cpu(), sr)
        print("Saved:", out_path)
        return wav, sr, out_path

    def _split_sentences(self, text):
        parts = re.split(r'([\.!?])', text)
        sents = ["".join(parts[i:i+2]).strip() for i in range(0, len(parts), 2)]
        return [s for s in sents if s]

    def _generate_chunked(self, tts, text, **gen_kwargs):
        wavs = []
        for s in self._split_sentences(text):
            w = tts.generate(s, **gen_kwargs)
            w = torch.as_tensor(w)
            if w.ndim == 1: w = w.unsqueeze(0)
            elif w.shape[0] > w.shape[-1]: w = w.t()
            wavs.append(w.to(torch.float32).contiguous().clamp_(-1,1).cpu())
        return torch.cat(wavs, dim=-1)

    def load_whisper_model(self, size="medium"):
        print(f"Loading Whisper model ({size})...")
        self.whisper_model = whisper.load_model(size)
        print("Whisper model loaded.")
        return self.whisper_model

    def transcribe_audio(self, audio_path, language="sv"):
        if self.whisper_model is None:
            self.load_whisper_model()
        result = self.whisper_model.transcribe(audio_path, language=language, task="transcribe") # type: ignore
        return result["text"]

    def normalize_text(self, s):
        s = s.lower()
        s = re.sub(r"[^\wåäöéüõçñ ]+", "", s, flags=re.UNICODE)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def cer(self, ref, hyp):
        ref_norm = self.normalize_text(ref)
        hyp_norm = self.normalize_text(hyp)
        return jiwer_cer(ref_norm, hyp_norm)

    def wer(self, ref, hyp):
        ref_norm = self.normalize_text(ref)
        hyp_norm = self.normalize_text(hyp)
        return jiwer_wer(ref_norm, hyp_norm)

    def evaluate(self, eval_sentences, base_model=None, ft_model=None, output_dir_base=None, output_dir_ft=None, language="sv"):
        base_model = base_model or self.base_model
        ft_model = ft_model or self.ft_model
        ft_dir = Path(output_dir_ft) if output_dir_ft else (self.data_root / "examples" / "eval_audio" / "finetuned")
        base_dir = Path(output_dir_base) if output_dir_base else (self.data_root / "examples" / "eval_audio" / "base")
        ft_dir.mkdir(parents=True, exist_ok=True)
        base_dir.mkdir(parents=True, exist_ok=True)
        base_scores_cer, ft_scores_cer = [], []
        base_scores_wer, ft_scores_wer = [], []

        # Create audio clips for finetuned.
        for i, text in enumerate(eval_sentences):
            finetune_filename = f"finetune_{i:03d}"
            self.synthesize_with_model(ft_model, text, output_dir=str(ft_dir), language=language, filename=finetune_filename)
        
        # Create audio clips for base.
        for i, text in enumerate(eval_sentences):
            base_filename = f"base_{i:03d}"
            self.synthesize_with_model(base_model, text, output_dir=str(base_dir), language=language, filename=base_filename)
        
        # Evaluate.
        for i, ref_text in enumerate(eval_sentences):
            finetune_filename = f"finetune_{i:03d}.wav"
            base_filename = f"base_{i:03d}.wav"
            base_path = base_dir / base_filename
            ft_path = ft_dir / finetune_filename
            base_transcribed = self.transcribe_audio(str(base_path))
            if isinstance(base_transcribed, list):
                base_transcribed = " ".join(str(x) for x in base_transcribed)
            ft_transcribed = self.transcribe_audio(str(ft_path))
            if isinstance(ft_transcribed, list):
                ft_transcribed = " ".join(str(x) for x in ft_transcribed)
            base_cer = self.cer(ref_text, base_transcribed)
            ft_cer = self.cer(ref_text, ft_transcribed)
            base_wer = self.wer(ref_text, base_transcribed)
            ft_wer = self.wer(ref_text, ft_transcribed)
            base_scores_cer.append(base_cer)
            ft_scores_cer.append(ft_cer)
            base_scores_wer.append(base_wer)
            ft_scores_wer.append(ft_wer)
            print(f"\n[{i:03d}] -----------------------------")
            print(f"REF:       {ref_text}")
            print(f"BASE:  {base_transcribed}")
            print(f"FT:    {ft_transcribed}")
            print(f"CER base:  {base_cer:.3f}")
            print(f"CER ft:    {ft_cer:.3f}")
            print(f"WER base:  {base_wer:.3f}")
            print(f"WER ft:    {ft_wer:.3f}")
        base_cer_avg = float(np.mean(base_scores_cer))
        ft_cer_avg = float(np.mean(ft_scores_cer))
        base_wer_avg = float(np.mean(base_scores_wer))
        ft_wer_avg = float(np.mean(ft_scores_wer))
        print("\n===== CER and WER SUMMARY =====")
        print(f"Base CER: {base_cer_avg:.3f}")
        print(f"Finetuned CER: {ft_cer_avg:.3f}")
        print(f"Base WER: {base_wer_avg:.3f}")
        print(f"Finetuned WER: {ft_wer_avg:.3f}")
        return {
            "base_cer_avg": base_cer_avg,
            "ft_cer_avg": ft_cer_avg,
            "base_wer_avg": base_wer_avg,
            "ft_wer_avg": ft_wer_avg,
        }
    

    def evaluate_existing_audio(self, eval_sentences, base_dir, ft_dir, language="sv"):
        """
        Evaluate CER and WER using already existing audio files in base_dir and ft_dir.
        Assumes files are named base_000.wav, base_001.wav, ... and finetune_000.wav, finetune_001.wav, ...
        Returns an EvaluationResult dataclass.
        """
        base_dir = Path(base_dir)
        ft_dir = Path(ft_dir)
        base_scores_cer, ft_scores_cer = [], []
        base_scores_wer, ft_scores_wer = [], []
        sentence_results = []

        for i, ref_text in enumerate(eval_sentences):
            finetune_filename = f"finetune_{i:03d}.wav"
            base_filename = f"base_{i:03d}.wav"
            base_path = base_dir / base_filename
            ft_path = ft_dir / finetune_filename
            base_transcribed = self.transcribe_audio(str(base_path))
            ft_transcribed = self.transcribe_audio(str(ft_path))
            base_cer = self.cer(ref_text, base_transcribed)
            ft_cer = self.cer(ref_text, ft_transcribed)
            base_wer = self.wer(ref_text, base_transcribed)
            ft_wer = self.wer(ref_text, ft_transcribed)
            base_scores_cer.append(base_cer)
            ft_scores_cer.append(ft_cer)
            base_scores_wer.append(base_wer)
            ft_scores_wer.append(ft_wer)
            sentence_results.append(EvaluationSentenceResult(
                index=i,
                ref_text=str(ref_text),
                base_transcribed=str(base_transcribed),
                ft_transcribed=str(ft_transcribed),
                base_cer=base_cer,
                ft_cer=ft_cer,
                base_wer=base_wer,
                ft_wer=ft_wer
            ))
        base_cer_avg = float(np.mean(base_scores_cer))
        ft_cer_avg = float(np.mean(ft_scores_cer))
        base_wer_avg = float(np.mean(base_scores_wer))
        ft_wer_avg = float(np.mean(ft_scores_wer))
        return EvaluationResult(
            sentence_results=sentence_results,
            base_cer_avg=base_cer_avg,
            ft_cer_avg=ft_cer_avg,
            base_wer_avg=base_wer_avg,
            ft_wer_avg=ft_wer_avg
        )

    

if __name__ == "__main__":
    service = TTSService()
    base_ckpt_dir = service.download_base_model()

    # Finetune model.
    service.finetune(
        metadata_file="../chatterbox/data/david/metadata.txt",
        dataset_dir="../chatterbox/data/david",
        output_dir="./models/checkpoints/chatterbox_finetuned_swedish"
    )
    merged_dir = service.merge_finetuned_checkpoint()
    print(f"Finetuning complete. Merged model to {merged_dir}")

    # Load models
    base_model, ft_model = service.load_models()

    ###########################################################################
    # # Create voice profile from wav
    # voice_profile_path = service.create_voice_profile_from_wav(
    #     tts=ft_model,
    #     wav_path="../chatterbox/data/david/soundfiles/david_sentence25.wav",
    #     name="david"
    # )

    # # Apply voice profile
    # service.apply_voice_profile_from_path(ft_model, voice_profile_path)

    # # Synthesize audio
    # wav, sr, out_path = service.synthesize_with_model(
    #     tts=ft_model,
    #     text="Hej, det här är ett test av TTS",
    #     output_dir="./output",
    #     language="sv",
    #     filename="test_output.wav"
    # )

    ########################################################################
    # Evaluate models
    # eval_sentences = [
    #     "Det här är ett test.",
    #     "Idag ska vi utvärdera svensk talsyntes."
    # ]
    # results = service.evaluate(eval_sentences)
    # print(results)

