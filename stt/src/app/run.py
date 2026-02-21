import argparse
import ast
import os
import torch
from typing import Dict, Any, List
from unsloth import FastModel

from model_setup import ModelSetup
from parameter_optimizer import ParameterOptimizer
from data_collator import DataCollatorSpeechSeq2SeqWithPadding

from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoProcessor


class STTService:
    """Service wrapper around training/evaluation utilities so the API can import
    and call methods on an instance.

    Example usage from API:
        svc = STTService(setup=ModelSetup())
        svc.train_with_best()
    """

    def __init__(self, setup: ModelSetup = None, device: str = None):
        self.setup = setup or ModelSetup()
        self.model = getattr(self.setup, "model", None)
        self.tokenizer = getattr(self.setup, "tokenizer", None)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def load_params_from_file(path: str) -> Dict[str, Any]:
        if not os.path.exists(path):
            return {}
        txt = open(path, "r", encoding="utf-8").read()
        start = txt.find("{")
        end = txt.rfind("}") + 1
        if start == -1 or end == 0:
            return {}
        try:
            return ast.literal_eval(txt[start:end])
        except Exception:
            return {}

    def train_with_best(self, best_params_path: str = "optuna_best_params.txt") -> str:
        """Train the model using best params (or defaults) and save model.

        Returns path to saved model directory.
        """
        train_ds, val_ds, _ = self.setup.get_formatted_datasets()
        model, tokenizer = self.setup.model, self.setup.tokenizer

        params = STTService.load_params_from_file(best_params_path)

        collator = DataCollatorSpeechSeq2SeqWithPadding(processor=tokenizer)

        training_args = Seq2SeqTrainingArguments(
            output_dir=params.get("output_dir", "outputs_best_run"),
            per_device_train_batch_size=int(params.get("per_device_train_batch_size", 1)),
            per_device_eval_batch_size=int(params.get("per_device_eval_batch_size", 1)),
            gradient_accumulation_steps=int(params.get("gradient_accumulation_steps", 8)),
            num_train_epochs=int(params.get("num_train_epochs", 10)),
            learning_rate=float(params.get("learning_rate", 5e-5)),
            fp16=False,
            bf16=True,
            remove_unused_columns=False,
            label_names=["labels"],
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=collator,
            tokenizer=tokenizer,
        )

        trainer.train()
        outdir = "saved_model_best_lora"
        trainer.save_model(outdir)
        try:
            tokenizer.save_pretrained(outdir)
        except Exception:
            # tokenizer object from FastModel may differ; ignore if unavailable
            pass
        print(f"Saved model to {outdir}")
        return outdir

    def run_optuna(self, n_trials: int = 1) -> Dict[str, Any]:
        train_ds, val_ds, _ = self.setup.get_formatted_datasets()
        opt = ParameterOptimizer(self.setup, train_ds, val_ds, self.setup.tokenizer, device=self.device)
        best_params, best_value = opt.run(n_trials=n_trials)
        print("Optuna done:", best_params, best_value)
        return {"best_params": best_params, "best_value": best_value}

    def evaluate_saved(self) -> List[Dict[str, Any]]:
        """Evaluate a saved model (looks for `saved_model_best_lora`) and write mismatches CSV.

        Returns the list of result dicts.
        """
        _, _, test_ds = self.setup.get_formatted_datasets()
        model, tokenizer = self.setup.model, self.setup.tokenizer

        proc = None
        if os.path.exists("saved_model_best_lora"):
            try:
                proc = AutoProcessor.from_pretrained("saved_model_best_lora")
                model.config.forced_decoder_ids = proc.get_decoder_prompt_ids(language="sv", task="transcribe")
            except Exception:
                proc = None

        tok_for_decode = proc.tokenizer if proc is not None else (tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer)

        import pandas as pd
        import re

        def normalize_text(s: str) -> str:
            return re.sub(r"\s+", " ", s.strip().lower())

        results = []
        model.eval()
        model.to(self.device)
        for i, sample in enumerate(test_ds):
            dtype = getattr(model, "dtype", torch.float32)
            feats = torch.tensor(sample["input_features"], dtype=dtype).unsqueeze(0).to(model.device)
            with torch.no_grad():
                pred_ids = model.generate(feats, do_sample=False, max_new_tokens=256, language="sv", task="transcribe")
            pred_text = tok_for_decode.decode(pred_ids[0], skip_special_tokens=True)
            true_text = tok_for_decode.decode(sample["labels"], skip_special_tokens=True)
            filename = self.setup.raw_dataset["test"][i].get("filename", f"sample_{i}.wav")
            results.append({
                "filename": filename,
                "correct": true_text,
                "transcribed": pred_text,
                "correct_norm": normalize_text(true_text),
                "transcribed_norm": normalize_text(pred_text),
            })

        df = pd.DataFrame(results)
        df_incorrect = df[df["correct_norm"] != df["transcribed_norm"]][["filename", "correct", "transcribed"]].reset_index(drop=True)
        df_incorrect.to_csv("transcription_mismatches.csv", index=False, encoding="utf-8")
        print(f"Saved mismatches to transcription_mismatches.csv ({len(df_incorrect)} rows)")
        return results

    def transcribe_audio(self, audio_path: str) -> str:
        """Transcribe a single audio file."""
        import librosa
        import numpy as np
        
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Load and resample
        y, sr = librosa.load(audio_path, sr=16000)
        
        # Process
        processor = self.tokenizer
        
        inputs = processor(y, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(self.device)
        
        # Enable native 2x faster inference for Unsloth 4bit models
        FastModel.for_inference(self.model)
        if hasattr(self.model, "get_base_model"):
             FastModel.for_inference(self.model.get_base_model())
        
        self.model.eval()
        
        with torch.no_grad():
            # Use autocast to ensure compatibility with 4-bit quantization
            with torch.cuda.amp.autocast(dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
                generated_ids = self.model.generate(input_features, language="sv", task="transcribe")
            
        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return transcription


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", "-m", choices=["optuna", "train", "eval"], required=False, default="train",
                   help="Action to run: optuna (search), train (train with best params/defaults), eval (evaluate)")
    p.add_argument("--n_trials", type=int, default=1)
    args = p.parse_args()

    svc = STTService()

    if args.mode == "optuna":
        svc.run_optuna(n_trials=args.n_trials)
    elif args.mode == "train":
        svc.train_with_best()
    else:
        svc.evaluate_saved()


if __name__ == "__main__":
    main()