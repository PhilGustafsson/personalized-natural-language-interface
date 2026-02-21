from unsloth import FastModel
from unsloth import is_bf16_supported
import ast
import os
import gc
import torch
from typing import Dict, Any, List

import optuna
from optuna.pruners import MedianPruner, HyperbandPruner
from transformers import (
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    AutoProcessor,
    pipeline,
    WhisperForConditionalGeneration,
    TrainerCallback,
)
from data_collator import DataCollatorSpeechSeq2SeqWithPadding
import time
import traceback
from peft import PeftConfig
import jiwer
import numpy as np
import shutil
from optuna_earlystopping import OptunaEarlyStoppingCallback

class ProgressCallback(TrainerCallback):
    def __init__(self, progress_callback):
        self.progress_callback = progress_callback

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.max_steps > 0:
            progress = (state.global_step / state.max_steps) * 100
            if self.progress_callback:
                self.progress_callback(progress)

class STTService:
    def __init__(self, model_setup, dataset=None):
        self.model_setup = model_setup
        self.dataset = dataset
        self.tokenizer = model_setup.tokenizer
        self.processor = model_setup.processor

    def compute_metrics(self, pred):
        """
        Calculates WER using jiwer.
        Handles both:
        - predict_with_generate=True: predictions are token IDs (2D)
        - predict_with_generate=False: predictions are logits (3D)
        """
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]
        
        if hasattr(pred_ids, 'cpu'):
            pred_ids = pred_ids.cpu().numpy()
        
        if len(pred_ids.shape) == 3:
            pred_ids = np.argmax(pred_ids, axis=-1)

        label_ids = np.where(label_ids == -100, self.tokenizer.pad_token_id, label_ids)

        pred_str = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * jiwer.wer(reference=label_str, hypothesis=pred_str)

        return {"wer": wer}

    @staticmethod
    def load_params_from_file(path: str) -> Dict[str, Any]:
        if not os.path.exists(path): return {}
        try:
            txt = open(path, "r", encoding="utf-8").read()
            start, end = txt.find("{"), txt.rfind("}") + 1
            return ast.literal_eval(txt[start:end]) if start != -1 else {}
        except Exception: return {}

    def _get_fresh_model(self, seed):
        """
        Creates a dedicated Unsloth model for TRAINING only.
        """
        # 1. Determine Base Path
        target_model_path = self.model_setup.model_dir
        if os.path.exists(os.path.join(target_model_path, "adapter_config.json")):
            conf = PeftConfig.from_pretrained(target_model_path)
            target_model_path = conf.base_model_name_or_path

        print(f"Initializing Unsloth Training Model from: {target_model_path}")

        # 2. Load with Unsloth (Fast!)
        model, _ = FastModel.from_pretrained(
            model_name=target_model_path, 
            load_in_4bit=False, 
            dtype=None,
            auto_model=WhisperForConditionalGeneration,
            whisper_language=self.model_setup.whisper_language,
            whisper_task="transcribe",
        )

        # 3. Resize Logic (Still needed for training setup)
        if hasattr(self.tokenizer, "tokenizer"):
            vocab_size = len(self.tokenizer.tokenizer)
        else:
            vocab_size = len(self.tokenizer)
            
        if model.get_input_embeddings().weight.shape[0] != vocab_size:
            model.resize_token_embeddings(vocab_size)

        # 4. Config & PEFT Setup
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        if hasattr(self.tokenizer, "tokenizer"):
            pad_id = self.tokenizer.tokenizer.pad_token_id
        else:
            pad_id = self.tokenizer.pad_token_id
            
        model.config.pad_token_id = pad_id
        model.config.suppress_tokens = []
        model.config.forced_decoder_ids = None
        model.generation_config.forced_decoder_ids = None

        model = FastModel.get_peft_model(
            model,
            r=64, 
            target_modules=["q_proj", "v_proj"],
            lora_alpha=64, 
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth", 
            random_state=seed,
            use_rslora=False, 
            loftq_config=None, 
            task_type=None, 
        )
                    
        return model

    def run_optuna(
        self,
        user: str,
        n_trials: int,
        lr_range: List[float],
        epochs_range: List[int],
        weight_decay_range: List[float],
        warmup_range: List[float],
        grad_norm_range: List[float],
        train_batch_size_range: List[int],
        eval_batch_size_range: List[int],
        grad_accum_range: List[int],
        label_smoothing_range: List[float],
        lr_scheduler_choices: List[str],
        optimizer_choices: List[str],
        seed: int,
        pruning_warmup_trials: int, 
        pruning_warmup_epochs: int, 
        max_wer_threshold: float,
        patience: int,
        progress_callback=None,
    ):
        print(f"Starting Optuna for {n_trials} trials...")

        self.model_setup.unload_model()

        def objective(trial):
            if progress_callback:
                prog = (trial.number / n_trials) * 100
                progress_callback(prog)

            lr = trial.suggest_float("learning_rate", lr_range[0], lr_range[1], log=True)
            weight_decay = trial.suggest_float("weight_decay", weight_decay_range[0], weight_decay_range[1])
            warmup_ratio = trial.suggest_float("warmup_ratio", warmup_range[0], warmup_range[1])
            max_grad_norm = trial.suggest_float("max_grad_norm", grad_norm_range[0], grad_norm_range[1])
            label_smoothing = trial.suggest_float("label_smoothing_factor", label_smoothing_range[0], label_smoothing_range[1])
            
            num_epochs = trial.suggest_int("num_train_epochs", epochs_range[0], epochs_range[1])
            train_bs = trial.suggest_int("per_device_train_batch_size", train_batch_size_range[0], train_batch_size_range[1])
            eval_bs = trial.suggest_int("per_device_eval_batch_size", eval_batch_size_range[0], eval_batch_size_range[1])
            grad_accum = trial.suggest_int("gradient_accumulation_steps", grad_accum_range[0], grad_accum_range[1])
            lr_scheduler = trial.suggest_categorical("lr_scheduler_type", lr_scheduler_choices)
            optimizer = trial.suggest_categorical("optimizer", optimizer_choices)
            
            # Reload processor BEFORE creating the model
            self.processor = AutoProcessor.from_pretrained(self.model_setup.model_dir)
            self.tokenizer = self.processor.tokenizer
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            trial_model = self._get_fresh_model(seed)
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            args = Seq2SeqTrainingArguments(
                output_dir=f"optuna_outputs/trial_{trial.number}",
                learning_rate=lr,
                weight_decay=weight_decay,
                warmup_ratio=warmup_ratio,
                num_train_epochs=num_epochs,
                per_device_train_batch_size=train_bs,
                per_device_eval_batch_size=eval_bs,
                gradient_accumulation_steps=grad_accum,
                max_grad_norm=max_grad_norm,
                label_smoothing_factor=label_smoothing,
                lr_scheduler_type=lr_scheduler,
                seed=seed,
                optim=optimizer,
            
                dataloader_num_workers=4,
                dataloader_pin_memory=True,
                
                greater_is_better=False, # WER: lower is better
                metric_for_best_model="wer",  # Track WER to find best model
                # load_best_model_at_end=True,  # Load best model when training ends
                
                fp16=not is_bf16_supported(), 
                bf16=is_bf16_supported(), # b16 is faster if supported
                
                eval_strategy="epoch",
                # save_strategy="epoch",  # Don't save checkpoints
                logging_strategy="epoch",
                # save_total_limit=1,  # Keep only the best checkpoint
                
                disable_tqdm=True,
                push_to_hub=False,

                generation_max_length=225,
                predict_with_generate=True,  # Needed to compute WER
                remove_unused_columns=False,
                label_names=["labels"],
                gradient_checkpointing=True,
            )
            
            pruning_callback = OptunaEarlyStoppingCallback(
                trial=trial,
                metric_name="eval_wer",
                max_wer_threshold=max_wer_threshold,  # Stop if WER > 150%
                patience=patience,                # Stop if WER increases 2x in a row
            )
                
            trainer = Seq2SeqTrainer(
                model=trial_model,
                args=args,
                train_dataset=self.dataset.ds_dict["train"], 
                eval_dataset=self.dataset.ds_dict["validation"],
                data_collator=DataCollatorSpeechSeq2SeqWithPadding(
                    processor=self.processor,
                    model_dtype=self.model_setup.torch_dtype  
                ),
                tokenizer=self.tokenizer,
                compute_metrics=self.compute_metrics,
                callbacks=[pruning_callback], 
            )
            
            try:
                trainer.train()
                metrics = trainer.evaluate()
                result = metrics.get("eval_wer", float("inf"))
            except optuna.TrialPruned:
                # Re-raise to let Optuna handle it properly
                raise
            except Exception as e:
                print(f"Trial {trial.number} failed: {e}")
                result = float("inf")
            finally:
                # Cleanup after each trial
                del trial_model
                del trainer
                gc.collect()
                torch.cuda.empty_cache()
            
            return result

        study_name = f"whisper_finetune_{user}_{int(time.time())}"
        pruner = MedianPruner(
            n_startup_trials=pruning_warmup_trials,  # Don't prune first N trials
            n_warmup_steps=pruning_warmup_epochs,    # Don't prune first N epochs
            interval_steps=1,                         # Check every epoch
        )
        study = optuna.create_study(
            direction="minimize", 
            study_name=study_name,
            pruner=pruner
        )
        study.optimize(objective, n_trials=n_trials)
        
        best_params = study.best_params
        best_params["user"] = user
        print("Optuna Best Params:", best_params)
        
        ts = int(time.time())
        path = f"optuna_best_params_{user}_{ts}.txt"
        with open(path, "w") as f:
            f.write(str(best_params))
            
        return {"best_params": best_params, "best_value": study.best_value, "params_path": path}

    def train_with_best(self, best_params_path: str = "optuna_best_params.txt", progress_callback=None):
        print(f"Starting final training with best params from {best_params_path}...", flush=True)

        self.model_setup.unload_model()
        
        if os.path.exists(best_params_path):
            params = self.load_params_from_file(best_params_path)
        else:
            params = {}

        model_to_train = self._get_fresh_model(seed=int(params.get("seed", 1337)))
        
        outdir = params.get("saved_model_dir", "finetuned_models")
        # Get user from params (profile username/folder name)
        user = params.get("user", "unknown_user")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        unique_folder_name = f"{user}_{timestamp}"
        full_outdir = os.path.join(outdir, unique_folder_name)

        training_args = Seq2SeqTrainingArguments(
            output_dir=full_outdir,
            warmup_ratio=float(params.get("warmup_ratio", 0.3)),
            learning_rate=float(params.get("learning_rate", 5e-5)),
            num_train_epochs=int(params.get("num_train_epochs", 4)),
            weight_decay=float(params.get("weight_decay", 0.01)),
            per_device_train_batch_size=int(params.get("per_device_train_batch_size", 16)),
            per_device_eval_batch_size=int(params.get("per_device_eval_batch_size", 8)),
            max_grad_norm=float(params.get("max_grad_norm", 1.0)),
            label_smoothing_factor=float(params.get("label_smoothing_factor", 0.0)),
            lr_scheduler_type=params.get("lr_scheduler_type", "cosine"),
            gradient_accumulation_steps=int(params.get("gradient_accumulation_steps", 1)),
            optim=params.get("optimizer", "adamw_8bit"),
            seed=int(params.get("seed", 1337)),
            
            dataloader_num_workers=4,
            dataloader_pin_memory=True,
            
            greater_is_better=False, # WER: lower is better
            metric_for_best_model="wer",  # Track WER to find best model
            load_best_model_at_end=True,  # Load best model when training ends
            
            fp16=not is_bf16_supported(), 
            bf16=is_bf16_supported(), # b16 is faster if supported
            
            eval_strategy="epoch",
            save_strategy="epoch",  # Don't save checkpoints
            logging_strategy="epoch",
            save_total_limit=1,  # Keep only the best checkpoint
            
            disable_tqdm=True,
            push_to_hub=False,

            generation_max_length=225,
            predict_with_generate=True, 
            remove_unused_columns=False,
            label_names=["labels"],
            gradient_checkpointing=True,
        )
            
        print("Preparing Trainer...", flush=True)
        
        callbacks = []
        if progress_callback:
            callbacks.append(ProgressCallback(progress_callback))

        trainer = Seq2SeqTrainer(
            model=model_to_train,
            args=training_args,
            train_dataset=self.dataset.ds_dict["train"], 
            eval_dataset=self.dataset.ds_dict["validation"],
            data_collator=DataCollatorSpeechSeq2SeqWithPadding(
                processor=self.processor,
                model_dtype=self.model_setup.torch_dtype  
            ),
            tokenizer=self.tokenizer, 
            compute_metrics=self.compute_metrics, 
            callbacks=callbacks
        )

        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        print(f"{start_gpu_memory} GB of memory reserved.")

        print("Starting Final Training...", flush=True)
        try:
            trainer_stats = trainer.train()
            print("Training finished.", flush=True)
        except Exception as e:
            traceback.print_exc()

        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
        used_percentage = round(used_memory / max_memory * 100, 3)
        lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
        print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
        print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
        print(f"Peak reserved memory = {used_memory} GB.")
        print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
        print(f"Peak reserved memory % of max memory = {used_percentage} %.")
        print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

        print("Final Training Completed.", flush=True)
        
        # Evaluate on validation set to get final WER
        print("Evaluating final model...", flush=True)
        try:
            eval_results = trainer.evaluate(eval_dataset=self.dataset.ds_dict["validation"])
            final_wer = float(eval_results.get("eval_wer", float("inf")))
            print(f"Final validation WER: {final_wer}", flush=True)
        except Exception as e:
            print(f"Final evaluation failed: {e}", flush=True)
            final_wer = float("inf")
        
        print(f"Saving final model to {full_outdir}...", flush=True)
        model_to_train.save_pretrained(full_outdir)
        self.tokenizer.save_pretrained(full_outdir)
        
        for item in os.listdir(full_outdir):
            item_path = os.path.join(full_outdir, item)
            if os.path.isdir(item_path) and item.startswith("checkpoint-"):
                shutil.rmtree(item_path)
                print(f"Removed checkpoint: {item_path}")
        
        # Cleanup   
        gc.collect()
        torch.cuda.empty_cache()
        
        return {"model_dir": full_outdir, "final_wer": final_wer}
    

    def transcribe(self, audio_path: str, language: str = "sv") -> str:
        """
        Clean Transcribe Function using the Standard Model from ModelSetup.
        """
        if not os.path.isfile(audio_path): 
            raise FileNotFoundError(f"{audio_path}")
        
        # Reload if it was unloaded for training
        if self.model_setup.model is None:
            self.model_setup.load_model()

        print(f"Transcribing: {audio_path}", flush=True)

        pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model_setup.model,
            tokenizer=self.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=self.model_setup.torch_dtype, 
            chunk_length_s=30, 
            stride_length_s=5,
            return_timestamps=True,
            device=0 if torch.cuda.is_available() else -1
        )
        
        result = pipe(
            audio_path, 
            batch_size=8, 
            generate_kwargs={"language": language, "task": "transcribe"}
        )
        return result

    def evaluate_saved(self, eval_split: str = "test", per_device_eval_batch_size: int = 64) -> Dict[str, Any]:
        print(f"Starting evaluation on '{eval_split}' split...", flush=True)
        
        if self.dataset is None:
            raise ValueError("No dataset loaded. Please provide a dataset.")
        
        if eval_split not in self.dataset.ds_dict:
            raise ValueError(f"Split '{eval_split}' not found. Available: {list(self.dataset.ds_dict.keys())}")
        
        if self.model_setup.model is None:
            print("Model was unloaded, reloading...", flush=True)
            self.model_setup.load_model()
            
            
        eval_args = Seq2SeqTrainingArguments(
            output_dir="eval_output",
            per_device_eval_batch_size=per_device_eval_batch_size,
            
            dataloader_num_workers=4,
            dataloader_pin_memory=True,
            
            greater_is_better=False, # WER: lower is better
            
            fp16=not is_bf16_supported(), 
            bf16=is_bf16_supported(), # b16 is faster if supported
            
            eval_strategy="epoch",
            save_strategy="no",  # Don't save checkpoints
            logging_strategy="epoch",
            
            disable_tqdm=True,
            push_to_hub=False,

            generation_max_length=225,
            predict_with_generate=True,  # Much faster - compute_metrics handles logits
            remove_unused_columns=False,
            label_names=["labels"],
            gradient_checkpointing=True,
        )

        trainer = Seq2SeqTrainer(
            model=self.model_setup.model, 
            args=eval_args,
            eval_dataset=self.dataset.ds_dict[eval_split],
            data_collator=DataCollatorSpeechSeq2SeqWithPadding(
                processor=self.processor,
                model_dtype=self.model_setup.torch_dtype  # Pass the model's dtype
            ),
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
        )

        try:
            eval_results = trainer.evaluate()
            print("Evaluation metrics:", eval_results, flush=True)
            
            return {
                "wer": float(eval_results.get("eval_wer", float("inf"))),
                "loss": float(eval_results.get("eval_loss", float("inf"))),
                "eval_split": eval_split,
                "num_samples": len(self.dataset.ds_dict[eval_split]),
                "runtime": eval_results.get("eval_runtime", 0),
            }
        except Exception as e:
            print(f"Evaluation failed: {e}", flush=True)
            traceback.print_exc()
            return {"error": str(e), "wer": float("inf")}