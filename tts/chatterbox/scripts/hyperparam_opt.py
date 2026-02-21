import os
import logging
from pathlib import Path

import numpy as np
import torch
import optuna
from transformers import Trainer

from finetune_mtl import (
    patched_from_local,
    CustomTrainingArguments,
    ModelArguments,
    SpeechDataCollator,
    T3ForFineTuning,
)
from finetune_mtl_precomputed import PrecomputedT3Dataset

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

# Optimizatino  constants
BASE_MODEL_DIR = Path("../models/base/chatterbox")
PRECOMPUTED_PATH = Path("../data/david/precomputed_t3_features.pt")
TRIAL_ROOT = Path("../models/hparam_trials")

EVAL_SPLIT = 0.1
BATCH_SIZE = 2
GRAD_ACCUM = 4
# May want to increase this, but 5 epochs should show the general trend
# for the current hyperparameters.
MAX_EPOCHS_PER_TRIAL = 5  
NO_OF_TRIALS = 30
SEED = 42

has_cuda = torch.cuda.is_available()
has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
no_cuda = not has_cuda if not has_mps else True

logger.info(f"CUDA available: {has_cuda}, MPS available: {has_mps}, using no_cuda={no_cuda}")

def build_model_and_data(seed: int):
    if not BASE_MODEL_DIR.exists():
        raise FileNotFoundError(f"Missing: {BASE_MODEL_DIR}")
    if not PRECOMPUTED_PATH.exists():
        raise FileNotFoundError(f"Missing: {PRECOMPUTED_PATH}")

    logger.info(f"[Trial Setup] Loading base model from {BASE_MODEL_DIR}")
    mtl_model = patched_from_local(str(BASE_MODEL_DIR), device="cpu")
    t3_model = mtl_model.t3
    t3_config = t3_model.hp

    # Freeze all but T3
    for p in mtl_model.ve.parameters(): p.requires_grad = False
    for p in mtl_model.s3gen.parameters(): p.requires_grad = False
    for p in t3_model.parameters(): p.requires_grad = True

    logger.info("[Trial Setup] VE & S3Gen frozen, T3 trainable")

    all_data = torch.load(PRECOMPUTED_PATH, weights_only=False)
    if not isinstance(all_data, list):
        raise ValueError("Expected list of precomputed examples")

    n_total = len(all_data)
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n_total)
    n_eval = max(1, int(n_total * EVAL_SPLIT))

    train_data = [all_data[i] for i in indices[n_eval:]]
    eval_data = [all_data[i] for i in indices[:n_eval]]

    train_dataset = PrecomputedT3Dataset(train_data)
    eval_dataset = PrecomputedT3Dataset(eval_data)

    data_collator = SpeechDataCollator(
        t3_config,
        t3_config.stop_text_token,
        t3_config.stop_speech_token,
    )

    hf_model = T3ForFineTuning(t3_model, t3_config)
    return hf_model, train_dataset, eval_dataset, data_collator

def run_finetune_trial(trial, hparams):
    hf_model, train_dataset, eval_dataset, data_collator = build_model_and_data(SEED)

    TRIAL_ROOT.mkdir(parents=True, exist_ok=True)
    trial_output_dir = TRIAL_ROOT / f"trial_{trial.number:04d}"
    trial_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"[Trial {trial.number}] Output at: {trial_output_dir}")
    logger.info(f"[Trial {trial.number}] Params: {hparams}")

    training_args = CustomTrainingArguments(
        output_dir=str(trial_output_dir),
        learning_rate=hparams["learning_rate"],
        weight_decay=hparams["weight_decay"],
        warmup_ratio=hparams["warmup_ratio"],
        max_grad_norm=hparams["max_grad_norm"],
        label_smoothing_factor=hparams["label_smoothing_factor"],
        lr_scheduler_type=hparams["lr_scheduler_type"],
        num_train_epochs=hparams.get("num_train_epochs", MAX_EPOCHS_PER_TRIAL),
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        evaluation_strategy="epoch",
        save_strategy="no",
        logging_steps=10,
        report_to="none",
        seed=SEED,
        no_cuda=no_cuda,
    )

    trainer = Trainer(
        model=hf_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    metrics = trainer.evaluate()
    loss = metrics.get("eval_loss", None)

    if loss is None:
        raise RuntimeError("Eval loss missing from metrics")

    logger.info(f"[Trial {trial.number}] Eval loss: {loss:.4f}")

    del trainer, hf_model
    torch.cuda.empty_cache()
    return loss

def objective(trial: optuna.Trial):
    hparams = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 5e-5, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.05),
        "warmup_ratio": trial.suggest_categorical("warmup_ratio", [0.01, 0.05, 0.1]),
        "max_grad_norm": trial.suggest_categorical("max_grad_norm", [0.5, 1.0, 2.0]),
        "label_smoothing_factor": trial.suggest_categorical("label_smoothing_factor", [0.0, 0.05, 0.1]),
        "lr_scheduler_type": trial.suggest_categorical("lr_scheduler_type", ["cosine", "linear"]),
        "num_train_epochs": MAX_EPOCHS_PER_TRIAL,  # or trial.suggest_int("epochs", 2, 4) if you'd like
    }
    return run_finetune_trial(trial, hparams)

if __name__ == "__main__":
    study = optuna.create_study(
        study_name="chatterbox_hparam_opt_final",
        direction="minimize",
        storage="sqlite:///optuna_hparams_final.db",
        load_if_exists=True,
    )

    logger.info("Starting hyperparameter search…")
    study.optimize(objective, n_trials=NO_OF_TRIALS, show_progress_bar=True)

    logger.info(f"Best Loss: {study.best_value}")
    logger.info(f"Best Params: {study.best_params}")
