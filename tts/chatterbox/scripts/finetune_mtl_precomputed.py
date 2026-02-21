import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from transformers import (
    HfArgumentParser,
    Trainer,
    EarlyStoppingCallback,
    set_seed,
    PretrainedConfig,
)
from transformers import TrainingArguments as HfTrainingArguments

from finetune_mtl import (
    patched_from_local,  # your CPU-safe loader
    CustomTrainingArguments,  # includes language_id, early_stopping_patience
    ModelArguments,
    SpeechDataCollator,
    T3ForFineTuning,
)  # this import will also patch T3.loss via finetune_mtl

logger = logging.getLogger(__name__)


# ---- New DataArguments for precomputed data ----
@dataclass
class PrecomputedDataArguments:
    precomputed_path: str = field(
        metadata={"help": "Path to .pt file with precomputed features (list of dicts)."}
    )
    eval_split_size: float = field(
        default=0.1,
        metadata={"help": "Fraction of data to use for eval split."},
    )


# ---- Dataset that wraps precomputed features ----
class PrecomputedT3Dataset(Dataset):
    def __init__(self, data_list: List[Dict[str, Any]]):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict[str, Any]:
        ex = self.data[idx]

        # convert numpy arrays / lists to tensors
        text_tokens = torch.as_tensor(ex["text_tokens"], dtype=torch.long)
        speech_tokens = torch.as_tensor(ex["speech_tokens"], dtype=torch.long)
        speaker_emb = torch.as_tensor(ex["t3_cond_speaker_emb"], dtype=torch.float)
        cond_prompt = torch.as_tensor(
            ex["t3_cond_prompt_speech_tokens"], dtype=torch.long
        )
        emotion_scalar = torch.tensor(
            ex.get("emotion_adv_scalar", 0.5), dtype=torch.float
        )

        return {
            "text_tokens": text_tokens,
            "text_token_lens": torch.tensor(len(text_tokens), dtype=torch.long),
            "speech_tokens": speech_tokens,
            "speech_token_lens": torch.tensor(len(speech_tokens), dtype=torch.long),
            "t3_cond_speaker_emb": speaker_emb,
            "t3_cond_prompt_speech_tokens": cond_prompt,
            "t3_cond_emotion_adv": emotion_scalar,
        }


def main():
    parser = HfArgumentParser(
        (ModelArguments, PrecomputedDataArguments, CustomTrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Model parameters %s", model_args)
    logger.info("Data parameters %s", data_args)

    set_seed(training_args.seed)

    # language_id is technically not needed now (data is already tokenized),
    # but we keep it for logging / consistency
    language_id = getattr(training_args, "language_id", "en")
    logger.info(f"Using language_id: {language_id}")

    # ---- Load base Chatterbox model (T3 + others) ----
    if not model_args.local_model_dir:
        raise ValueError(
            "You must provide --local_model_dir pointing to the base Chatterbox checkpoint."
        )

    local_dir_path = Path(model_args.local_model_dir)
    logger.info(f"Loading base model from local directory: {local_dir_path}")
    mtl_model = patched_from_local(str(local_dir_path), device="cpu")

    t3_model = mtl_model.t3
    chatterbox_t3_config_instance = t3_model.hp

    # Freeze VE + S3GEN as before; only T3 is trainable
    if model_args.freeze_voice_encoder:
        for p in mtl_model.ve.parameters():
            p.requires_grad = False
        logger.info("Voice Encoder frozen.")

    if model_args.freeze_s3gen:
        for p in mtl_model.s3gen.parameters():
            p.requires_grad = False
        logger.info("S3Gen model frozen.")

    for p in t3_model.parameters():
        p.requires_grad = True
    logger.info("T3 model set to trainable.")

    # ---- Load precomputed data ----
    logger.info(f"Loading precomputed features from: {data_args.precomputed_path}")
    all_data = torch.load(data_args.precomputed_path, weights_only=False)
    if not isinstance(all_data, list):
        raise ValueError(
            "Expected precomputed .pt file to contain a list of dict examples."
        )

    n_total = len(all_data)
    logger.info(f"Total precomputed examples: {n_total}")

    # ---- Train / eval split ----
    train_data = all_data
    eval_data = None

    if training_args.do_eval and data_args.eval_split_size > 0.0 and n_total > 1:
        rng = np.random.RandomState(training_args.seed)
        indices = rng.permutation(n_total)
        n_eval = max(1, int(n_total * data_args.eval_split_size))
        eval_idx = indices[:n_eval]
        train_idx = indices[n_eval:]

        train_data = [all_data[i] for i in train_idx]
        eval_data = [all_data[i] for i in eval_idx]

        logger.info(
            f"Train examples: {len(train_data)} | Eval examples: {len(eval_data)}"
        )
    elif training_args.do_eval:
        logger.warning(
            "Evaluation requested but eval_split_size <= 0 or dataset too small. Skipping eval split."
        )

    train_dataset = PrecomputedT3Dataset(train_data)
    eval_dataset = (
        PrecomputedT3Dataset(eval_data)
        if (eval_data is not None and training_args.do_eval)
        else None
    )

    # ---- Data collator (same as before) ----
    data_collator = SpeechDataCollator(
        chatterbox_t3_config_instance,
        chatterbox_t3_config_instance.stop_text_token,
        chatterbox_t3_config_instance.stop_speech_token,
    )

    # ---- Wrap T3 into HF-compatible module ----
    hf_trainable_model = T3ForFineTuning(t3_model, chatterbox_t3_config_instance)

    # ---- Callbacks ----
    callbacks = []
    if (
        training_args.early_stopping_patience is not None
        and training_args.early_stopping_patience > 0
    ):
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=training_args.early_stopping_patience
            )
        )

    trainer = Trainer(
        model=hf_trainable_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=callbacks if callbacks else None,
    )

    # Optional but nicer: log both label sets
    trainer.label_names = ["labels_text", "labels_speech"]

    # ---- Training ----
    if training_args.do_train:
        logger.info("*** Training Multilingual T3 model on precomputed data ***")
        train_result = trainer.train(
            resume_from_checkpoint=training_args.resume_from_checkpoint
        )
        trainer.save_model()

        # Save T3 weights as safetensors (as in original script)
        t3_to_save = (
            trainer.model.t3
            if hasattr(trainer.model, "t3")
            else trainer.model.module.t3
        )
        finetuned_t3_state_dict = t3_to_save.state_dict()

        output_t3_safetensor_path = (
            Path(training_args.output_dir) / "t3_mtl23ls_v2.safetensors"
        )
        from safetensors.torch import save_file

        save_file(finetuned_t3_state_dict, output_t3_safetensor_path)
        logger.info(
            f"Finetuned Multilingual T3 model weights saved to {output_t3_safetensor_path}"
        )

        # Copy auxiliary components (ve, s3gen, tokenizers, conds, etc.)
        original_model_dir_for_copy = local_dir_path
        if original_model_dir_for_copy:
            import shutil

            for f_name in [
                "ve.pt",
                "s3gen.pt",
                "tokenizer.json",
                "grapheme_mtl_merged_expanded_v1.json",
                "Cangjie5_TC.json",
            ]:
                src_path = original_model_dir_for_copy / f_name
                if src_path.exists():
                    shutil.copy2(src_path, Path(training_args.output_dir) / f_name)

            if (original_model_dir_for_copy / "conds.pt").exists():
                shutil.copy2(
                    original_model_dir_for_copy / "conds.pt",
                    Path(training_args.output_dir) / "conds.pt",
                )

            logger.info(
                f"Full model components structured in {training_args.output_dir}"
            )

        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # ---- Evaluation ----
    if training_args.do_eval and eval_dataset is not None:
        logger.info("*** Evaluating Multilingual T3 model ***")
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    logger.info("Multilingual finetuning with precomputed data finished.")


if __name__ == "__main__":
    main()
