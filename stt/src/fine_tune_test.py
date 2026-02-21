from unsloth import FastModel
from transformers import WhisperForConditionalGeneration
import torch
import numpy as np
import tqdm
from datasets import load_dataset, Audio
from pathlib import Path
import evaluate
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import copy

# Model & Tokenizer
model, tokenizer = FastModel.from_pretrained(
    model_name = "../../models/kb-whisper-large",
    dtype = None, # Leave as None for auto detection
    load_in_4bit = False, # Set to True to do 4bit quantization which reduces memory
    auto_model = WhisperForConditionalGeneration,
    whisper_language = "Swedish",
    whisper_task = "transcribe",
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

model = FastModel.get_peft_model(
    model,
    r = 64, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "v_proj"],
    lora_alpha = 64,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
    task_type = None, # ** MUST set this for Whisper **
)

# Force PAD = EOS everywhere (Whisper-friendly)
# EOS = end of sentence token
# PAD = padding token
# Whisper does not have a PAD token by default, so we set PAD = EOS
tok = tokenizer.tokenizer
tok.pad_token = tok.eos_token
tok.pad_token_id = tok.eos_token_id  # ensure id switches too

model.config.pad_token_id = tok.eos_token_id
model.generation_config.pad_token_id = tok.eos_token_id


# # Data Prep
from sklearn.model_selection import train_test_split

model.generation_config.language = "<|sv|>"
model.generation_config.task = "transcribe"
model.config.suppress_tokens = []
model.generation_config.forced_decoder_ids = None


def formatting_prompts_func(example):
    audio_arrays = example['sentence']['array']
    sampling_rate = example["sentence"]["sampling_rate"]
    features = tokenizer.feature_extractor(
        audio_arrays, sampling_rate=sampling_rate
    )
    tokenized_text = tokenizer.tokenizer(example["text"])
    return {
        "input_features": features.input_features[0],
        "labels": tokenized_text.input_ids,
    }

USER_NAME = "david"
RECORDINGS_ROOT = Path("/data_generation/recordings")  # relative to this script
meta = Path("/data_generation/data_script.jsonl").resolve()  # your JSONL
base = meta.parent

dataset = load_dataset("json", data_files={"train": str(meta)}, encoding="utf-8")["train"]

def indices_to_paths(batch, user=USER_NAME, root=str(RECORDINGS_ROOT)):
    root = Path(root)
    out = []
    for a in batch["sentence"]:
        # Keep existing paths as-is
        if isinstance(a, str) and not a.strip().isdigit():
            out.append(a)
            continue
        # Convert numeric or numeric-string to ./record/recordings/<user>/<user>_sentenceXX.wav
        idx = int(a) if isinstance(a, (int, float)) else int(a.strip())
        rel = Path(root / user / f"{user}_sentence{idx:02d}.wav")
        # Use relative path (as requested). If you prefer absolute, resolve():
        out.append(str(rel))
    batch["sentence"] = out
    return batch

dataset = dataset.map(indices_to_paths, batched=True)
dataset = dataset.cast_column("sentence", Audio(sampling_rate=16000))
splits = dataset.train_test_split(test_size=0.25, seed=42)
train_val = splits["train"].train_test_split(test_size=0.20, seed=42)

from datasets import DatasetDict
dataset = DatasetDict({
    "train": train_val["train"],
    "validation": train_val["test"],
    "test": splits["test"],
})

print(dataset)

# Prepare model-ready dicts
train_dataset = [formatting_prompts_func(ex) for ex in tqdm.tqdm(dataset["train"], desc="Train split")]
val_dataset   = [formatting_prompts_func(ex) for ex in tqdm.tqdm(dataset["validation"], desc="Validation split")]
test_dataset  = [formatting_prompts_func(ex) for ex in tqdm.tqdm(dataset["test"], desc="Test split")]

print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

metric = evaluate.load("wer")
def compute_metrics(pred):

    pred_logits = pred.predictions[0]
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id


    pred_ids = np.argmax(pred_logits, axis=-1)

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:

        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch
    
    rom transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from unsloth import is_bf16_supported


# https://medium.com/@chris.xg.wang/a-guide-to-fine-tune-whisper-model-with-hyper-parameter-tuning-c13645ba2dba


# Dataset is small so we want to use evalation steps.
training_args = Seq2SeqTrainingArguments(
        # predict_with_generate=True,
        per_device_train_batch_size = 8,
        gradient_accumulation_steps = 1,
        warmup_steps = 100,
        eval_steps = 50, 
        max_steps = 200,
        eval_strategy="steps",
        metric_for_best_model='wer',
        learning_rate = 1e-5,
        # save_steps = 50, # We do not want to save too often
        # logging_steps = 50, # We do not want to log too often
        optim = "adamw_8bit",
        fp16 = not is_bf16_supported(),  # Use fp16 if bf16 is not supported
        bf16 = is_bf16_supported(),  # Use bf16 if supported
        weight_decay = 0.01,
        remove_unused_columns=False,  # required as the PeftModel forward doesn't have the signature of the wrapped model's forward
        lr_scheduler_type = "linear",
        label_names = ['labels'],
        seed = 42,
        output_dir = "outputs",
        report_to = ['tensorboard'] 
)

from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from unsloth import is_bf16_supported


# https://medium.com/@chris.xg.wang/a-guide-to-fine-tune-whisper-model-with-hyper-parameter-tuning-c13645ba2dba


# Dataset is small so we want to use evalation steps.
training_args = Seq2SeqTrainingArguments(
        # predict_with_generate=True,
        per_device_train_batch_size = 8,
        gradient_accumulation_steps = 1,
        warmup_steps = 100,
        eval_steps = 50, 
        max_steps = 200,
        eval_strategy="steps",
        metric_for_best_model='wer',
        learning_rate = 1e-5,
        # save_steps = 50, # We do not want to save too often
        # logging_steps = 50, # We do not want to log too often
        optim = "adamw_8bit",
        fp16 = not is_bf16_supported(),  # Use fp16 if bf16 is not supported
        bf16 = is_bf16_supported(),  # Use bf16 if supported
        weight_decay = 0.01,
        remove_unused_columns=False,  # required as the PeftModel forward doesn't have the signature of the wrapped model's forward
        lr_scheduler_type = "linear",
        label_names = ['labels'],
        seed = 42,
        output_dir = "outputs",
        report_to = ['tensorboard'] 
)

new_model = copy.deepcopy(model)
trainer = Seq2SeqTrainer(
    model=new_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor=tokenizer),
    tokenizer=tokenizer.feature_extractor,
    compute_metrics=compute_metrics,
)

# @title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

# Train the model
trainer_stats = trainer.train()


#@title Show final memory and time stats
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(
    f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
)
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")


# Plot accuracy, loss for each training step
logs = trainer.state.log_history

# Training loss
loss_steps = [e.get("step", i) for i, e in enumerate(logs) if "loss" in e]
loss_vals  = [e["loss"] for e in logs if "loss" in e]

# Eval WER
eval_steps = [e.get("step", i) for i, e in enumerate(logs) if "eval_wer" in e]
eval_vals  = [e["eval_wer"] for e in logs if "eval_wer" in e]

print(f"loss points: {len(loss_vals)}, steps: {len(loss_steps)}")
print(f"eval points: {len(eval_vals)}, steps: {len(eval_steps)}")

import matplotlib.pyplot as plt
plt.figure(figsize=(8,4))
plt.plot(loss_steps, loss_vals, label="Training Loss")
plt.scatter(eval_steps, eval_vals, color='red', label="Eval WER")
plt.xlabel("Step")
plt.ylabel("Loss / WER")
plt.title("Training Loss and Eval WER over Steps")
plt.legend()
plt.grid(True)
plt.show()
