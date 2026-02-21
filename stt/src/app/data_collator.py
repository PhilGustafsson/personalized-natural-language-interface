import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    def __init__(self, processor, model_dtype=torch.float16):
        self.processor = processor
        self.model_dtype = model_dtype
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"]
        decoder_input_ids = labels.clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        batch["decoder_input_ids"] = decoder_input_ids
        
        if "input_features" in batch:
            batch["input_features"] = batch["input_features"].to(self.model_dtype)
            
        return batch
