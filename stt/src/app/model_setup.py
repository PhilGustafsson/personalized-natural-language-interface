import gc
import torch
import os
from transformers import (
    WhisperForConditionalGeneration, 
    AutoProcessor,
    AutoTokenizer
)
from peft import PeftModel, PeftConfig

class ModelSetup:
    def __init__(
        self,
        model_dir: str = "models/kb-whisper-large",
        whisper_language: str = "Swedish",
    ):
        gc.collect()
        with torch.no_grad():
            torch.cuda.empty_cache()

        self.model_dir = model_dir
        self.whisper_language = whisper_language
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        print(f"Initializing ModelSetup with: {self.model_dir}")
        print(f"Using device: {self.device}, dtype: {self.torch_dtype}")
        
        self.model = None
        self.processor = None
        self.tokenizer = None
        
        # Lazy loading - model will be loaded on first use
        # Call load_model() explicitly when needed

    def load_model(self):
        """Smart loader that handles Base Models, LoRA Adapters, and Checkpoints."""
        try:
            print(f"DEBUG: Checking for adapter_config.json in {self.model_dir}")
            is_lora = os.path.exists(os.path.join(self.model_dir, "adapter_config.json"))
            print(f"DEBUG: is_lora = {is_lora}")
            
            if is_lora:
                peft_config = PeftConfig.from_pretrained(self.model_dir)
                base_model_path = peft_config.base_model_name_or_path
                print(f"Detected LoRA/Checkpoint. Loading processor from Base: {base_model_path}")
                self.processor = AutoProcessor.from_pretrained(base_model_path)
                
                # Attempt to load tokenizer from the fine-tuned directory
                try:
                    print(f"Attempting to load tokenizer from: {self.model_dir}")
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
                    self.processor.tokenizer = self.tokenizer
                    print("Successfully loaded tokenizer from fine-tuned directory.")
                except Exception as e:
                    print(f"No tokenizer found in fine-tuned directory ({e}), using base tokenizer.")
                    self.tokenizer = self.processor.tokenizer
            else:
                print(f"Loading processor from: {self.model_dir}")
                self.processor = AutoProcessor.from_pretrained(self.model_dir)
                self.tokenizer = self.processor.tokenizer
            
            self.tokenizer.padding_side = "right"
            if hasattr(self.tokenizer, "tokenizer"):
                self.tokenizer.tokenizer.padding_side = "right"

            if is_lora:
                print("DEBUG: Calling _load_lora_model")
                self._load_lora_model()
            else:
                print("DEBUG: Calling _load_base_model")
                self._load_base_model()
            
            self._configure_model()
            
            if self.device == "cuda":
                print(f"Moving model to {self.device} and casting to {self.torch_dtype}...")
                self.model = self.model.to(self.device, dtype=self.torch_dtype)
            else:
                self.model = self.model.to(self.device)
            
            self.model.eval()

        except Exception as e:
            print(f"ERROR in load_model: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Failed to load model from {self.model_dir}: {e}")

    def unload_model(self):
        del self.model
        self.model = None
        gc.collect()
        torch.cuda.empty_cache()
        print("Model unloaded.")

    def _load_base_model(self):
        print(f"Loading Base Model: {self.model_dir}")
        self.model = WhisperForConditionalGeneration.from_pretrained(
            self.model_dir,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True
        )
        self._resize_if_needed()

    def _load_lora_model(self):
        """Loads Base Model + LoRA Adapter."""
        peft_config = PeftConfig.from_pretrained(self.model_dir)
        base_model_path = peft_config.base_model_name_or_path
        
        print(f"Loading Base Model for LoRA: {base_model_path}")
        base_model = WhisperForConditionalGeneration.from_pretrained(
            base_model_path,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
        )
        
        self.model = base_model 
        self._resize_if_needed()
        base_model = self.model
        
        print(f"Loading LoRA Adapter: {self.model_dir}")
        self.model = PeftModel.from_pretrained(base_model, self.model_dir)
        
        print("Merging LoRA weights into base model for inference...")
        self.model = self.model.merge_and_unload()

    def _resize_if_needed(self):
        if hasattr(self.tokenizer, "tokenizer"):
            vocab_size = len(self.tokenizer.tokenizer)
        else:
            vocab_size = len(self.tokenizer)

        current_size = self.model.get_input_embeddings().weight.shape[0]
        
        if current_size != vocab_size:
            print(f"Resizing Embeddings: {current_size} -> {vocab_size}")
            self.model.resize_token_embeddings(vocab_size)
            
    def _configure_model(self):
        tok = self.tokenizer.tokenizer if hasattr(self.tokenizer, "tokenizer") else self.tokenizer
        
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
            tok.pad_token_id = tok.eos_token_id
        
        self.model.config.pad_token_id = tok.pad_token_id
        self.model.generation_config.pad_token_id = tok.pad_token_id
        
        self.model.generation_config.language = "<|sv|>"
        self.model.generation_config.task = "transcribe"
        self.model.config.suppress_tokens = []