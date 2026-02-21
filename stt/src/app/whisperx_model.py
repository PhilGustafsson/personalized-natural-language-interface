import os
import shutil
import subprocess
import torch
import whisperx
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import gc

def convert_finetuned_to_ct2(
    finetuned_model_path: str,
    base_model_path: str = "models/kb-whisper-large",
    output_path: str = None,
    quantization: str = "float16",
) -> str:

    if output_path is None:
        output_path = finetuned_model_path
    
    if os.path.exists(os.path.join(output_path, "model.bin")):
        print(f"CT2 model already exists at {output_path}", flush=True)
        return output_path
    
    print(f"Converting fine-tuned model to CTranslate2...", flush=True)
    print(f"  Fine-tuned: {finetuned_model_path}", flush=True)
    print(f"  Base model: {base_model_path}", flush=True)
    print(f"  Output: {output_path}", flush=True)
    
    merged_path = finetuned_model_path.rstrip("/") + "_merged_temp"
    ct2_temp_path = finetuned_model_path.rstrip("/") + "_ct2_temp"
    os.makedirs(merged_path, exist_ok=True)
    
    try:
        print("Loading base model...", flush=True)
        base_model = WhisperForConditionalGeneration.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        
        print("Loading and merging LoRA adapter...", flush=True)
        model = PeftModel.from_pretrained(base_model, finetuned_model_path)
        merged_model = model.merge_and_unload()
        
        print("Saving merged model...", flush=True)
        merged_model.save_pretrained(merged_path)
        
        processor = WhisperProcessor.from_pretrained(base_model_path)
        processor.save_pretrained(merged_path)
        
        for f in ["generation_config.json", "preprocessor_config.json"]:
            src = os.path.join(base_model_path, f)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(merged_path, f))
        
        print("Converting to CTranslate2...", flush=True)
        result = subprocess.run([
            "ct2-transformers-converter",
            "--model", merged_path,
            "--output_dir", ct2_temp_path,
            "--quantization", quantization,
            "--force",
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"CT2 conversion failed: {result.stderr}")
        
        print(f"Moving CT2 files to {output_path}...", flush=True)
        for item in os.listdir(ct2_temp_path):
            src_item = os.path.join(ct2_temp_path, item)
            dst_item = os.path.join(output_path, item)
            if os.path.exists(dst_item):
                if os.path.isdir(dst_item):
                    shutil.rmtree(dst_item)
                else:
                    os.remove(dst_item)
            shutil.move(src_item, dst_item)
        
        preprocessor_src = os.path.join(base_model_path, "preprocessor_config.json")
        if os.path.exists(preprocessor_src):
            shutil.copy(preprocessor_src, os.path.join(output_path, "preprocessor_config.json"))

        # Copy other necessary config files from base model
        for f in ["vocabulary.json", "tokenizer.json"]:
            src = os.path.join(base_model_path, f)
            if os.path.exists(src):
                print(f"Copying {f} to {output_path}...", flush=True)
                shutil.copy(src, os.path.join(output_path, f))
        
        print(f"Conversion complete! CT2 model saved to: {output_path}", flush=True)
        
    finally:
        if os.path.exists(merged_path):
            shutil.rmtree(merged_path)
        if os.path.exists(ct2_temp_path):
            shutil.rmtree(ct2_temp_path)
    
    return output_path

def convert_hf_to_ct2(
    model_path: str,
    output_path: str = None,
    quantization: str = "float16",
) -> str:
    """
    Converts a standard Hugging Face model (not fine-tuned/LoRA) to CTranslate2 format.
    """
    if output_path is None:
        output_path = model_path
    
    if os.path.exists(os.path.join(output_path, "model.bin")):
        print(f"CT2 model already exists at {output_path}", flush=True)
        return output_path
    
    print(f"Converting HF model to CTranslate2...", flush=True)
    print(f"  Model: {model_path}", flush=True)
    print(f"  Output: {output_path}", flush=True)
    
    ct2_temp_path = model_path.rstrip("/") + "_ct2_temp"
    
    try:
        print("Converting to CTranslate2...", flush=True)
        result = subprocess.run([
            "ct2-transformers-converter",
            "--model", model_path,
            "--output_dir", ct2_temp_path,
            "--quantization", quantization,
            "--force",
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"CT2 conversion failed: {result.stderr}")
        
        print(f"Moving CT2 files to {output_path}...", flush=True)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            
        for item in os.listdir(ct2_temp_path):
            src_item = os.path.join(ct2_temp_path, item)
            dst_item = os.path.join(output_path, item)
            if os.path.exists(dst_item):
                if os.path.isdir(dst_item):
                    shutil.rmtree(dst_item)
                else:
                    os.remove(dst_item)
            shutil.move(src_item, dst_item)
        
        # Copy necessary config files from original model
        for f in ["vocabulary.json", "tokenizer.json", "preprocessor_config.json"]:
            src = os.path.join(model_path, f)
            if os.path.exists(src):
                print(f"Copying {f} to {output_path}...", flush=True)
                shutil.copy(src, os.path.join(output_path, f))
        
        print(f"Conversion complete! CT2 model saved to: {output_path}", flush=True)
        
    finally:
        if os.path.exists(ct2_temp_path):
            shutil.rmtree(ct2_temp_path)
    
    return output_path

class WhisperX_Model:
    def __init__(
        self, 
        model_name: str = None,
        align_model_name: str = None,
        cache_dir: str = None,
        language: str = "sv",
        compute_type: str = None,
        batch_size: int = 64,
        chunk_size: int = 30,
        device: str = None,
        device_index: int = 0,
    ):
        
        gc.collect()
        with torch.no_grad():
            torch.cuda.empty_cache()

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device_index = device_index
        self.compute_type = compute_type or ("float16" if self.device == "cuda" else "int8")
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.language = language
        
        self.model_name = model_name
        self.align_model_name = align_model_name 
        self.cache_dir = cache_dir
        
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.model = None
        self.align_model = None
        self.align_metadata = None
        
        self._load_model()
        self._load_align_model()
        
        print(f"WhisperX Model initialized!", flush=True)
    
    def _load_model(self):
        if self.model is None:
            print(f"Loading Whisper model: {self.model_name}...", flush=True)
            self.model = whisperx.load_model(
                self.model_name,
                self.device,
                device_index=self.device_index,
                compute_type=self.compute_type,
                download_root=self.cache_dir,
                language=self.language,
            )
            print("Whisper model loaded.", flush=True)
        return self.model

    def _load_align_model(self):
        """Load the alignment model (lazy loading)."""
        if self.align_model is None:
            print(f"Loading alignment model: {self.align_model_name}...", flush=True)
            self.align_model, self.align_metadata = whisperx.load_align_model(
                language_code=self.language,
                device=self.device,
                model_name=self.align_model_name,
                model_dir=self.cache_dir,
            )
            print("Alignment model loaded.", flush=True)
        return self.align_model, self.align_metadata

    def transcribe_whisperx(
        self, 
        audio_path: str, 
        do_align: bool = True, 
    ) -> dict:
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        if not self.model:
            raise RuntimeError("WhisperX model not loaded. Call /load_whisperx_model to load the WhisperX Model first.")

        print(f"Transcribing: {audio_path}", flush=True)
        
        audio = whisperx.load_audio(audio_path)
        
        result = self.model.transcribe(audio, batch_size=self.batch_size)
        
        print(f"Transcription complete. Found {len(result.get('segments', []))} segments.", flush=True)
        
        if do_align and result.get("segments"):
            print("Performing word-level alignment...", flush=True)
            try:
                result = whisperx.align(
                    result["segments"],
                    self.align_model,
                    self.align_metadata,
                    audio,
                    self.device,
                    return_char_alignments=False,
                )
                print("Alignment complete.", flush=True)
            except Exception as e:
                print(f"WARNING: Alignment failed: {e}", flush=True)
        
        segments = result.get("segments", [])
        full_text = " ".join([seg.get("text", "").strip() for seg in segments])
        result["text"] = full_text
        
        return result