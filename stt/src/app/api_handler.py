from fastapi import FastAPI, HTTPException, Response, BackgroundTasks  
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import time, pathlib
import uuid
from datetime import datetime
from enum import Enum
import os

from model_setup import ModelSetup
from whisperx_model import WhisperX_Model, convert_finetuned_to_ct2, convert_hf_to_ct2
from stt_service import STTService
from data_set import DataSet

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
whisperx_model = None
model_setup = None 
cached_dataset = None  
dataset_info = None    

job_store: Dict[str, "JobInfo"] = {}

class ModelSelect(BaseModel):
    model_dir: str = "models/kb-whisper-large"
    whisper_language: str = "Swedish"

class FineTuneRequest(BaseModel):
    user: Optional[str] = None  # Should be set from profile
    saved_model_dir: Optional[str] = "finetuned_models"
    seed: Optional[int] = 1337
    
    # Training params
    warmup_ratio: Optional[float] = 0.1
    learning_rate: Optional[float] = 0.000165
    num_train_epochs: Optional[int] = 5
    weight_decay: Optional[float] = 0.1
    per_device_train_batch_size: Optional[int] = 32
    per_device_eval_batch_size: Optional[int] = 32
    max_grad_norm: Optional[float] = 1.0
    label_smoothing_factor: Optional[float] = 0
    lr_scheduler_type: Optional[str] = "cosine"
    gradient_accumulation_steps: Optional[int] = 3
    optimizer: Optional[str] = "adamw_8bit"

class OptunaRequest(BaseModel):
    user: Optional[str] = None  # Should be set from profile
    
    seed: Optional[int] = 1337
    n_trials: Optional[int] = 5
    learning_rate_range: Optional[List[float]] = [1e-6, 5e-4]
    num_train_epochs_range: Optional[List[int]] = [9, 12]
    weight_decay_range: Optional[List[float]] = [0.0, 0.15]
    warmup_ratio_range: Optional[List[float]] = [0.05, 0.3]
    max_grad_norm_range: Optional[List[float]] = [0.5, 2.0]
    per_device_train_batch_size_range: Optional[List[int]] = [16, 32]
    per_device_eval_batch_size_range: Optional[List[int]] = [16, 32]
    gradient_accumulation_steps_range: Optional[List[int]] = [1, 4] 
    label_smoothing_factor_range: Optional[List[float]] = [0.0, 0.1]
    lr_scheduler_type_choices: Optional[List[str]] = ["cosine", "linear"]
    optimizer: Optional[List[str]] = ["adamw_8bit"]
    pruning_warmup_trials: Optional[int] = 1 # Don't prune first N trials
    pruning_warmup_epochs: Optional[int] = 1 # Don't prune first N epochs
    max_wer_threshold: Optional[float] = 30.0  # Prune if WER exceeds this
    pruning_wer_patience: Optional[int] = 2  # Prune if WER increases for N consecutive epochs

class EvaluateModelRequest(BaseModel):
    eval_split: Optional[str] = "test"  # Which split to evaluate: 'test', 'validation', or 'train'
    per_device_eval_batch_size: Optional[int] = 64

class LoadDatasetRequest(BaseModel):
    manifest_path: str  # Path to metadata.jsonl from profile's audio_prompts
    recordings_root: str  # Path to profile's audio_prompts directory
    user: str  # Profile username or folder name
    seed: Optional[int] = 1337
    split_ratios: Optional[Dict[str, float]] = {"train": 0.8, "val": 0.1, "test": 0.1}
    use_data_augmentation: Optional[bool] = False

class DatasetInfo(BaseModel):
    loaded: bool
    manifest_path: Optional[str] = None
    recordings_root: Optional[str] = None
    user: Optional[str] = None
    train_size: Optional[int] = None
    validation_size: Optional[int] = None
    test_size: Optional[int] = None

class FineTuneResponse(BaseModel):
    wer: float
    time: str
    finetuned_dir: Optional[str] = None

class TranscribeRequest(BaseModel):
    audio_path: str  # Path to audio file to transcribe
    transcribe_language: Optional[str] = "sv"

class WhisperXLoadRequest(BaseModel):
    model_name: Optional[str] = "models/kb-whisper-large" 
    align_model_name: Optional[str] = "models/wav2vec2-large-voxrex-swedish"  
    cache_dir: Optional[str] = "models/"
    base_model_path: Optional[str] = "models/kb-whisper-large"  # For auto-converting fine-tuned models
    language: Optional[str] = "sv"
    compute_type: Optional[str] = "float16"
    batch_size: Optional[int] = 64
    chunk_size: Optional[int] = 30

class WhisperXTranscribeRequest(BaseModel):
    audio_path: str  # Path to audio file to transcribe
    do_align: Optional[bool] = True

class TranscribeResponse(BaseModel):
    text: str
    language: Optional[str] = "sv"
    timestamps: Optional[List[Dict[str, Any]]] = []

class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class JobInfo(BaseModel):
    job_id: str
    status: JobStatus
    created_at: str
    updated_at: str
    message: str = ""
    progress: float = 0.0
    result: Optional[Dict[str, Any]] = None

# --- Helpers ---

def update_job(job_id: str, status: JobStatus, message: str = "", result=None, progress: float = None):
    if job_id in job_store:
        job = job_store[job_id]
        job.status = status
        job.message = message
        job.updated_at = datetime.now().isoformat()
        if result:
            job.result = result
        if progress is not None:
            job.progress = progress

# --- Background Task Wrappers ---

def _run_fine_tuning_task(job_id: str, params_path: str, request: FineTuneRequest):
    try:
        global cached_dataset
        
        if model_setup is None:
            raise ValueError("Model not selected. Please call /select_model first.")
        
        if cached_dataset is None:
            raise ValueError("Dataset not loaded. Please call /load_dataset first.")
        
        print("Fine-tuning Task Started, job id:", job_id)

        update_job(job_id, JobStatus.RUNNING, "Using cached dataset...")
        dataset = cached_dataset
        
        update_job(job_id, JobStatus.RUNNING, "Initializing Service...")
        svc = STTService(model_setup=model_setup, dataset=dataset)
        
        update_job(job_id, JobStatus.RUNNING, "Training started...")
        
        def progress_callback(prog):
            update_job(job_id, JobStatus.RUNNING, "Training in progress...", progress=prog)
            
        outdir = svc.train_with_best(best_params_path=params_path, progress_callback=progress_callback)
        
        final_wer = outdir["final_wer"]
        outdir_path = outdir["model_dir"]
        
        update_job(job_id, JobStatus.COMPLETED, "Training finished successfully.", result={"finetuned_dir": outdir_path, "final_wer": final_wer})
        
    except Exception as e:
        print(f"Error in fine_tune task: {e}")
        update_job(job_id, JobStatus.FAILED, message=str(e))

def _run_optuna_task(job_id: str, req: OptunaRequest):
    try:
        global cached_dataset
        
        if model_setup is None:
            raise ValueError("Model not selected. Please call /select_model first.")
        
        if cached_dataset is None:
            raise ValueError("Dataset not loaded. Please call /load_dataset first.")

        update_job(job_id, JobStatus.RUNNING, "Using cached dataset for Optuna...")
        dataset = cached_dataset
        
        svc = STTService(model_setup=model_setup, dataset=dataset)
        
        update_job(job_id, JobStatus.RUNNING, f"Running Optuna search ({req.n_trials} trials)...")
        
        def progress_callback(prog):
            update_job(job_id, JobStatus.RUNNING, f"Running Optuna search ({req.n_trials} trials)...", progress=prog)

        result = svc.run_optuna(
            user=req.user,
            n_trials=req.n_trials,
            lr_range=req.learning_rate_range,
            epochs_range=req.num_train_epochs_range,
            weight_decay_range=req.weight_decay_range,
            warmup_range=req.warmup_ratio_range,
            grad_norm_range=req.max_grad_norm_range,
            train_batch_size_range=req.per_device_train_batch_size_range,
            eval_batch_size_range=req.per_device_eval_batch_size_range,
            grad_accum_range=req.gradient_accumulation_steps_range,
            label_smoothing_range=req.label_smoothing_factor_range,
            lr_scheduler_choices=req.lr_scheduler_type_choices,
            optimizer_choices=req.optimizer,
            seed=req.seed,
            pruning_warmup_trials=req.pruning_warmup_trials,
            pruning_warmup_epochs=req.pruning_warmup_epochs,
            max_wer_threshold=req.max_wer_threshold,
            patience=req.pruning_wer_patience,
            progress_callback=progress_callback
        )
        
        update_job(job_id, JobStatus.COMPLETED, "Optuna search finished.", result=result)

    except Exception as e:
        print(f"Error in optuna task: {e}")
        update_job(job_id, JobStatus.FAILED, message=str(e))

def _run_evaluation_task(job_id: str, req: EvaluateModelRequest):
    try:
        global cached_dataset
        
        if model_setup is None:
            raise ValueError("Model not selected. Please call /select_model first.")
        
        if cached_dataset is None:
            raise ValueError("Dataset not loaded. Please call /load_dataset first.")
            
        update_job(job_id, JobStatus.RUNNING, "Using cached dataset for evaluation...")
        dataset = cached_dataset

        svc = STTService(model_setup=model_setup, dataset=dataset)
        
        update_job(job_id, JobStatus.RUNNING, f"Evaluating model on '{req.eval_split}' split...")
        
        eval_results = svc.evaluate_saved(eval_split=req.eval_split, per_device_eval_batch_size=req.per_device_eval_batch_size)
        
        update_job(job_id, JobStatus.COMPLETED, "Evaluation finished.", result=eval_results)
        
    except Exception as e:
        print(f"Error in evaluation task: {e}")
        import traceback
        traceback.print_exc()
        update_job(job_id, JobStatus.FAILED, message=str(e))

# --- API Endpoints ---

@app.get("/jobs/{job_id}", response_model=JobInfo)
async def get_job_status(job_id: str):
    if job_id not in job_store:
        raise HTTPException(status_code=404, detail="Job not found")
    return job_store[job_id]

@app.get("/finetuned_models")
async def get_finetuned_models():
    """
    Lists all available fine-tuned models in the current directory.
    Looks for directories containing 'adapter_config.json'.
    """
    models = []
    for root, dirs, files in os.walk("."):
        if "adapter_config.json" in files:
            path = os.path.relpath(root, ".")
            if path == ".": continue
            models.append(path)
        
        if root == ".":
            pass
        
    return {"models": models}
    
@app.post("/select_model", status_code=204)
async def select_model(request: ModelSelect):
    global model_setup
    try:
        model_setup = ModelSetup(
            model_dir=request.model_dir, 
            whisper_language=request.whisper_language
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")
    return Response(status_code=204)


@app.post("/load_dataset")
async def load_dataset(request: LoadDatasetRequest):
    global cached_dataset, dataset_info, model_setup
    
    if model_setup is None:
        raise HTTPException(status_code=400, detail="Model not selected. Please call /select_model first.")
    
    try:
        # Convert dict to tuple in order: (train, val, test)
        split_ratios_tuple = (
            request.split_ratios["train"],
            request.split_ratios["val"],
            request.split_ratios["test"]
        )
        
        cached_dataset = DataSet(
            request.manifest_path,
            request.recordings_root,
            request.user,
            model_setup,
            seed=request.seed,
            split_ratios=split_ratios_tuple,
            use_data_augmentation=request.use_data_augmentation
        )
        
        dataset_info = {
            "manifest_path": request.manifest_path,
            "recordings_root": request.recordings_root,
            "user": request.user,
            "seed": request.seed,
            "split_ratios": request.split_ratios,
            "use_data_augmentation": request.use_data_augmentation,
            "loaded_at": datetime.now().isoformat(),
            "train_samples": len(cached_dataset.ds_dict["train"]) if "train" in cached_dataset.ds_dict else 0,
            "val_samples": len(cached_dataset.ds_dict["validation"]) if "validation" in cached_dataset.ds_dict else 0,
            "test_samples": len(cached_dataset.ds_dict["test"]) if "test" in cached_dataset.ds_dict else 0,
        }
        
        return {"message": "Dataset loaded successfully", "dataset_info": dataset_info}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load dataset: {e}")

@app.post("/fine_tune", response_model=JobInfo)
async def fine_tune(request: FineTuneRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    
    # Save params to file
    params = {k: v for k, v in request.model_dump().items() if v is not None}
    ts = int(time.time())
    params_path = f"optuna_params_{ts}.txt"
    pathlib.Path(params_path).write_text(str(params), encoding="utf-8")

    # Create Job
    new_job = JobInfo(
        job_id=job_id,
        status=JobStatus.PENDING,
        created_at=datetime.now().isoformat(),
        updated_at=datetime.now().isoformat(),
        message="Queued for training"
    )
    job_store[job_id] = new_job

    background_tasks.add_task(_run_fine_tuning_task, job_id, params_path, request)
    
    return new_job

@app.post("/optuna_search", response_model=JobInfo)
async def optuna_search(request: OptunaRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    
    # Create Job
    new_job = JobInfo(
        job_id=job_id,
        status=JobStatus.PENDING,
        created_at=datetime.now().isoformat(),
        updated_at=datetime.now().isoformat(),
        message="Queued for Optuna search"
    )
    job_store[job_id] = new_job

    # Launch background task
    background_tasks.add_task(_run_optuna_task, job_id, request)
    
    return new_job

@app.post("/evaluate_model", response_model=JobInfo)
async def evaluate_model(request: EvaluateModelRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())

    # Create Job
    new_job = JobInfo(
        job_id=job_id,
        status=JobStatus.PENDING,
        created_at=datetime.now().isoformat(),
        updated_at=datetime.now().isoformat(),
        message="Queued for evaluation"
    )
    job_store[job_id] = new_job

    # Launch background task
    background_tasks.add_task(_run_evaluation_task, job_id, request)
    
    return new_job

@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(request: TranscribeRequest):
    if not request.audio_path:
        raise HTTPException(status_code=400, detail="audio_path required in JSON body")
    
    if model_setup is None:
        raise HTTPException(status_code=400, detail="Model not initialized. Call /select_model first.")

    print("Using model:", model_setup.model_dir, flush=True) 
    svc = STTService(model_setup=model_setup, dataset=None)
    print("Transcribing audio:", request.audio_path, flush=True)
    result = svc.transcribe(audio_path=request.audio_path, language=request.transcribe_language)
    
    formatted_timestamps = []
    
    if "chunks" in result:
        for chunk in result["chunks"]:
            ts = chunk.get("timestamp")
            
            start_time, end_time = None, None
            if isinstance(ts, (tuple, list)) and len(ts) >= 2:
                start_time = ts[0]
                end_time = ts[1]
                
            formatted_timestamps.append({
                "start": start_time,
                "end": end_time,
                "text": chunk.get("text", "").strip()
            })

    return TranscribeResponse(
        text=result["text"], 
        language=request.transcribe_language, 
        timestamps=formatted_timestamps
    )

@app.post("/transcribe_whisperx", response_model=TranscribeResponse)
async def transcribe_whisperx(request: WhisperXTranscribeRequest):
    if not request.audio_path:
        raise HTTPException(status_code=400, detail="audio_path required in JSON body")
    
    global whisperx_model
    if whisperx_model is None:
        raise HTTPException(status_code=500, detail="WhisperX model not loaded. Call /load_whisperx_model first.")
    
    try:
        result = whisperx_model.transcribe_whisperx(
            audio_path=request.audio_path,
            do_align=request.do_align
        )

        segments = result.get("segments", [])
        text = result.get("text", " ".join([seg.get("text", "").strip() for seg in segments]))
        
        timestamps = []
        for segment in segments:
            ts_entry = {
                "start": segment.get("start"),
                "end": segment.get("end"),
                "text": segment.get("text", "").strip(),
            }
            # Include word-level timestamps if available
            if "words" in segment:
                ts_entry["words"] = segment["words"]
            timestamps.append(ts_entry)
        
        return TranscribeResponse(
            text=text, 
            language=result.get("language", whisperx_model.language), 
            timestamps=timestamps
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


@app.post("/load_whisperx_model", status_code=204)
async def load_whisperx_model(request: WhisperXLoadRequest):
    """
    Load WhisperX model. 
    
    If model_name is a local path to a fine-tuned LoRA model (has adapter_config.json but no model.bin),
    it will automatically convert it to CTranslate2 format first.
    """
    global whisperx_model
    try:
        model_path = request.model_name
        
        # Check if it's a local fine-tuned model that needs conversion
        if os.path.isdir(model_path):
            has_adapter = os.path.exists(os.path.join(model_path, "adapter_config.json"))
            has_ct2 = os.path.exists(os.path.join(model_path, "model.bin"))
            has_config = os.path.exists(os.path.join(model_path, "config.json"))
            
            if not has_ct2:
                ct2_output_path = model_path.rstrip("/") + "_ct2"
                if os.path.exists(os.path.join(ct2_output_path, "model.bin")):
                    print(f"Found existing CT2 model at {ct2_output_path}, using it.", flush=True)
                    model_path = ct2_output_path
                elif has_adapter:
                    # This is a fine-tuned LoRA model - convert to CT2
                    print(f"Detected fine-tuned LoRA model, converting to CT2...", flush=True)
                    model_path = convert_finetuned_to_ct2(
                        finetuned_model_path=model_path,
                        base_model_path=request.base_model_path,
                        output_path=ct2_output_path,
                        quantization=request.compute_type or "float16",
                    )
                    print(f"Using converted model: {model_path}", flush=True)
                elif has_config:
                    # This is a base HF model - convert to CT2
                    print(f"Detected HF model (no adapter), converting to CT2...", flush=True)
                    model_path = convert_hf_to_ct2(
                        model_path=model_path,
                        output_path=ct2_output_path,
                        quantization=request.compute_type or "float16",
                    )
                    print(f"Using converted model: {model_path}", flush=True)
        
        whisperx_model = WhisperX_Model(
            model_name=model_path,
            align_model_name=request.align_model_name,
            cache_dir=request.cache_dir,
            language=request.language,
            compute_type=request.compute_type,
            batch_size=request.batch_size,
            chunk_size=request.chunk_size,
        )
        return Response(status_code=204)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load WhisperX model: {str(e)}")


@app.get("/training_status")
async def training_status():
    """
    Return a simple JSON indicating whether any training job is currently running.
    This endpoint is polled by the backend job tracker (see `services/jobs.py`).
    """
    # If any job in job_store has status RUNNING, report training active.
    for j in job_store.values():
        if j.status == JobStatus.RUNNING:
            return {"is_training": True, "job_id": j.job_id, "message": j.message}
    return {"is_training": False}


@app.post("/unload-model")
def unload_model():
    """Unload STT model from memory to free VRAM."""
    global model_setup
    try:
        if model_setup is not None and model_setup.model is not None:
            model_setup.unload_model()
            model_setup = None
        return {"status": "success", "message": "STT model unloaded"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
