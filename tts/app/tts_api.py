from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime
import shutil
import os
import sys
import uuid
from pathlib import Path

from tts_service import TTSService

# Add src directory to path for training service
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from tts_training_service import TTSTrainingService, TTSTrainingConfig

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize service without loading models - lazy loading on first use
service = TTSService()
models_loaded = False


@app.post("/unload-models")
def unload_models():
    """Unload TTS models from memory to free VRAM."""
    global models_loaded
    try:
        service.unload_models()
        models_loaded = False
        return {"status": "success", "message": "TTS models unloaded"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Training service and job tracking
training_service = None
job_store: Dict[str, "JobInfo"] = {}

class FinetuneRequest(BaseModel):
    output_dir: str = "./models/checkpoints/chatterbox_finetuned_swedish"
    metadata_file: str  # Path to metadata.txt from profile's audio_prompts
    dataset_dir: str  # Path to profile's audio_prompts directory
    local_model_dir: Optional[str] = "None" # The "doc page" insists on sending strings
    train_split_name: str = "train"
    eval_split_size: float = 0.1
    num_train_epochs: int = 10
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-5
    warmup_steps: int = 200
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    logging_steps: int = 10
    evaluation_strategy: str = "epoch"
    save_strategy: str = "epoch"
    save_total_limit: int = 5
    report_to: str = "tensorboard"
    do_train: bool = True
    do_eval: bool = True
    dataloader_pin_memory: bool = False
    eval_on_start: bool = True
    label_names: str = "labels_speech"
    text_column_name: str = "text_scribe"
    language_id: str = "sv"
    dataloader_num_workers: int = 0
    seed: int = 42

class LoadDatasetRequest(BaseModel):
    """Load dataset for TTS training, matching STT pattern."""
    manifest_path: str  # Path to metadata.txt from profile's audio_prompts
    recordings_root: str  # Path to profile's audio_prompts directory
    user: str  # Profile username or folder name
    seed: Optional[int] = 42
    eval_split_size: Optional[float] = 0.1

class DatasetInfo(BaseModel):
    """Dataset info response, matching STT pattern."""
    loaded: bool
    manifest_path: Optional[str] = None
    recordings_root: Optional[str] = None
    user: Optional[str] = None
    train_samples: Optional[int] = None
    eval_samples: Optional[int] = None
    total_samples: Optional[int] = None

class FineTuneRequestSTT(BaseModel):
    """Fine-tune request matching STT pattern."""
    user: Optional[str] = None  # Should be set from profile
    saved_model_dir: Optional[str] = "./models/checkpoints/tts_finetuned"
    seed: Optional[int] = 42
    
    # Training params matching common pattern
    learning_rate: Optional[float] = 1e-5
    num_train_epochs: Optional[int] = 10
    weight_decay: Optional[float] = 0.01
    per_device_train_batch_size: Optional[int] = 2
    per_device_eval_batch_size: Optional[int] = 2
    gradient_accumulation_steps: Optional[int] = 4
    warmup_steps: Optional[int] = 200
    max_grad_norm: Optional[float] = 1.0
    eval_split_size: Optional[float] = 0.1
    language_id: Optional[str] = "sv"

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

class SynthesizeRequest(BaseModel):
    text: str
    output_dir: Optional[str] = "./output"
    language: str = "sv"
    exaggeration: float = 0.5
    cfg_weight: float = 0.5
    filename: Optional[str] = None
    use_finetuned: bool = True

class EvaluateRequest(BaseModel):
    eval_sentences: List[str]
    output_dir_base: Optional[str] = "./examples/eval_audio/base"
    output_dir_ft: Optional[str] = "./examples/eval_audio/finetuned"
    language: str = "sv"
    audio_clip_directory: Optional[str] = None # Provide this to use already existing audio, instead of synthesizing new.

class FineTuneResponse(BaseModel):
    pass

class FineTunePollProgressResponse(BaseModel):
    time_left: float # time in milliseconds.
    progress: int # {0,...,100}


@app.post("/finetune")
def finetune(req: FinetuneRequest):
    data = req.dict()
    if data.get("local_model_dir") == "None":
        data["local_model_dir"] = None
    service.download_base_model()
    output_dir = service.finetune(**data)
    merged_dir = service.merge_finetuned_checkpoint()
    return {"output_dir": output_dir, "merged_dir": str(merged_dir)}

@app.get("/finetune/progress", response_model=FineTunePollProgressResponse)
async def poll_fine_tune_progress() :
    progress = service.get_finetuning_progress()
    return FineTunePollProgressResponse(
        time_left=progress.time_left,
        progress=progress.progress
    )

@app.post("/synthesize")
def synthesize(req: SynthesizeRequest):
    global models_loaded
    tts = service.ft_model if req.use_finetuned else service.base_model
    if tts is None:
        if not models_loaded:
            service.download_base_model()
        service.load_models()
        models_loaded = True
        tts = service.ft_model if req.use_finetuned else service.base_model
    wav, sr, out_path = service.synthesize_with_model(
        tts=tts,
        text=req.text,
        output_dir=req.output_dir,
        language=req.language,
        exaggeration=req.exaggeration,
        cfg_weight=req.cfg_weight,
        filename=req.filename or "tts_output.wav"
    )
    return FileResponse(
        path=str(out_path),
        media_type="audio/wav",
        filename=os.path.basename(str(out_path)),
        headers={
            "Content-Disposition": f"attachment; filename={os.path.basename(str(out_path))}",
            "Content-Type": "audio/wav"
        }
    )

@app.post("/evaluate")
def evaluate(req: EvaluateRequest):
    global models_loaded
    if not models_loaded:
        service.download_base_model()
        service.load_models()
        models_loaded = True
    else:
        service.load_models()
    # If audio_clip_directory is provided, use evaluate_existing_audio
    if req.audio_clip_directory:
        base_dir = os.path.join(req.audio_clip_directory, "base")
        ft_dir = os.path.join(req.audio_clip_directory, "finetuned")
        results = service.evaluate_existing_audio(
            eval_sentences=req.eval_sentences,
            base_dir=base_dir,
            ft_dir=ft_dir,
            language=req.language
        )
        # Convert dataclass to dict for JSONResponse
        from dataclasses import asdict
        results_dict = asdict(results)
        results_dict["sentence_results"] = [asdict(sr) for sr in results.sentence_results]
        return JSONResponse(content=results_dict)
    else:
        results = service.evaluate(
            eval_sentences=req.eval_sentences,
            output_dir_base=req.output_dir_base,
            output_dir_ft=req.output_dir_ft,
            language=req.language
        )
        return JSONResponse(content=results)



@app.post("/voice-profile")
def create_voice_profile(
    wav_file: UploadFile = File(...),
    name: str = Form(...),
    use_finetuned: bool = Form(True),
    output_dir: str = Form("./models/voices")
):
    global models_loaded
    tts = service.ft_model if use_finetuned else service.base_model
    if tts is None:
        if not models_loaded:
            service.download_base_model()
        service.load_models()
        models_loaded = True
        tts = service.ft_model if use_finetuned else service.base_model
    wav_path = os.path.join(output_dir, wav_file.filename)
    os.makedirs(output_dir, exist_ok=True)
    with open(wav_path, "wb") as buffer:
        shutil.copyfileobj(wav_file.file, buffer)
    profile_path = service.create_voice_profile_from_wav(
        tts=tts,
        wav_path=wav_path,
        output_dir=output_dir,
        name=name
    )
    return {"profile_path": str(profile_path)}

@app.post("/apply-voice-profile")
def apply_voice_profile(
    profile_path: str = Form(...),
    use_finetuned: bool = Form(True)
):
    global models_loaded
    tts = service.ft_model if use_finetuned else service.base_model
    if tts is None:
        if not models_loaded:
            service.download_base_model()
        service.load_models()
        models_loaded = True
        tts = service.ft_model if use_finetuned else service.base_model
    service.apply_voice_profile_from_path(tts, profile_path)
    return {"status": "applied", "profile_path": profile_path}


# ===== Training Endpoints (matching STT pattern) =====

def update_job(job_id: str, status: JobStatus, message: str = "", result=None, progress: float = None):
    """Update job status in job store."""
    if job_id in job_store:
        job = job_store[job_id]
        job.status = status
        job.message = message
        job.updated_at = datetime.now().isoformat()
        if result:
            job.result = result
        if progress is not None:
            job.progress = progress


@app.post("/load_dataset", response_model=DatasetInfo)
async def load_dataset(req: LoadDatasetRequest):
    """Load and prepare dataset for TTS training."""
    global training_service
    
    try:
        config = TTSTrainingConfig(
            metadata_file=req.manifest_path,
            dataset_dir=req.recordings_root,
            eval_split_size=req.eval_split_size,
            seed=req.seed
        )
        
        training_service = TTSTrainingService(config)
        dataset_info = training_service.prepare_dataset(
            manifest_path=req.manifest_path,
            audio_dir=req.recordings_root,
            user=req.user
        )
        
        return DatasetInfo(**dataset_info)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/fine_tune")
async def fine_tune(req: FineTuneRequestSTT, background_tasks: BackgroundTasks):
    """Start TTS fine-tuning job in background, matching STT pattern."""
    global training_service
    
    if training_service is None:
        raise HTTPException(
            status_code=400,
            detail="Dataset not loaded. Call /load_dataset first."
        )
    
    # Check if training already in progress
    for job in job_store.values():
        if job.status == JobStatus.RUNNING:
            return JSONResponse(
                status_code=409,
                content={"message": "Training already in progress", "job_id": job.job_id}
            )
    
    # Create new job
    job_id = str(uuid.uuid4())
    job = JobInfo(
        job_id=job_id,
        status=JobStatus.PENDING,
        created_at=datetime.now().isoformat(),
        updated_at=datetime.now().isoformat(),
        message="TTS training queued"
    )
    job_store[job_id] = job
    
    # Update training config with request params
    training_service.config.learning_rate = req.learning_rate
    training_service.config.num_train_epochs = req.num_train_epochs
    training_service.config.weight_decay = req.weight_decay
    training_service.config.per_device_train_batch_size = req.per_device_train_batch_size
    training_service.config.per_device_eval_batch_size = req.per_device_eval_batch_size
    training_service.config.gradient_accumulation_steps = req.gradient_accumulation_steps
    training_service.config.warmup_steps = req.warmup_steps
    training_service.config.max_grad_norm = req.max_grad_norm
    training_service.config.output_dir = req.saved_model_dir
    training_service.config.language_id = req.language_id
    training_service.config.seed = req.seed
    
    # Run training in background
    def run_training():
        try:
            update_job(job_id, JobStatus.RUNNING, "TTS training started", progress=0.0)
            result = training_service.train(user=req.user)
            update_job(
                job_id,
                JobStatus.COMPLETED,
                "TTS training completed",
                result=result,
                progress=100.0
            )
        except Exception as e:
            update_job(
                job_id,
                JobStatus.FAILED,
                f"TTS training failed: {str(e)}",
                progress=0.0
            )
    
    background_tasks.add_task(run_training)
    
    return {
        "job_id": job_id,
        "message": "TTS training started",
        "status": "pending"
    }


@app.get("/job_status/{job_id}")
async def get_job_status(job_id: str):
    """Get status of training job, matching STT pattern."""
    if job_id not in job_store:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = job_store[job_id]
    return {
        "job_id": job.job_id,
        "status": job.status.value,
        "message": job.message,
        "progress": job.progress,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "result": job.result
    }


@app.get("/dataset_info", response_model=DatasetInfo)
async def get_dataset_info():
    """Get current dataset info, matching STT pattern."""
    if training_service is None:
        return DatasetInfo(loaded=False)
    
    # Return cached info if available
    return DatasetInfo(
        loaded=True,
        manifest_path=training_service.config.metadata_file,
        recordings_root=training_service.config.dataset_dir,
        user="current_user",
        train_samples=0,  # Can be populated from dataset
        eval_samples=0,
        total_samples=0
    )

