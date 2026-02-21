# Endpoints for managing fine-tuning jobs.

from flask import Blueprint, request, jsonify
from db import DATA_DIR, get_db
import os
import time
import requests
from services import jobs
import services.storage as st

finetuning = Blueprint('finetuning', __name__)

TTS_SERVICE_URL = "http://tts-app:8002"
STT_SERVICE_URL = "http://stt-app:5080"


@finetuning.route("/start-tts/", methods=["POST"])
def start_fine_tuning_tts():
    """
    Start a fine-tuning job for TTS, matching STT pattern.
    """
    data = request.get_json()
    profile_id = data.get('profileID') if data else request.form.get('profileID')

    if not profile_id:
        return jsonify({"error": "profileID is required"}), 400

    folder_name = st._get_folder_name(profile_id)
    
    # Paths as seen by the TTS container (which mounts same volume at /app/data)
    # TTS uses metadata.txt format instead of JSONL
    metadata_path = f"/app/data/profiles/{folder_name}/audio_prompts/metadata.txt"
    wav_root = f"/app/data/profiles/{folder_name}/audio_prompts"
    
    payload = {
        "manifest_path": metadata_path,
        "recordings_root": wav_root,
        "user": folder_name,
        "eval_split_size": 0.1,
        "seed": 42
    }

    print(f"Loading dataset on TTS service: {payload}")

    try:
        # First call TTS /load_dataset to prepare the dataset from the shared volume
        load_resp = requests.post(f"{TTS_SERVICE_URL}/load_dataset", json=payload, timeout=60)
        if load_resp.status_code != 200:
            print(f"TTS load_dataset failed: {load_resp.status_code} - {load_resp.text}")
            return jsonify({"error": "TTS failed to load dataset", "details": load_resp.text}), 500

        # Now request fine-tuning on the TTS service
        ft_payload = {
            "user": folder_name,
            "saved_model_dir": f"/app/data/profiles/{folder_name}/tts_models/finetuned",
            "learning_rate": 1e-5,
            "num_train_epochs": 10,
            "per_device_train_batch_size": 2,
            "per_device_eval_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "warmup_steps": 200,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "eval_split_size": 0.1,
            "language_id": "sv",
            "seed": 42
        }
        print(f"Sending fine-tune request to TTS service: {ft_payload}")

        response = requests.post(f"{TTS_SERVICE_URL}/fine_tune", json=ft_payload, timeout=60)

        if response.status_code == 200:
            resp_data = response.json()
            tts_job_id = resp_data.get("job_id")

            # Create a local tracker job so frontend can poll `/finetuning/status/<jobId>`.
            job_id = str(int(time.time() * 1000))
            job = jobs.TTSFineTuningJob(job_id, tts_job_id=tts_job_id)
            jobs.job_manager.add_job(job)
            job.start()

            out = {
                **resp_data,
                "jobId": job_id,
                "ttsJobId": tts_job_id
            }
            return jsonify(out), 200
        elif response.status_code == 409:
            print("Training already in progress on TTS. Attaching to existing session.")
            resp_data = response.json()
            tts_job_id = resp_data.get("job_id")

            job_id = str(int(time.time() * 1000))
            job = jobs.TTSFineTuningJob(job_id, tts_job_id=tts_job_id)
            jobs.job_manager.add_job(job)
            job.start()

            return jsonify({
                "message": "Training already in progress. Attached to existing session.",
                "jobId": job_id,
                "ttsJobId": tts_job_id
            }), 200
        else:
            print(f"TTS service error: {response.status_code} - {response.text}")
            return jsonify({"error": f"TTS service returned {response.status_code}", "details": response.text}), 500
    except requests.exceptions.ConnectionError:
        return jsonify({"error": "Could not connect to TTS service. Is it running?"}), 503
    except Exception as e:
        print(f"Failed to call TTS service: {e}")
        return jsonify({"error": f"Failed to call TTS service: {str(e)}"}), 500


@finetuning.route("/start-stt/", methods=["POST"])
def start_fine_tuning_stt():
    """
    Start a fine-tuning job for STT.
    """
    data = request.get_json()
    profile_id = data.get('profileID') if data else request.form.get('profileID')

    if not profile_id:
        return jsonify({"error": "profileID is required"}), 400

    folder_name = st._get_folder_name(profile_id)
    
    # Paths as seen by the STT container (which mounts same volume at /app/data)
    # Note: DATA_DIR in backend is /app/data
    manifest_path = f"/app/data/profiles/{folder_name}/audio_prompts/metadata.jsonl"
    wav_root = f"/app/data/profiles/{folder_name}/audio_prompts"
    
    payload = {
        "manifest_path": manifest_path,
        "recordings_root": wav_root,
        "user": folder_name,
        "split_ratios": {
            "train": 0.8,
            "val": 0.1,
            "test": 0.1
        }
    }

    print(f"Loading dataset on STT service: {payload}")

    try:
        # Ensure base model is selected first
        select_payload = {
            "model_dir": "models/kb-whisper-large",
            "whisper_language": "Swedish"
        }
        print(f"Selecting base model on STT service: {select_payload}")
        select_resp = requests.post("http://stt-app:5080/select_model", json=select_payload, timeout=60)
        if select_resp.status_code not in [200, 204]:
             print(f"STT select_model failed: {select_resp.status_code} - {select_resp.text}")
             return jsonify({"error": "STT failed to select model", "details": select_resp.text}), 500

        # First call STT /load_dataset so it prepares the dataset from the shared volume
        load_resp = requests.post("http://stt-app:5080/load_dataset", json=payload, timeout=60)
        if load_resp.status_code != 200:
            print(f"STT load_dataset failed: {load_resp.status_code} - {load_resp.text}")
            return jsonify({"error": "STT failed to load dataset", "details": load_resp.text}), 500

        # Now request fine-tuning on the STT service. Keep same payload shape as FineTuneRequest (user etc.)
        ft_payload = {"user": folder_name}
        print(f"Sending fine-tune request to STT service: {ft_payload}")

        response = requests.post("http://stt-app:5080/fine_tune", json=ft_payload, timeout=60)

        if response.status_code == 200:
            resp_data = response.json()
            # STT returns its own job id as 'job_id' (FastAPI JobInfo.job_id). Use that to track remote job.
            stt_job_id = resp_data.get("job_id") or resp_data.get("jobId")

            # Create a local tracker job so frontend can poll `/finetuning/status/<jobId>`.
            job_id = str(int(time.time() * 1000))
            job = jobs.STTFineTuningJob(job_id, stt_job_id=stt_job_id)
            jobs.job_manager.add_job(job)
            job.start()

            # Return STT response plus our local job id so frontend can poll our status endpoint.
            out = {
                **resp_data,
                "jobId": job_id,
                "sttJobId": stt_job_id
            }
            return jsonify(out), 200
        elif response.status_code == 409:
            print("Training already in progress on STT. Attaching to existing session.")
            resp_data = response.json()
            stt_job_id = resp_data.get("job_id") or resp_data.get("jobId")

            job_id = str(int(time.time() * 1000))
            job = jobs.STTFineTuningJob(job_id, stt_job_id=stt_job_id)
            jobs.job_manager.add_job(job)
            job.start()

            return jsonify({
                "message": "Training already in progress. Attached to existing session.",
                "jobId": job_id,
                "sttJobId": stt_job_id,
                "wer": 0.0,
                "time": "Resumed"
            }), 200
        else:
            print(f"STT service error: {response.status_code} - {response.text}")
            return jsonify({"error": f"STT service returned {response.status_code}", "details": response.text}), 500
    except requests.exceptions.ConnectionError:
        return jsonify({"error": "Could not connect to STT service. Is it running?"}), 503
    except Exception as e:
        print(f"Failed to call STT service: {e}")
        return jsonify({"error": f"Failed to call STT service: {str(e)}"}), 500


@finetuning.route("/status/<job_id>", methods=["GET"])
def get_status(job_id):
    """
    Get the status of a fine-tuning job.
    """
    job = jobs.job_manager.get_job(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    progress, estimated_time_remaining = job.get_progress()
    
    response = {
        "jobId": job_id,
        "progress": progress,
        "estimatedTimeRemaining": estimated_time_remaining
    }
    
    if job.result:
        response["result"] = job.result
        
    return jsonify(response), 200


@finetuning.route("/cancel/<job_id>", methods=["POST"])
def cancel_fine_tuning(job_id):
    """
    Cancel a fine-tuning job.
    """
    success = jobs.job_manager.update_job_status(job_id, "cancelled")
    if not success:
        return jsonify({"error": "Job not found"}), 404

    # Implement actual cancellation logic in the job processing code.
    return jsonify({"message": "Job cancelled"}), 200

@finetuning.route("/unload-tts-model/", methods=["POST"])
def unload_tts_model():
    """Unload TTS models to free VRAM."""
    try:
        resp = requests.post(f"{TTS_SERVICE_URL}/unload-models", timeout=30)
        return jsonify(resp.json()), resp.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@finetuning.route("/unload-stt-model/", methods=["POST"])
def unload_stt_model():
    """Unload STT model to free VRAM."""
    try:
        resp = requests.post(f"{STT_SERVICE_URL}/unload-model", timeout=30)
        return jsonify(resp.json()), resp.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500
