# Endpoints for TTS operations.

from flask import Blueprint, request, jsonify, send_file
from db import DATA_DIR, get_db
import os
import time
import random
import csv
from datetime import datetime
import requests

inference_bp = Blueprint('inference', __name__)

STT_SERVICE_URL = "http://stt-app:5080"

def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def calculate_wer(reference, hypothesis):
    # Simple normalization
    r = reference.lower().replace('.', '').replace(',', '').split()
    h = hypothesis.lower().replace('.', '').replace(',', '').split()
    return levenshtein(r, h) / len(r) if len(r) > 0 else 0

def calculate_cer(reference, hypothesis):
    return levenshtein(reference, hypothesis) / len(reference) if len(reference) > 0 else 0

@inference_bp.route("/select_model", methods=["POST"])
def select_model():
    try:
        # Forward request to STT service
        resp = requests.post(f"{STT_SERVICE_URL}/select_model", json=request.get_json())
        if resp.status_code == 204:
            return "", 204
        return jsonify(resp.json() if resp.content else {}), resp.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@inference_bp.route("/unload-stt-model", methods=["POST"])
def unload_stt_model():
    """Unload STT model to free VRAM."""
    try:
        resp = requests.post(f"{STT_SERVICE_URL}/unload-model", timeout=30)
        return jsonify(resp.json()), resp.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@inference_bp.route("/transcribe", methods=["POST"])
def transcribe():
    try:
        data = request.get_json()
        audio_path = data.get("audio_path")
        if not audio_path:
             return jsonify({"error": "audio_path required"}), 400
             
        # Prepend /app/data/ if not present (assuming relative path from data dir)
        if not audio_path.startswith("/app/data/"):
             audio_path = os.path.join("/app/data", audio_path)
             
        payload = {
            "audio_path": audio_path,
            "transcribe_language": data.get("transcribe_language", "sv")
        }
        
        resp = requests.post(f"{STT_SERVICE_URL}/transcribe", json=payload)
        return jsonify(resp.json()), resp.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500



@inference_bp.route("/tts/default/", methods=["POST"])
def text_to_speech_default():
    data = request.get_json()
    text = data.get("text", "")

    # Simulate processing delay (e.g., generating audio)
    time.sleep(3)  # <-- Delay for 3 seconds

    # tmp: use a dummy sound file.
    file_path = os.path.join('./assets/sounds/sound.m4a')

    return send_file(
        file_path,
        #mimetype='audio/mpeg',
        as_attachment=False,
        download_name='speech.m4a' # todo: change to actual file type.
    )

@inference_bp.route("/tts/fine-tuned/<profile_id>", methods=["POST"])
def text_to_speech_finetuned(profile_id):
    data = request.get_json()
    text = data.get("text", "")

    # Simulate processing delay (e.g., generating audio)
    time.sleep(3)  # <-- Delay for 3 seconds

    # tmp: use a dummy sound file.
    file_path = os.path.join('./assets/sounds/sound.m4a')

    return send_file(
        file_path,
        #mimetype='audio/mpeg',
        as_attachment=False,
        download_name='speech.m4a' # todo: change to actual file type.
    )

@inference_bp.route("/stt/", methods=["POST"])
def speech_to_text():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files['audio']
    use_finetuned = request.form.get('use_finetuned', 'false').lower() == 'true'
    
    # Save audio file to a shared location
    # DATA_DIR is /app/data
    temp_dir = os.path.join(DATA_DIR, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    filename = f"upload_{int(time.time())}_{random.randint(1000,9999)}.wav"
    file_path = os.path.join(temp_dir, filename)
    audio_file.save(file_path)
    
    # Path as seen by STT container (same mount)
    stt_container_path = f"/app/data/temp/{filename}"

    payload = {
        "audio_path": stt_container_path,
        "return_timestamps": False
    }
    
    if use_finetuned:
        profile_id = request.form.get('profile_id')
        if profile_id:
             # Construct path to the fine-tuned model/adapter
             # Assuming it's saved in the profile directory or a standard location
             # For now, let's point to the one created by train_with_best if it's global, 
             # or we need to know where it is.
             # The current training saves to "saved_model_best_lora" in the container's CWD.
             # We might need to move it to /app/data/profiles/{user}/model
             pass

    print(f"Sending transcribe request to STT service: {payload}")

    # Dummy transcription result
    transcriptions = [
        "Hello, how can I help you today?",
        "The quick brown fox jumps over the lazy dog.",
        "Please repeat your request.",
        "I'm sorry, I didn't catch that.",
        "This is a randomly generated transcription."
    ]
    transcription = random.choice(transcriptions)

    return jsonify({"text": transcription})

@inference_bp.route("/upload-audio", methods=["POST"])
def upload_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files['audio']
    
    # Save audio file to a shared location
    # DATA_DIR is /app/data
    temp_dir = os.path.join(DATA_DIR, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    filename = f"upload_{int(time.time())}_{random.randint(1000,9999)}.wav"
    file_path = os.path.join(temp_dir, filename)
    audio_file.save(file_path)
    
    # Path as seen by STT container (same mount)
    stt_container_path = f"/app/data/temp/{filename}"
    
    return jsonify({"path": stt_container_path})

@inference_bp.route("/audio-files/<profile_id>/<folder>", methods=["GET"])
def get_audio_files(profile_id, folder):
    """Get list of audio files from a profile's audio folder."""
    try:
        audio_dir = os.path.join(DATA_DIR, "profiles", profile_id, folder)
        
        if not os.path.exists(audio_dir):
            return jsonify({"files": []}), 200
        
        # Get all audio files (.wav, .mp3, .m4a, .ogg)
        audio_extensions = ('.wav', '.mp3', '.m4a', '.ogg', '.flac')
        files = [f for f in os.listdir(audio_dir) 
                if f.lower().endswith(audio_extensions) and os.path.isfile(os.path.join(audio_dir, f))]
        
        return jsonify({"files": sorted(files)}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@inference_bp.route("/audio/<profile_id>/<folder>/<filename>", methods=["GET"])
def get_audio_file(profile_id, folder, filename):
    """Serve an audio file from a profile's audio folder."""
    try:
        audio_dir = os.path.join(DATA_DIR, "profiles", profile_id, folder)
        file_path = os.path.join(audio_dir, filename)
        
        # Security check: ensure the file path is within the audio directory
        if not os.path.abspath(file_path).startswith(os.path.abspath(audio_dir)):
            return jsonify({"error": "Invalid file path"}), 403
        
        if not os.path.exists(file_path):
            return jsonify({"error": "File not found"}), 404
        
        return send_file(file_path, as_attachment=False)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@inference_bp.route("/save-evaluation/<profile_id>/<folder>", methods=["POST"])
def save_evaluation(profile_id, folder):
    """Save audio evaluation to CSV file."""
    try:
        data = request.get_json()
        audio_filename = data.get('audioFilename')
        evaluations = data.get('evaluations', {})
        
        if not audio_filename:
            return jsonify({"error": "No audio filename provided"}), 400
        
        # Create path to evaluation CSV file
        eval_dir = os.path.join(DATA_DIR, "profiles", profile_id, folder)
        os.makedirs(eval_dir, exist_ok=True)
        
        csv_filename = 'evaluation.csv'
        csv_filepath = os.path.join(eval_dir, csv_filename)
        
        # Check if file exists to determine if we need headers
        file_exists = os.path.isfile(csv_filepath)
        
        # Prepare the row data
        row = {
            'timestamp': datetime.now().isoformat(),
            'audio_filename': audio_filename,
            'Likeness': evaluations.get('Likeness', ''),
            'Naturalness': evaluations.get('Naturalness', ''),
            'Intelligability': evaluations.get('Intelligability', ''),
            'Artifacts': evaluations.get('Artifacts', '')
        }
        
        # Write to CSV
        fieldnames = ['timestamp', 'audio_filename', 'Likeness', 'Naturalness', 'Intelligability', 'Artifacts']
        
        with open(csv_filepath, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(row)
        
        return jsonify({"message": "Evaluation saved successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@inference_bp.route("/audio-metadata/<profile_id>/<folder>", methods=["GET"])
def get_audio_metadata(profile_id, folder):
    """
    Return metadata entries for a given profile folder.
    Supports both metadata.jsonl and CSV formats.
    """
    print(f"Getting metadata for {profile_id}/{folder}")
    try:
        audio_dir = os.path.join(DATA_DIR, "profiles", profile_id, folder)
        if not os.path.exists(audio_dir):
            print(f"Directory not found: {audio_dir}")
            return jsonify({"metadata": []}), 200

        entries = []
        
        # Try metadata.jsonl first
        jsonl_path = os.path.join(audio_dir, "metadata.jsonl")
        if os.path.exists(jsonl_path):
            print(f"Found jsonl: {jsonl_path}")
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    try:
                        obj = json.loads(line)
                        sentence = obj.get('sentence') or obj.get('audio') or ''
                        text = obj.get('text') or ''
                        filename = os.path.basename(sentence)
                        entries.append({"filename": filename, "path": sentence, "text": text})
                    except Exception as e:
                        print(f"Error parsing jsonl line: {e}")
                        continue
            
            if entries:
                print(f"Loaded {len(entries)} entries from jsonl")
                return jsonify({"metadata": entries}), 200

        # Try CSV if jsonl not found or empty
        csv_files = [f for f in os.listdir(audio_dir) if f.endswith('.csv') and 'evaluation' not in f]
        if csv_files:
            csv_path = os.path.join(audio_dir, csv_files[0])
            print(f"Found csv: {csv_path}")
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter='|', quotechar='"')
                for row in reader:
                    if len(row) >= 2:
                        filename = row[0].strip('"')
                        text = row[1].strip('"')
                        # Construct path assuming it's in the same folder
                        path = os.path.join(audio_dir, filename)
                        entries.append({"filename": filename, "path": path, "text": text})
            print(f"Loaded {len(entries)} entries from csv")
            return jsonify({"metadata": entries}), 200

        print("No metadata found")
        return jsonify({"metadata": []}), 200
    except Exception as e:
        print(f"Error getting metadata: {e}")
        return jsonify({"error": str(e)}), 500

@inference_bp.route("/datasets/<profile_id>", methods=["GET"])
def get_datasets(profile_id):
    """List available datasets (folders) for a profile."""
    try:
        profile_dir = os.path.join(DATA_DIR, "profiles", profile_id)
        if not os.path.exists(profile_dir):
            return jsonify({"datasets": []}), 200
            
        datasets = []
        for item in os.listdir(profile_dir):
            item_path = os.path.join(profile_dir, item)
            if os.path.isdir(item_path):
                # Check if it has audio files or metadata
                has_audio = any(f.endswith(('.wav', '.mp3', '.m4a')) for f in os.listdir(item_path))
                has_metadata = os.path.exists(os.path.join(item_path, "metadata.jsonl")) or \
                               any(f.endswith('.csv') for f in os.listdir(item_path))
                
                if has_audio or has_metadata:
                    datasets.append(item)
                    
        return jsonify({"datasets": sorted(datasets)}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@inference_bp.route("/evaluate_transcription", methods=["POST"])
def evaluate_transcription():
    try:
        data = request.get_json()
        profile_id = data.get("profile_id")
        folder = data.get("folder", "audio_prompts")
        filename = data.get("filename")
        transcription = data.get("transcription")
        
        if not all([profile_id, filename, transcription]):
            return jsonify({"error": "Missing parameters"}), 400
            
        audio_dir = os.path.join(DATA_DIR, "profiles", profile_id, folder)
        ground_truth = ""
        
        # Try metadata.jsonl
        jsonl_path = os.path.join(audio_dir, "metadata.jsonl")
        if os.path.exists(jsonl_path):
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        f_name = os.path.basename(obj.get('sentence') or obj.get('audio') or '')
                        if f_name == filename:
                            ground_truth = obj.get('text', '')
                            break
                    except: continue
        
        # Try CSV if not found
        if not ground_truth:
            csv_files = [f for f in os.listdir(audio_dir) if f.endswith('.csv') and 'evaluation' not in f]
            if csv_files:
                csv_path = os.path.join(audio_dir, csv_files[0])
                with open(csv_path, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f, delimiter='|', quotechar='"')
                    for row in reader:
                        if len(row) >= 2 and row[0].strip('"') == filename:
                            ground_truth = row[1].strip('"')
                            break
        
        if not ground_truth:
             return jsonify({"error": "Ground truth not found"}), 404
             
        wer = calculate_wer(ground_truth, transcription)
        cer = calculate_cer(ground_truth, transcription)
        
        return jsonify({
            "wer": wer,
            "cer": cer,
            "ground_truth": ground_truth
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500