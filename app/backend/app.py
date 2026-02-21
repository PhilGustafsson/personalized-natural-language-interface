from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sock import Sock
import tempfile, os, uuid, sqlite3, io, subprocess, json
from datetime import datetime
import soundfile as sf
import os
import uuid
import sqlite3
from datetime import datetime

# Routes.
from routes.profiles import profile_bp
from routes.models import models_bp
from routes.prompts import prompts_bp
from routes.audio import audio_bp
from routes.finetuning import finetuning
from routes.inference import inference_bp
from routes.llm import llm_bp

# Services.
from audio_processing import quality_checks

# Init Flask
app = Flask(__name__)
CORS(app)  # allow requests from frontend
sock = Sock(app)

# Initialize analyzer
stt_analyzer = quality_checks.STTQualityAnalyzer()

# Register blueprints.
app.register_blueprint(profile_bp, url_prefix='/api/profiles')
app.register_blueprint(models_bp, url_prefix='/api/models')
app.register_blueprint(prompts_bp, url_prefix='/api/prompts')
app.register_blueprint(audio_bp, url_prefix='/api/audio')
app.register_blueprint(finetuning, url_prefix='/api/finetuning')
app.register_blueprint(inference_bp, url_prefix='/api/inference')
app.register_blueprint(llm_bp, url_prefix='/api/llm')

# --- Audio ---

def decode_webm_to_wav_bytes(webm_bytes: bytes) -> bytes:
    if len(webm_bytes) < 100:
        raise Exception(f"Input too small: {len(webm_bytes)} bytes")

    process = subprocess.Popen([
        "ffmpeg", "-f", "webm", "-i", "pipe:0",
        "-f", "wav", "-acodec", "pcm_s16le",
        "-ar", "44100", "-ac", "1", "-y",
        "-loglevel", "error", "pipe:1"
    ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    wav_bytes, stderr = process.communicate(input=webm_bytes)
    if process.returncode != 0:
        raise Exception(f"FFmpeg conversion failed: {stderr.decode()}")

    return wav_bytes


@sock.route('/audio')
def audio_stream(ws):
    while True:
        data = ws.receive()
        if data is None:
            break

        try:
            if isinstance(data, str):
                continue

            # Extract profile_username and prompt_name from the wrapped binary data
            # Format: [username_length_4bytes][username][prompt_length_4bytes][prompt_name][audio_data]
            profile_username = None
            prompt_name = None
            audio_bytes = data
            
            if len(data) > 8:
                try:
                    # Read the username length (first 4 bytes, big-endian)
                    username_length = int.from_bytes(data[:4], byteorder='big')
                    if username_length > 0 and username_length < 256:
                        profile_username = data[4:4+username_length].decode('utf-8')
                        
                        # Read the prompt length (next 4 bytes)
                        prompt_offset = 4 + username_length
                        prompt_length = int.from_bytes(data[prompt_offset:prompt_offset+4], byteorder='big')
                        if prompt_length > 0 and prompt_length < 256:
                            prompt_name = data[prompt_offset+4:prompt_offset+4+prompt_length].decode('utf-8')
                        
                        # Rest is audio data
                        audio_bytes = data[prompt_offset+4+prompt_length:]
                        print(f"Extracted username: {profile_username}, prompt: {prompt_name}, audio_bytes: {len(audio_bytes)}")
                except Exception as e:
                    print(f"Error extracting metadata: {e}")
                    audio_bytes = data

            print(f"Received {len(audio_bytes)} bytes of audio (username: {profile_username}, prompt: {prompt_name})")

            wav_bytes = decode_webm_to_wav_bytes(audio_bytes)

            bio = io.BytesIO(wav_bytes)
            bio.seek(0)
            audio, sr = sf.read(bio)

            analysis = stt_analyzer.analyze_full_quality(audio, sr)

            def convert_for_json(obj):
                if hasattr(obj, 'item'):
                    return obj.item()
                elif isinstance(obj, dict):
                    return {k: convert_for_json(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_for_json(item) for item in obj]
                else:
                    return obj

            response = convert_for_json(analysis)
            response['audio_info'] = {
                'samples': len(audio),
                'sample_rate': int(sr),
                'duration_seconds': float(len(audio) / sr)
            }

            # Save audio to profile folder if profile_username is available
            if profile_username:
                try:
                    from db import DATA_DIR
                    audio_dir = os.path.join(DATA_DIR, 'profiles', profile_username, 'raw_audio')
                    os.makedirs(audio_dir, exist_ok=True)
                    
                    # Create filename with prompt name and timestamp
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
                    if prompt_name:
                        # Use prompt_name in filename (e.g., "prompt_20251111_1205.wav")
                        audio_file = os.path.join(audio_dir, f'{prompt_name}_{timestamp}.wav')
                    else:
                        audio_file = os.path.join(audio_dir, f'audio_{timestamp}.wav')
                    
                    sf.write(audio_file, audio, int(sr))
                    
                    print(f"Audio saved to {audio_file}")
                    response['saved_to'] = audio_file
                except Exception as e:
                    print(f"Error saving audio: {e}")
                    response['save_error'] = str(e)

            print(f"Quality Score: {response['quality_score']:.1f}/100")
            ws.send(json.dumps(response))

        except Exception as e:
            print(f"Error: {str(e)}")
            ws.send(json.dumps({"error": str(e)}))

# Temp code ---------------------------------------------------------

@app.route("/api/data")
def data():
    return jsonify({"message": "Hello from Python backend!"})

@app.route("/api/analyze-audio", methods=["POST"])
def analyze_audio():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    # Save temporarily (or use file.read() for memory processing)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
        file.save(tmp.name)
        filename = tmp.name

    # TODO: Run your audio analysis code here (speech-to-text, ML, etc.)
    print(f"Received audio file: {filename}")

    return jsonify({"message": "Audio received and analyzed!", "filename": filename})

# ----------------------------------------------------------------------


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
