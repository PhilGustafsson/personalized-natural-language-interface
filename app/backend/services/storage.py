"""File management utility.

Recordings are stored under DATA_DIR/profiles/<folder_name>/audio_prompts or audio_transcribe.
Prefer the username folder when the DB row for the given profile_id contains a username.
"""
import os
import csv
import json
from db import DATA_DIR, get_db
from datetime import datetime


def _get_folder_name(profile_id):
    """Get the folder name for a profile, preferring username if available."""
    folder_name = profile_id

    try:
        conn = get_db()
        cur = conn.execute("SELECT username FROM profiles WHERE profile_id=?", (profile_id,))
        row = cur.fetchone()
        if row and row["username"]:
            folder_name = row["username"].replace("/", "_").replace("\\\\", "_")
        conn.close()
    except Exception:
        pass

    return folder_name


def save_transcribe_recording(profile_id, recording_file, base_name: str | None = None):
    """Save a transcribe recording to audio_transcribe folder as WAV format.

    Args:
        profile_id: either the DB profile_id (uuid) or a folder name (username).
        recording_file: Werkzeug FileStorage-like object with .save()
        base_name: optional filename prefix (no extension); if provided a timestamp will be appended.
    
    Returns:
        filepath: full path to the saved file
    """
    folder_name = _get_folder_name(profile_id)
    
    user_dir = os.path.join(DATA_DIR, "profiles", folder_name, 'audio_transcribe')
    os.makedirs(user_dir, exist_ok=True)

    if base_name:
        # sanitize base_name further and append timestamp
        safe = ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in base_name)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        filename = f"{safe}_{timestamp}.wav"
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        filename = f"recording_{timestamp}.wav"

    filepath = os.path.join(user_dir, filename)
    recording_file.save(filepath)
    return filepath


def save_recording(profile_id, recording_id, recording_file, base_name: str | None = None):
    """Save an uploaded recording file to the audio_prompts folder.

    Args:
        profile_id: either the DB profile_id (uuid) or a folder name (username).
        recording_id: unique id for this recording (used for filename if base_name is not provided).
        recording_file: Werkzeug FileStorage-like object with .save()
        base_name: optional filename prefix (no extension); if provided a timestamp will be appended.
    """
    folder_name = _get_folder_name(profile_id)

    user_dir = os.path.join(DATA_DIR, "profiles", folder_name, 'audio_prompts')
    os.makedirs(user_dir, exist_ok=True)

    if base_name:
        # sanitize base_name further and append timestamp
        safe = ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in base_name)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        filename = f"{safe}_{timestamp}.wav"
    else:
        filename = f"{recording_id}.wav"

    filepath = os.path.join(user_dir, filename)
    recording_file.save(filepath)
    return filepath


def log_audio_to_csv(profile_id, audio_filename, prompt_text):
    """Log audio recording and prompt text to multiple formats for training.
    
    Creates or appends to files in profiles/<folder_name>/audio_prompts/:
    - <folder_name>_data.csv: CSV with pipe delimiter (for legacy compatibility)
    - metadata.jsonl: JSONL format for STT training (Whisper format)
    - metadata.txt: Plain text format for TTS training (Chatterbox format)
    
    Format specifications:
    - CSV: audio_filename|prompt_text (pipe-delimited, quoted)
    - JSONL: {"sentence": "/absolute/path/to/audio.wav", "text": "transcript"}
    - TXT: audio_filename|prompt_text (pipe-delimited, relative filename)
    
    Args:
        profile_id: either the DB profile_id (uuid) or a folder name (username).
        audio_filename: the filename of the saved audio (e.g., "prompt1_20251112_1120.wav").
        prompt_text: the prompt text that was read.
    """
    folder_name = _get_folder_name(profile_id)

    user_dir = os.path.join(DATA_DIR, "profiles", folder_name, 'audio_prompts')
    os.makedirs(user_dir, exist_ok=True)

    csv_filename = f"{folder_name}_data.csv"
    csv_filepath = os.path.join(user_dir, csv_filename)

    # Check if file exists to determine if we need to write headers
    file_exists = os.path.isfile(csv_filepath)

    try:
        with open(csv_filepath, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile, delimiter='|', quoting=csv.QUOTE_ALL)
            
            # Write the audio and prompt data
            writer.writerow([audio_filename, prompt_text])

        # ALSO append to metadata.jsonl for STT training
        jsonl_filename = "metadata.jsonl"
        jsonl_filepath = os.path.join(user_dir, jsonl_filename)
        
        # Construct absolute path for the audio file as seen by the container
        # DATA_DIR is /app/data
        audio_abspath = os.path.join(user_dir, audio_filename).replace('\\', '/')
        
        entry = {"sentence": audio_abspath, "text": prompt_text}
        
        with open(jsonl_filepath, 'a', encoding='utf-8') as jsonlfile:
            jsonlfile.write(json.dumps(entry) + "\n")

        # ALSO append to metadata.txt for TTS training (Chatterbox format)
        # Format: filename|transcript
        metadata_txt_filename = "metadata.txt"
        metadata_txt_filepath = os.path.join(user_dir, metadata_txt_filename)
        
        with open(metadata_txt_filepath, 'a', encoding='utf-8') as txtfile:
            txtfile.write(f"{audio_filename}|{prompt_text}\n")

    except Exception as e:
        print(f"Error logging audio to CSV/JSONL/TXT: {e}")


def save_script(profile_id, script_name: str, script_text: str):
    """Save a user-written script to a JSON file.
    
    Creates or overwrites a JSON file in profiles/<folder_name>/scripts/<script_name>.json
    with the script content.
    
    Args:
        profile_id: either the DB profile_id (uuid) or a folder name (username).
        script_name: the name of the script (used as filename).
        script_text: the script text content.
    
    Returns:
        filepath: full path to the saved script file
        script_id: the script identifier (script_name)
    """
    folder_name = _get_folder_name(profile_id)
    
    user_dir = os.path.join(DATA_DIR, "profiles", folder_name, 'scripts')
    os.makedirs(user_dir, exist_ok=True)
    
    # Sanitize script name
    safe_name = ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in script_name)
    
    filename = f"{safe_name}.json"
    filepath = os.path.join(user_dir, filename)
    
    # Create script object
    script_data = {
        "script_name": script_name,
        "script_text": script_text,
        "created_at": datetime.now().strftime('%Y%m%d_%H%M')
    }
    
    # Save as JSON
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(script_data, f, ensure_ascii=False, indent=2)
    
    return filepath, safe_name


def save_own_prompt_recording(profile_id, recording_file, base_name: str, script_name: str, script_text: str):
    """Save a recording of a user-written script to scripts folder and log to CSV.
    
    Args:
        profile_id: either the DB profile_id (uuid) or a folder name (username).
        recording_file: Werkzeug FileStorage-like object with .save()
        base_name: filename prefix (no extension); timestamp will be appended.
        script_name: the name of the script being recorded.
        script_text: the text of the script being recorded.
    
    Returns:
        filepath: full path to the saved audio file
        csv_path: full path to the CSV log file
    """
    folder_name = _get_folder_name(profile_id)
    
    user_dir = os.path.join(DATA_DIR, "profiles", folder_name, 'scripts')
    os.makedirs(user_dir, exist_ok=True)
    
    # Sanitize base_name and append timestamp
    safe = ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in base_name)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    filename = f"{safe}_{timestamp}.wav"
    
    filepath = os.path.join(user_dir, filename)
    recording_file.save(filepath)
    
    # Log to CSV in the scripts folder
    csv_filename = f"{folder_name}_data.csv"
    csv_filepath = os.path.join(user_dir, csv_filename)
    
    try:
        with open(csv_filepath, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile, delimiter='|')
            writer.writerow([filename, script_text])
    except Exception as e:
        print(f"Error logging to scripts CSV: {e}")
    
    return filepath, csv_filepath


def save_domain_recording(profile_id, recording_file, domain_id: str, base_name: str | None = None):
    """Save a domain recording to audio_domain folder. """
    folder_name = _get_folder_name(profile_id)

    user_dir = os.path.join(DATA_DIR, "profiles", folder_name, 'audio_domain')
    os.makedirs(user_dir, exist_ok=True)

    # Build filename using provided base_name or domain id.
    if base_name:
        safe = ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in base_name)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        filename = f"{safe}_{timestamp}.wav"
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        filename = f"{domain_id}_{timestamp}.wav"

    filepath = os.path.join(user_dir, filename)
    recording_file.save(filepath)
    return filepath


def log_domain_audio_to_csv(profile_id, audio_filename, domain_id, domain_text=""):
    """Log domain audio recording to a CSV inside audio_domain folder.
    Fields: audio_filename|domain_id|domain_text
    """
    folder_name = _get_folder_name(profile_id)
    user_dir = os.path.join(DATA_DIR, "profiles", folder_name, 'audio_domain')
    os.makedirs(user_dir, exist_ok=True)

    csv_filename = f"{folder_name}_domain_data.csv"
    csv_filepath = os.path.join(user_dir, csv_filename)

    try:
        with open(csv_filepath, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile, delimiter='|')
            writer.writerow([audio_filename, domain_id, domain_text])
    except Exception as e:
        print(f"Error logging domain audio to CSV: {e}")


def get_saved_scripts(profile_id):
    """Retrieve all saved scripts for a profile.
    
    Args:
        profile_id: either the DB profile_id (uuid) or a folder name (username).
    
    Returns:
        list of dicts with 'name' and 'text' keys for each saved script
    """
    folder_name = _get_folder_name(profile_id)
    
    user_dir = os.path.join(DATA_DIR, "profiles", folder_name, 'scripts')
    
    scripts = []
    
    if not os.path.isdir(user_dir):
        return scripts
    
    try:
        # Read all .json files in the scripts folder
        for filename in os.listdir(user_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(user_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        scripts.append({
                            'name': data.get('script_name', filename[:-5]),
                            'text': data.get('script_text', '')
                        })
                except Exception as e:
                    print(f"Error reading script {filepath}: {e}")
    except Exception as e:
        print(f"Error reading scripts directory: {e}")
    
    return scripts

