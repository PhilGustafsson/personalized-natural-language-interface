from flask import Blueprint, request, jsonify
import uuid
from db import DATA_DIR, get_db
import os
import requests

profile_bp = Blueprint('profiles', __name__)

@profile_bp.route("/", methods=["POST"])
def create_profile():
    data = request.json or {}
    profile_id = str(uuid.uuid4())
    username = (data.get("username") or "").strip() if data.get("username") else None

    # Check if username already exists in the filesystem (if username is provided)
    if username:
        profiles_root = os.path.join(DATA_DIR, "profiles")
        safe_name = username.replace("/", "_").strip()
        candidate = os.path.join(profiles_root, safe_name)
        if os.path.exists(candidate):
            return jsonify({
                "error": "Username already exists. Please choose a different name.",
                "message": "Duplicate username"
            }), 409

    conn = get_db()
    # Store username (column added by db.init_db migration if missing)
    conn.execute(
        "INSERT INTO profiles (profile_id, consent, language, device_info, username) VALUES (?, ?, ?, ?, ?)",
        (profile_id, data.get("consent"), data.get("language"), data.get("device_info"), username), # type: ignore
    )
    conn.commit()
    conn.close()

    # Create profile directory with subfolder structure.
    profiles_root = os.path.join(DATA_DIR, "profiles")
    os.makedirs(profiles_root, exist_ok=True)

    # Prefer to create a folder using the username when provided.
    if username:
        safe_name = username.replace("/", "_").strip()
        candidate = os.path.join(profiles_root, safe_name)
        # If a folder with the username already exists, append the uuid to keep it unique.
        if os.path.exists(candidate):
            candidate = os.path.join(profiles_root, f"{safe_name}_{profile_id}")
    else:
        candidate = os.path.join(profiles_root, profile_id)

    os.makedirs(candidate, exist_ok=True)

    # Create subdirectories for audio
    audio_prompts_dir = os.path.join(candidate, 'audio_prompts')
    audio_transcribe_dir = os.path.join(candidate, 'audio_transcribe')
    os.makedirs(audio_prompts_dir, exist_ok=True)
    os.makedirs(audio_transcribe_dir, exist_ok=True)

    # Select base model on STT service to ensure it's ready
    try:
        select_payload = {
            "model_dir": "models/kb-whisper-large"        
        }
        print(f"Selecting base model on STT service: {select_payload}")
        requests.post("http://stt-app:5080/select_model", json=select_payload, timeout=5)
    except Exception as e:
        print(f"Warning: Failed to select base model on STT service: {e}")

    return jsonify({
        "profile_id": profile_id,
        "username": username,
        "message": "User profile created."
    })


@profile_bp.route("/<profile_id>", methods=["DELETE"])
def delete_profile(profile_id):
    conn = get_db()
    conn.execute("DELETE FROM profiles WHERE profile_id=?", (profile_id,))
    conn.execute("DELETE FROM recordings WHERE profile_id=?", (profile_id,))
    conn.execute("DELETE FROM models WHERE profile_id=?", (profile_id,))
    conn.execute("DELETE FROM prompts WHERE profile_id=?", (profile_id,))
    conn.commit()
    conn.close()
    # Optionally delete filesystem data
    user_dir = os.path.join(DATA_DIR, "profiles", profile_id)
    if os.path.exists(user_dir):
        import shutil
        shutil.rmtree(user_dir)
    return jsonify({"message": "User data deleted successfully."})
@profile_bp.route("/", methods=["GET"])
def list_profiles_db_and_files():
    """Return a simple listing of profiles stored in the database (ids/usernames) and
    the filesystem folders under DATA_DIR/profiles. This keeps the DB POST route at
    the same URL while supporting GET to list filesystem entries as well as DB rows.
    """
    conn = get_db()
    cur = conn.execute("SELECT profile_id, username FROM profiles")
    rows = cur.fetchall()
    conn.close()

    profiles = []
    for r in rows:
        profiles.append({"profile_id": r[0], "username": r[1]})

    # Also list directories under data/profiles
    profiles_root = os.path.join(DATA_DIR, "profiles")
    filesystem = []
    if os.path.exists(profiles_root):
        for entry in os.listdir(profiles_root):
            p = os.path.join(profiles_root, entry)
            if os.path.isdir(p):
                filesystem.append(entry)

    return jsonify({"db": profiles, "filesystem": filesystem})


@profile_bp.route('/filesystem', methods=['GET'])
def list_profiles_filesystem():
    """Return a list of folder names under DATA_DIR/profiles."""
    profiles_root = os.path.join(DATA_DIR, "profiles")
    items = []
    if os.path.exists(profiles_root):
        for name in os.listdir(profiles_root):
            path = os.path.join(profiles_root, name)
            if os.path.isdir(path):
                items.append(name)
    return jsonify(items)


@profile_bp.route('/filesystem/<name>', methods=['DELETE'])
def delete_profile_filesystem(name):
    """Safely delete a profile folder by name (no path traversal).
    This endpoint only removes the filesystem folder and does not touch the DB.
    """
    # Prevent path traversal and enforce simple name
    if ".." in name or "/" in name or "\\" in name:
        return jsonify({"message": "Invalid profile name."}), 400

    profiles_root = os.path.join(DATA_DIR, "profiles")
    target = os.path.join(profiles_root, name)
    # Ensure target is under profiles_root
    try:
        target_real = os.path.realpath(target)
        root_real = os.path.realpath(profiles_root)
        if not target_real.startswith(root_real):
            return jsonify({"message": "Invalid profile path."}), 400
    except Exception:
        return jsonify({"message": "Invalid path."}), 400

    if os.path.exists(target) and os.path.isdir(target):
        import shutil
        shutil.rmtree(target)
        return jsonify({"message": f"Filesystem profile '{name}' deleted."})
    else:
        return jsonify({"message": "Profile folder not found."}), 404