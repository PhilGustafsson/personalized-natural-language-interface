# Defensive storage wrapper that prefers DB username when saving recordings.
import os
from db import DATA_DIR, get_db


def save_recording(profile_id, recording_id, recording_file):
    """Save an uploaded recording file to the filesystem.

    This is a drop-in replacement for the original save_recording. It will
    attempt a DB lookup: if the provided profile_id matches a DB row that has
    a non-empty username, that username is used as the folder name. Otherwise
    the provided profile_id is used as the folder name.
    """

    folder_name = profile_id

    try:
        conn = get_db()
        cur = conn.execute("SELECT username FROM profiles WHERE profile_id=?", (profile_id,))
        row = cur.fetchone()
        if row and row["username"]:
            folder_name = row["username"].replace("/", "_").replace("\\\\", "_")
        conn.close()
    except Exception:
        # fallback to the provided profile_id
        pass

    user_dir = os.path.join(DATA_DIR, "profiles", folder_name, 'audio')
    os.makedirs(user_dir, exist_ok=True)
    filename = f"{recording_id}.wav"
    filepath = os.path.join(user_dir, filename)
    recording_file.save(filepath)
    return filepath
