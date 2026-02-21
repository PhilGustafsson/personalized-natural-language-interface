import sqlite3
import os

DATA_DIR = os.environ.get("DATA_DIR", "./data")
DB_FILE = os.path.join(DATA_DIR, "app.db")
os.makedirs(DATA_DIR, exist_ok=True)

def get_db():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    cursor = conn.cursor()
    cursor.executescript("""
    CREATE TABLE IF NOT EXISTS profiles (
        profile_id TEXT PRIMARY KEY,
        consent BOOLEAN NOT NULL,
        language TEXT NOT NULL,
        device_info TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS recordings (
        recording_id TEXT PRIMARY KEY,
        profile_id TEXT NOT NULL,
        filepath TEXT NOT NULL,
        duration REAL,
        qc_passed BOOLEAN,
        qc_score REAL,
        qc_by_user BOOLEAN,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (profile_id) REFERENCES profiles(profile_id)
    );
                    
    CREATE TABLE IF NOT EXISTS prompts (
        prompt_id TEXT,
        profile_id TEXT,
        recording_id TEXT NOT NULL,
        text TEXT NOT NULL,
        user_provided BOOLEAN,
        FOREIGN KEY (recording_id) REFERENCES recordings(recording_id),
        FOREIGN KEY (profile_id) REFERENCES profiles(profile_id)
        PRIMARY KEY (prompt_id, profile_id)
    );

    CREATE TABLE IF NOT EXISTS transcripts (
        transcript_id TEXT PRIMARY KEY,
        recording_id TEXT NOT NULL,
        raw_text TEXT,
        corrected_text TEXT,
        confidence REAL,
        FOREIGN KEY (recording_id) REFERENCES recordings(recording_id)
    );

    CREATE TABLE IF NOT EXISTS models (
        model_id TEXT PRIMARY KEY,
        profile_id TEXT NOT NULL,
        model_type TEXT NOT NULL,
        adapter_path TEXT NOT NULL,
        config_json TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (profile_id) REFERENCES profiles(profile_id)
    );
    """)
    conn.commit()

    # Ensure username column exists in profiles table (add if missing)
    cursor.execute("PRAGMA table_info(profiles);")
    cols = [row['name'] for row in cursor.fetchall()]
    if 'username' not in cols:
        try:
            cursor.execute("ALTER TABLE profiles ADD COLUMN username TEXT;")
            conn.commit()
        except Exception:
            pass

    # Ensure qc columns exist in recordings table (add if missing)
    cursor.execute("PRAGMA table_info(recordings);")
    rec_cols = [row['name'] for row in cursor.fetchall()]
    
    if 'qc_passed' not in rec_cols:
        try:
            cursor.execute("ALTER TABLE recordings ADD COLUMN qc_passed BOOLEAN;")
            conn.commit()
        except Exception:
            pass
            
    if 'qc_score' not in rec_cols:
        try:
            cursor.execute("ALTER TABLE recordings ADD COLUMN qc_score REAL;")
            conn.commit()
        except Exception:
            pass

    if 'qc_by_user' not in rec_cols:
        try:
            cursor.execute("ALTER TABLE recordings ADD COLUMN qc_by_user BOOLEAN;")
            conn.commit()
        except Exception:
            pass

    conn.close()

init_db()