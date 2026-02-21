# Backend Application
Note: some of these features have not been implemented yet, and is only here as a general sketch of what it will look like and may change.

## Backend Project Structure (general idea)
The project structure of the backend is presented below. It is for now a general sketch of what it will look like and may change.
```
app/
│
├── backend/                      # Main Flask app package
│   │ 
│   ├── routes/                   # Group endpoints by feature
│   │   ├── profiles.py           # Profile create/delete
│   │   ├── audio.py              # Audio upload + QC
│   │   ├── finetuning.py         # Fine-tuning job endpoints
│   │   ├── models.py             # Export/import adapters, download base models
│   │   └── inference.py          # TTS and STT endpoints
│   │
│   ├── services/                 # Logic helpers (not tied to Flask)
│   │   ├── qc.py                 # Audio quality checks (SNR, clipping, silence)
│   │   ├── stt.py                # Whisper / KB-Whisper inference
│   │   ├── tts.py                # Orpheus / Piper inference
│   │   ├── jobs.py               # Fine-tuning orchestration (Unsloth, etc.)
│   │   └── storage.py            # File management utilities
│   │ 
│   ├── db.py                     # SQLite connection helpers
│   ├── app.py                    # Flask entrypoint
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── README.md                 # This README file.
│   │
│   ├── data/                        # Mounted volume for profile data & DB
│   │   ├── app.db                   # SQLite DB (auto-created)
│   │   └── profile/
│   │       └── <profile_id>/
│   │           ├── audio/       # Original uploaded recordings
│   │           ├── segments/        # Auto-segmented utterances
│   │           ├── transcripts/     # JSON or text files with corrected transcriptions
│   │           └── models/          # Fine-tuned adapters
│   │
│   └── models/                      # (Optional) base STT & TTS models cache
│       ├── whisper-v3/
│       └── piper-swedish/
│
├── frontend/                    # The frontend part of the app. (not documented here)
│   ├── ...
│   └── ...
│
└── README.md                    # Project docs

```
## API Endpoints (`routes/`) 
This directory contains the API endpoints for the backend. It uses the basic REST calls (`GET`, `POST`, `PUT`, `DELETE`). The endpoints have been grouped together based on feature. 

Below is an example from `profiles.py` where a `POST` call to `/api/profiles/` creates a new profile.
```python
@profile_bp.route("/", methods=["POST"])
def create_profile():
    # Get the json from POST sent from the frontend.
    data = request.json
    
    # data base calls not in example.
    # ...
    # ...

    # Create json response. The frontend recieves this.
    return jsonify({"profile_id": profile_id, "message": "User profile created."})
```

## Database Management
Database management is done with SQLite. The `db.py` python script contains code for accessing the database object (`get_db`) and for initializing a new database (`init_db`). Initialization is done automatically by `db.py`. 

Below is an example from `profiles.py` where a `DELETE` call to `/api/profiles/<profile_id>` deletes a profile with a given profile ID.
```python
@profile_bp.route("/<profile_id>", methods=["DELETE"])
def delete_profile(profile_id):

    # Delete from profile data relevant tables.
    conn = get_db()
    conn.execute("DELETE FROM profiles WHERE profile_id=?", (profile_id,))
    conn.execute("DELETE FROM recordings WHERE profile_id=?", (profile_id,))
    conn.execute("DELETE FROM models WHERE profile_id=?", (profile_id,))
    conn.commit() # Commit changes to DB.
    conn.close()

    # deleting user files from file system not in example
    # ...
    # ...

    # Create json response for frontend. 
    return jsonify({"message": "User data deleted successfully."})
```

## Profiles (Personalization Data)
Each personalization is stored as a unique profile. A profile is identified by a unique `profile_id` (UUID4). Each profile has its own directory in the file system (`/data/<profile_id>/`) where recordings, transcripts and fine-tuned models are stored.

A profile ID looks, for example, like: `d79234b5-896f-4184-8228-7cdc1368754b`.

## File System for Data Storage
Data is stored on the local file system and is mounted as a volume when starting the Docker container. Both the `/data`and `/models` directories are mounted. These are listed in the `.gitignore` since they should not be part of the repo (the models are too huge and data belongs to the user). These directories are created automatically if they do not exist when starting the backend.

## Services (`services/`)
This essentially where business logic happens:
* Fine-tuning scripts.
* Audio quality checks.
* TTS and STT inference.

(Not implemented yet.)