from flask import Blueprint, request, jsonify
import uuid
from db import DATA_DIR, get_db
import os
import services.storage as st
import services.prompting as pro

prompts_bp = Blueprint('prompts', __name__)

@prompts_bp.route("/save-recording/", methods=["POST"])
def save_prompted_recording():
    """Save a recording for a prompted text."""

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    # Unpack form data.
    file = request.files["file"]
    profile_id = request.form.get("profileID")
    prompt_id = request.form.get("promptID")
    prompt_text = request.form.get("promptText")
    quality_checked_by_user = request.form.get("qualityCheckedByUser") == 'true'
    passed_automatic_quality_check = request.form.get("passedAutomaticQualityCheck") == 'true'
    automatic_quality_score = float(request.form.get("automaticQualityScore", 0))
    user_provided = request.form.get("userProvided", "false") == 'true'

    if not profile_id or not prompt_id or not prompt_text:
        return jsonify({"error": "Missing form data"}), 400

    # Save the file to the appropriate directory.
    recording_id = str(uuid.uuid4())
    filepath = st.save_recording(profile_id, recording_id, file)

    # Save recording metadata to the recordings table
    conn = get_db()
    conn.execute(
        "INSERT INTO recordings (recording_id, profile_id, filepath, duration, qc_passed, qc_score, qc_by_user) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (recording_id, profile_id, filepath, None, passed_automatic_quality_check, automatic_quality_score, quality_checked_by_user) # duration can be set later
    )
    conn.commit()

    # Save prompt to the prompt table
    conn.execute(
        """
        INSERT INTO prompts (
            prompt_id, profile_id, recording_id, text, user_provided
        ) VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(prompt_id, profile_id) DO UPDATE SET
            recording_id  = excluded.recording_id,
            text          = excluded.text,
            user_provided = excluded.user_provided
        """,
        (
            prompt_id,
            profile_id,
            recording_id,
            prompt_text,
            user_provided,
        )
    )
    conn.commit()
    conn.close()

    return jsonify({
        "message": "Recording saved successfully.",
        "filepath": filepath
    })

@prompts_bp.route("/sv-standard/", methods=["GET"])
def get_sv_standard_prompts():
    prompts = pro.get_sv_standard_prompts()
    return prompts.to_json()

