from flask import Blueprint, request, jsonify
import audio_processing.quality_checks as qc
import services.storage as st
import os
import json

audio_bp = Blueprint('audio', __name__)

@audio_bp.route("/quality-check-clip/", methods=["POST"])
def quality_check_clip():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    qc_passed, qc_score = qc.quality_check_clip(file)
    return jsonify({"passed": qc_passed, "score": qc_score})


@audio_bp.route("/save-transcribe-recording/", methods=["POST"])
def save_transcribe_recording():
    """Save a transcribe recording to the audio_transcribe folder."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    profile_id = request.form.get("profileID")
    base_name = request.form.get("base_name")  # Get custom filename if provided
    
    if not profile_id:
        return jsonify({"error": "No profile ID provided"}), 400

    try:
        filepath = st.save_transcribe_recording(profile_id, file, base_name)
        filename = filepath.split('/')[-1]  # Get just the filename
        
        return jsonify({
            "message": "Recording saved successfully.",
            "filepath": filepath,
            "filename": filename
        }), 200
    except Exception as e:
        return jsonify({"error": f"Failed to save recording: {str(e)}"}), 500


@audio_bp.route("/save-script/", methods=["POST"])
def save_script():
    """Save a user-written script to a JSON file."""
    try:
        profile_id = request.json.get("profileID")
        script_name = request.json.get("script_name")
        script_text = request.json.get("script_text")
        
        if not profile_id:
            return jsonify({"error": "No profile ID provided"}), 400
        if not script_name:
            return jsonify({"error": "No script name provided"}), 400
        if not script_text:
            return jsonify({"error": "No script text provided"}), 400
        
        filepath, script_id = st.save_script(profile_id, script_name, script_text)
        
        return jsonify({
            "message": "Script saved successfully.",
            "filepath": filepath,
            "script_id": script_id
        }), 200
    except Exception as e:
        return jsonify({"error": f"Failed to save script: {str(e)}"}), 500


@audio_bp.route("/get-scripts/", methods=["GET"])
def get_scripts():
    """Get all saved scripts for a profile."""
    try:
        profile_id = request.args.get("profileID")
        
        if not profile_id:
            return jsonify({"error": "No profile ID provided"}), 400
        
        scripts = st.get_saved_scripts(profile_id)
        
        return jsonify({
            "message": "Scripts retrieved successfully.",
            "scripts": scripts
        }), 200
    except Exception as e:
        return jsonify({"error": f"Failed to retrieve scripts: {str(e)}"}), 500


@audio_bp.route("/save-own-prompt-recording/", methods=["POST"])
def save_own_prompt_recording():
    """Save a recording of a user-written script with script metadata and CSV logging."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    profile_id = request.form.get("profileID")
    base_name = request.form.get("base_name")
    script_name = request.form.get("script_name")
    script_text = request.form.get("script_text")
    
    if not profile_id:
        return jsonify({"error": "No profile ID provided"}), 400
    if not base_name:
        return jsonify({"error": "No base name provided"}), 400
    if not script_name:
        return jsonify({"error": "No script name provided"}), 400
    if not script_text:
        return jsonify({"error": "No script text provided"}), 400

    try:
        filepath, csv_path = st.save_own_prompt_recording(
            profile_id, file, base_name, script_name, script_text
        )
        filename = filepath.split('/')[-1]  # Get just the filename
        
        return jsonify({
            "message": "Recording and script data saved successfully.",
            "filepath": filepath,
            "filename": filename,
            "csv_path": csv_path
        }), 200
    except Exception as e:
        return jsonify({"error": f"Failed to save recording: {str(e)}"}), 500



@audio_bp.route('/save-domain-recording/', methods=['POST'])
def save_domain_recording():
    """Save a domain recording to audio_domain folder."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    profile_id = request.form.get('profileID')
    base_name = request.form.get('base_name')
    domain_id = request.form.get('domainID')

    if not profile_id:
        return jsonify({'error': 'No profile ID provided'}), 400
    if not domain_id:
        return jsonify({'error': 'No domainID provided'}), 400

    try:
        filepath = st.save_domain_recording(profile_id, file, domain_id, base_name=base_name)
        filename = os.path.basename(filepath)
        # optional csv logging: try to get domain text from assets
        domain_text = ''
        try:
            base_dir = os.path.join(os.path.dirname(__file__), '..', 'assets', 'domains')
            files = os.listdir(base_dir)
            for f in files:
                if f.startswith(f'domain{domain_id}'):
                    try:
                        with open(os.path.join(base_dir, f), 'r', encoding='utf-8') as fh:
                            content = json.load(fh)
                            domain_text = content.get('domain_text', '')
                    except Exception:
                        domain_text = ''
                    break
        except Exception:
            domain_text = ''
        # Attempt to log to csv; skip if fails
        try:
            st.log_domain_audio_to_csv(profile_id, filename, domain_id, domain_text)
        except Exception:
            pass

        return jsonify({
            'message': 'Recording saved successfully.',
            'filepath': filepath,
            'filename': filename
        }), 200
    except Exception as e:
        return jsonify({'error': f'Failed to save recording: {str(e)}'}), 500


@audio_bp.route('/domains/<domain_name>/', methods=['GET'])
def get_domain(domain_name: str):
    """Return domain JSON from backend assets/domains/"""
    try:
        base_dir = os.path.join(os.path.dirname(__file__), '..', 'assets', 'domains')
        # Map friendly names or domainX to actual filenames.
        # Support domain1|domain2|domain3 or full filename
        candidates = [
            os.path.join(base_dir, f'{domain_name}.json'),
            os.path.join(base_dir, f'{domain_name}_*.json'),
            os.path.join(base_dir, f'domain{domain_name}_ai.json'),
        ]
        # Look for matching known files
        files = os.listdir(base_dir)
        match_file = None
        for f in files:
            if domain_name in f or f.startswith(domain_name) or f.startswith(f'domain{domain_name}'):
                match_file = os.path.join(base_dir, f)
                break
        if not match_file:
            return jsonify({'error': 'Domain not found'}), 404

        with open(match_file, 'r', encoding='utf-8') as fh:
            content = json.load(fh)
        return jsonify(content), 200
    except Exception as e:
        return jsonify({'error': f'Failed to load domain: {str(e)}'}), 500