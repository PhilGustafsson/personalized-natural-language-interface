from flask import Blueprint, request, jsonify
import uuid
from db import DATA_DIR, get_db
import os
import services.storage as st
import services.prompting as pro
import requests
import json

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

    # Build a safe base name for the file using the prompt id.
    # Example: prompt id 'sv-standard-1' -> base_name 'prompt1'.
    try:
        segment = prompt_id.split('-')[-1]
        base_name = f"prompt{segment}"
    except Exception:
        # Fallback to using prompt id directly
        base_name = prompt_id

    # Save the file to the appropriate directory. storage.save_recording will
    # append a timestamp and create a safe filename.
    recording_id = str(uuid.uuid4())
    filepath = st.save_recording(profile_id, recording_id, file, base_name=base_name)
    
    # Extract just the filename from the full path
    audio_filename = os.path.basename(filepath)
    
    # Log the audio and prompt to CSV
    st.log_audio_to_csv(profile_id, audio_filename, prompt_text)

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

@prompts_bp.route("/sv-standard2/", methods=["GET"])
def get_sv_standard_prompts2():
    # Load the secondary prompt file directly
    try:
        prompts_path = os.path.join(os.path.dirname(__file__), '..', 'assets', 'prompts', 'sv-standard-prompts.json')
        prompts_path = os.path.normpath(prompts_path)
        with open(prompts_path, 'r', encoding='utf-8') as f:
            prompts = json.load(f)
        return jsonify(prompts)
    except Exception as e:
        return jsonify({"error": f"Failed to load prompts: {str(e)}"}), 500

@prompts_bp.route("/generate-prompts", methods=["POST"])
def generate_prompts():
    data = request.get_json()
    domain = data.get('domain')
    # DEMO: Temporary change — do not call the local LLM (Ollama).
    # Instead, load and return the local `sv-standard-prompts.json` file from assets.
    # The original LLM-based logic is commented out below for reference.
    try:
        prompts_path = os.path.join(os.path.dirname(__file__), '..', 'assets', 'prompts', 'sv-standard-prompts.json')
        prompts_path = os.path.normpath(prompts_path)
        with open(prompts_path, 'r', encoding='utf-8') as f:
            prompts = json.load(f)
        return jsonify(prompts)
    except Exception as e:
        print(f"Failed to load local prompts: {e}")
        return jsonify({"error": f"Failed to load local prompts: {str(e)}"}), 500

# -----------------------------------------------------------------------------
# Original LLM-based implementation (commented out for demo). Do NOT delete.
# The code below attempted to call a local Ollama instance to generate domain
# specific prompts. It was commented out to provide a temporary local JSON
# fallback. Re-enable by removing leading '#' characters and adapting paths.
#
#    if not domain:
#        return jsonify({"error": "Domain is required"}), 400
#
#    # Try to use local LLM (Ollama)
#    try:
#        # We use 'llama3.2' as it is small and efficient. 
#        # Ensure you have run `ollama pull llama3.2` in your terminal.
#        model_name = "llama3.2" 
#        
#        system_prompt = "You are an expert in technical domains and the Swedish language. You generate high-quality, vocabulary-rich training data for speech-to-text models in Swedish."
#        user_prompt = f"Generate 10 short, distinct sentences in Swedish about '{domain}'. The sentences should be suitable for reading aloud. IMPORTANT: Use specific technical terminology, jargon, and concepts unique to {domain}. Avoid generic sentences. Do not just repeat the word '{domain}'. Return ONLY a JSON array of strings. Example: [\\"Det neurala nätverket konvergerar efter femtio epoker.\\", \\\"Den aerodynamiska lyftkoefficienten är kritisk för flygstabilitet.\\\"]"
#
#        print(f"Requesting prompts from Ollama ({model_name}) for domain: {domain}")
#        
#        # Try ollama service name first (Docker), then localhost, then host.docker.internal
#        ollama_urls = [
#            'http://ollama:11434/api/generate', 
#            'http://localhost:11434/api/generate', 
#            'http://host.docker.internal:11434/api/generate'
#        ]
#        response = None
#        connection_error = None
#
#        for url in ollama_urls:
#            try:
#                print(f"Trying to connect to Ollama at: {url}")
#                response = requests.post(url, json={
#                    "model": model_name,
#                    "prompt": user_prompt,
#                    "system": system_prompt,
#                    "stream": False,
#                    "format": "json"
#                }, timeout=60)
#                if response.status_code == 200:
#                    break # Success
#            except requests.exceptions.ConnectionError as e:
#                print(f"Failed to connect to {url}")
#                connection_error = e
#                continue
#
#        if response is None:
#             # If we exhausted all URLs and still have no response
#             raise connection_error if connection_error else Exception("Could not connect to any Ollama URL")
#
#        if response.status_code == 200:
#            result = response.json()
#            generated_text = result.get('response', '')
#            print(f"Ollama response: {generated_text}")
#            
#            # Clean up markdown code blocks if present (common with LLMs)
#            cleaned_text = generated_text.strip()
#            if cleaned_text.startswith("```json"):
#                cleaned_text = cleaned_text[7:]
#            elif cleaned_text.startswith("```"):
#                cleaned_text = cleaned_text[3:]
#            
#            if cleaned_text.endswith("```"):
#                cleaned_text = cleaned_text[:-3]
#            
#            cleaned_text = cleaned_text.strip()
#
#            try:
#                # Try to find JSON array in the text if direct parsing fails or if it's wrapped in text
#                import re
#                import ast
#                import sys
#                
#                # Log the raw text for debugging
#                print(f"DEBUG: Raw Ollama text: {generated_text}", file=sys.stderr)
#
#                json_match = re.search(r'\[.*\]', cleaned_text, re.DOTALL)
#                if json_match:
#                    cleaned_text = json_match.group(0)
#
#                try:
#                    parsed = json.loads(cleaned_text)
#                except json.JSONDecodeError:
#                    # Fallback to ast.literal_eval for single quotes or loose JSON
#                    try:
#                        parsed = ast.literal_eval(cleaned_text)
#                    except (ValueError, SyntaxError):
#                        raise json.JSONDecodeError("Could not parse via json or ast", cleaned_text, 0)
#
#                prompts = []
#
#                if isinstance(parsed, list):
#                    prompts = parsed
#                elif isinstance(parsed, dict):
#                    # Sometimes LLMs wrap the list in a key like {"prompts": [...]} 
#                    # Try to find the first list value in the dict
#                    found_list = False
#                    for key, value in parsed.items():
#                        if isinstance(value, list):
#                            prompts = value
#                            found_list = True
#                            break
#                    
#                    if not found_list:
#                        # Fallback: maybe the keys are the prompts? (Ollama sometimes does this)
#                        # or values are the prompts?
#                        # Let's collect all strings from keys and values that look like sentences.
#                        candidates = []
#                        for k, v in parsed.items():
#                            # Check if key is a long string (likely a prompt)
#                            if isinstance(k, str) and len(k) > 15:
#                                candidates.append(k)
#                            # Check if value is a long string
#                            if isinstance(v, str) and len(v) > 15:
#                                candidates.append(v)
#                        
#                        if candidates:
#                            prompts = candidates
#
#                # Filter to ensure we only have strings
#                final_prompts = []
#                for p in prompts:
#                    if isinstance(p, (str, int, float)):
#                        final_prompts.append(str(p))
#                    elif isinstance(p, dict):
#                        # If it's a dict, try to find a string value that looks like a prompt
#                        # or just take the first string value
#                        for v in p.values():
#                            if isinstance(v, str):
#                                final_prompts.append(v)
#                                break
#
#                if final_prompts:
#                    # Ensure we have at most 10
#                    return jsonify(final_prompts[:10])
#                else:
#                    print(f"Ollama returned valid JSON but could not extract a list of strings. Parsed: {parsed}")
#                    return jsonify({"error": "Ollama returned invalid format. Expected a list of strings."}), 500
#
#            except json.JSONDecodeError:
#                print(f"Failed to parse JSON from Ollama: {generated_text}")
#                return jsonify({"error": "Failed to parse JSON from Ollama."}), 500
#        else:
#            print(f"Ollama returned status code {response.status_code}: {response.text}")
#            return jsonify({"error": f"Ollama returned status code {response.status_code}"}), 500
#            
#    except requests.exceptions.ConnectionError:
#        print("Could not connect to Ollama. Is it running on port 11434?")
#        return jsonify({"error": "Could not connect to local LLM (Ollama). Is it running?"}), 503
#    except Exception as e:
#        print(f"Local LLM generation failed: {e}")
#        return jsonify({"error": f"Local LLM generation failed: {str(e)}"}), 500
