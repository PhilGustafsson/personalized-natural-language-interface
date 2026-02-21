"""
TTS CLI for FastAPI service. Transcribe text to speech using the /synthesize endpoint.
Example usage: 
python3 tts_synthesize.py "Hello world" --output-file ./cli-data/transcribed/new-audio.wav --use-finetuned false

python3 tts_synthesize.py "När något går fel då måste rätta till det" --output-file ./cli-data/synthesized/new-audio-ft-vp-test.wav --use-finetuned true --voice-sample ./cli-data/davi
d/soundfiles/david_sentence03.wav

"""

import argparse
import requests
import os

def synthesize(text, output_file, use_finetuned=True, api_url="http://localhost:8000/synthesize"):
    payload = {
        "text": text,
        "output_dir": None,
        "language": "sv",
        "exaggeration": 0.5,
        "cfg_weight": 0.5,
        "filename": os.path.basename(output_file),
        "use_finetuned": use_finetuned
    }

    # If a voice sample is provided, create and set the voice profile
    if synthesize.voice_sample_path:
        voice_sample_path = synthesize.voice_sample_path
        files = {"wav_file": open(voice_sample_path, "rb")}
        data = {
            "name": os.path.splitext(os.path.basename(voice_sample_path))[0],
            "use_finetuned": use_finetuned,
            "output_dir": "./models/voices"
        }
        resp = requests.post(synthesize.voice_profile_url, files=files, data=data)
        if resp.status_code == 200:
            profile_path = resp.json().get("profile_path")
            if profile_path:
                # Set the voice profile as active
                set_data = {"profile_path": profile_path, "use_finetuned": use_finetuned}
                set_resp = requests.post(synthesize.apply_profile_url, data=set_data)
                if set_resp.status_code == 200:
                    print(f"Voice profile set: {profile_path}")
                else:
                    print(f"Failed to set voice profile: {set_resp.text}")
            else:
                print(f"Failed to create voice profile: {resp.text}")
        else:
            print(f"Failed to upload voice sample: {resp.text}")

    response = requests.post(api_url, json=payload, stream=True)
    if response.status_code == 200:
        output_dir = os.path.dirname(os.path.abspath(output_file))
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        with open(output_file, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Audio saved to {output_file}")
    else:
        print(f"Error: {response.status_code} - {response.text}")

def main():
    parser = argparse.ArgumentParser(description="TTS CLI for FastAPI service: Synthesize text to speech.")
    parser.add_argument("text", type=str, help="Text to synthesize")
    parser.add_argument("--output-file", type=str, required=True, help="Output WAV file")
    parser.add_argument("--use-finetuned", type=str, default="true", help="Use finetuned model (true/false)")
    parser.add_argument("--api-url", type=str, default="http://localhost:8000/synthesize", help="API endpoint URL")
    parser.add_argument("--voice-sample", type=str, default=None, help="Path to voice sample WAV file")

    args = parser.parse_args()
    use_finetuned = args.use_finetuned.lower() == "true"
    # Attach extra arguments to synthesize function for voice profile
    synthesize.voice_sample_path = args.voice_sample
    synthesize.voice_profile_url = "http://localhost:8000/voice-profile"
    synthesize.apply_profile_url = "http://localhost:8000/apply-voice-profile"
    synthesize(args.text, args.output_file, use_finetuned, args.api_url)

if __name__ == "__main__":
    main()
