"""
TTS CLI for FastAPI service. Evaluate TTS model using the /evaluate endpoint.
Example usage:
python3 tts_evaluate.py --sentences_file ./eval_sentences.txt --audio_output_directory ./cli-data/eval_audio
"""

import argparse
import requests
import os

def evaluate(sentences_file, audio_output_directory=None, api_url="http://localhost:8000/evaluate"):
    # Read sentences from file
    with open(sentences_file, "r", encoding="utf-8") as f:
        eval_sentences = [line.strip() for line in f if line.strip()]

    payload = {
        "eval_sentences": eval_sentences,
        "language": "sv"
    }
    
    # Define path in the container: /data/evaluation/
    payload["output_dir_base"] = os.path.join("data", "evaluation", "base")
    payload["output_dir_ft"] = os.path.join("data", "evaluation", "finetuned")

    # Make request.
    response = requests.post(api_url, json=payload)
    if response.status_code == 200:
        print("Evaluation results:")
        print(response.json())

        # Copy audio output directory from container to target directory (if specified).
        if audio_output_directory:
            base_src = os.path.join("..", "app-data", "evaluation", "base")
            ft_src = os.path.join("..", "app-data", "evaluation", "finetuned")
            base_dst = os.path.join(audio_output_directory, "base")
            ft_dst = os.path.join(audio_output_directory, "finetuned")

            os.makedirs(base_dst, exist_ok=True)
            os.makedirs(ft_dst, exist_ok=True)

            # Copy files from container paths to local paths.
            for src, dst in [(base_src, base_dst), (ft_src, ft_dst)]:
                if os.path.exists(src):
                    for file_name in os.listdir(src):
                        full_file_name = os.path.join(src, file_name)
                        if os.path.isfile(full_file_name):
                            dest_file = os.path.join(dst, file_name)
                            with open(full_file_name, "rb") as fsrc:
                                with open(dest_file, "wb") as fdst:
                                    fdst.write(fsrc.read())
                    print(f"Copied audio files from {src} to {dst}")
                else:
                    print(f"Source directory {src} does not exist.")

    else:
        print(f"Error: {response.status_code} - {response.text}")

def main():
    parser = argparse.ArgumentParser(description="TTS CLI for FastAPI service: Evaluate model.")
    parser.add_argument("--sentences_file", type=str, required=True, help="Path to file with sentences to evaluate")
    parser.add_argument("--audio_output_directory", type=str, default=None, help="Directory for output audio files")
    parser.add_argument("--api_url", type=str, default="http://localhost:8000/evaluate", help="API endpoint URL")

    args = parser.parse_args()
    evaluate(args.sentences_file, args.audio_output_directory, args.api_url)

if __name__ == "__main__":
    main()
