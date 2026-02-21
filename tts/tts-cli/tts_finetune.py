"""
TTS CLI for FastAPI service. Trigger finetuning using the /finetune endpoint.
Example usage:
python3 tts_finetune.py --output-dir ./models/checkpoints/chatterbox_finetuned_swedish --metadata-file ../chatterbox/data/david/metadata.txt --dataset-dir ../chatterbox/data/david --local-model-dir None --train-split-name train --eval-split-size 0.1 --num-train-epochs 10 --per-device-train-batch-size 2 --gradient-accumulation-steps 4 --learning-rate 1e-5 --warmup-steps 200 --weight-decay 0.01 --max-grad-norm 1.0 --logging-steps 10 --evaluation-strategy epoch --save-strategy epoch --save-total-limit 5 --report-to tensorboard --do-train true --do-eval true --dataloader-pin-memory false --eval-on-start true --label-names labels_speech --text-column-name text_scribe --language-id sv --dataloader-num-workers 0 --seed 42
"""

import argparse
import requests
import os

def finetune(args, api_url="http://localhost:8000/finetune"):
    # Helper to prefix with 'data/' for container paths
    def ensure_data_prefix(path):
        if not path.startswith("data/"):
            return f"data/{path.lstrip('./')}"
        return path
    # Copy metadata_file and dataset_dir to TTS/app-data/
    shared_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app-data"))
    os.makedirs(shared_data_dir, exist_ok=True)

    import shutil
    metadata_file_src = os.path.abspath(args.metadata_file)
    dataset_dir_src = os.path.abspath(args.dataset_dir)
    dataset_dir_dst = os.path.join(shared_data_dir, os.path.basename(dataset_dir_src))

    # Check if metadata_file is inside dataset_dir
    metadata_inside_dataset = False
    if os.path.isdir(dataset_dir_src):
        metadata_inside_dataset = os.path.commonpath([metadata_file_src, dataset_dir_src]) == dataset_dir_src

    # Copy dataset_dir (if it's a directory)
    if os.path.isdir(dataset_dir_src):
        if not os.path.exists(dataset_dir_dst):
            shutil.copytree(dataset_dir_src, dataset_dir_dst)
    else:
        if not os.path.exists(dataset_dir_dst):
            shutil.copy2(dataset_dir_src, dataset_dir_dst)

    # Only copy metadata_file if not inside dataset_dir
    if not metadata_inside_dataset:
        metadata_file_dst = os.path.join(shared_data_dir, os.path.basename(metadata_file_src))
        if not os.path.exists(metadata_file_dst):
            shutil.copy2(metadata_file_src, metadata_file_dst)

    # Set payload paths
    if metadata_inside_dataset:
        # Path relative to dataset_dir inside container
        metadata_file_rel = os.path.relpath(metadata_file_src, dataset_dir_src)
        payload_metadata_file = ensure_data_prefix(os.path.join(os.path.basename(dataset_dir_dst), metadata_file_rel))
    else:
        payload_metadata_file = ensure_data_prefix(os.path.basename(metadata_file_src))

    payload = {
        "output_dir": args.output_dir,
        "metadata_file": payload_metadata_file,
        "dataset_dir": ensure_data_prefix(os.path.basename(dataset_dir_dst)),
        "local_model_dir": args.local_model_dir,
        "train_split_name": args.train_split_name,
        "eval_split_size": args.eval_split_size,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup_steps,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "logging_steps": args.logging_steps,
        "evaluation_strategy": args.evaluation_strategy,
        "save_strategy": args.save_strategy,
        "save_total_limit": args.save_total_limit,
        "report_to": args.report_to,
        "do_train": args.do_train.lower() == "true",
        "do_eval": args.do_eval.lower() == "true",
        "dataloader_pin_memory": args.dataloader_pin_memory.lower() == "true",
        "eval_on_start": args.eval_on_start.lower() == "true",
        "label_names": args.label_names,
        "text_column_name": args.text_column_name,
        "language_id": args.language_id,
        "dataloader_num_workers": args.dataloader_num_workers,
        "seed": args.seed
    }
    response = requests.post(api_url, json=payload)
    if response.status_code == 200:
        print(f"Finetuning started. Output: {response.json()}")
        # Cleanup copied files after finetuning
        try:
            # Remove copied dataset directory
            if os.path.isdir(dataset_dir_dst):
                shutil.rmtree(dataset_dir_dst)
            # Remove copied metadata file if it was copied separately
            if not metadata_inside_dataset:
                metadata_file_dst = os.path.join(shared_data_dir, os.path.basename(metadata_file_src))
                if os.path.exists(metadata_file_dst):
                    os.remove(metadata_file_dst)
        except Exception as cleanup_err:
            print(f"Warning: Cleanup failed: {cleanup_err}")
    else:
        print(f"Error: {response.status_code} - {response.text}")

def main():
    parser = argparse.ArgumentParser(description="TTS CLI for FastAPI service: Finetune model.")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--metadata-file", type=str, required=True)
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--local-model-dir", type=str, default="None")
    parser.add_argument("--train-split-name", type=str, default="train")
    parser.add_argument("--eval-split-size", type=float, default=0.1)
    parser.add_argument("--num-train-epochs", type=int, default=10)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--evaluation-strategy", type=str, default="epoch")
    parser.add_argument("--save-strategy", type=str, default="epoch")
    parser.add_argument("--save-total-limit", type=int, default=5)
    parser.add_argument("--report-to", type=str, default="tensorboard")
    parser.add_argument("--do-train", type=str, default="true")
    parser.add_argument("--do-eval", type=str, default="true")
    parser.add_argument("--dataloader-pin-memory", type=str, default="false")
    parser.add_argument("--eval-on-start", type=str, default="true")
    parser.add_argument("--label-names", type=str, default="labels_speech")
    parser.add_argument("--text-column-name", type=str, default="text_scribe")
    parser.add_argument("--language-id", type=str, default="sv")
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--api-url", type=str, default="http://localhost:8000/finetune", help="API endpoint URL")

    args = parser.parse_args()
    #print(args.dataset_dir)
    #print(args.metadata_file)
    finetune(args, args.api_url)

if __name__ == "__main__":
    main()
