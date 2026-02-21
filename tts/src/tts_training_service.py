"""
TTS Training Service - wrapper around chatterbox fine-tuning for API integration.
Matches the STT training pattern for consistency.
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

# Add chatterbox scripts to path
CHATTERBOX_SCRIPTS = Path(__file__).parent.parent / "chatterbox" / "scripts"
sys.path.insert(0, str(CHATTERBOX_SCRIPTS))


@dataclass
class TTSTrainingConfig:
    """Configuration for TTS training, matching STT pattern."""
    
    # Dataset paths (should be set from profile data)
    metadata_file: str = ""
    dataset_dir: str = ""
    
    # Model paths
    local_model_dir: Optional[str] = None  # If None, downloads from HF
    output_dir: str = "./models/checkpoints/chatterbox_finetuned"
    
    # Training hyperparameters
    num_train_epochs: int = 10
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-5
    warmup_steps: int = 200
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Data split
    eval_split_size: float = 0.1
    train_split_name: str = "train"
    
    # Other settings
    language_id: str = "sv"
    seed: int = 42
    early_stopping_patience: Optional[int] = 3
    
    # Logging
    logging_steps: int = 10
    evaluation_strategy: str = "epoch"
    save_strategy: str = "epoch"
    save_total_limit: int = 5
    report_to: str = "tensorboard"


class TTSTrainingService:
    """Service wrapper for TTS training, matching STT's architecture."""
    
    def __init__(self, config: Optional[TTSTrainingConfig] = None):
        self.config = config or TTSTrainingConfig()
        self.chatterbox_scripts_dir = CHATTERBOX_SCRIPTS
        
    def prepare_dataset(self, manifest_path: str, audio_dir: str, user: str) -> Dict[str, Any]:
        """
        Prepare dataset for training from user recordings.
        
        Args:
            manifest_path: Path to metadata file (metadata.txt format)
            audio_dir: Directory containing audio files
            user: User identifier for tracking
            
        Returns:
            Dict with dataset statistics
        """
        # Update config with user-specific paths
        self.config.metadata_file = manifest_path
        self.config.dataset_dir = audio_dir
        
        # Validate paths exist
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Metadata file not found: {manifest_path}")
        if not os.path.exists(audio_dir):
            raise FileNotFoundError(f"Audio directory not found: {audio_dir}")
            
        # Count samples
        with open(manifest_path, 'r', encoding='utf-8') as f:
            lines = [l.strip() for l in f if l.strip()]
            total_samples = len(lines)
            
        eval_samples = int(total_samples * self.config.eval_split_size)
        train_samples = total_samples - eval_samples
        
        return {
            "loaded": True,
            "manifest_path": manifest_path,
            "audio_dir": audio_dir,
            "user": user,
            "total_samples": total_samples,
            "train_samples": train_samples,
            "eval_samples": eval_samples
        }
    
    def train(self, user: str) -> Dict[str, Any]:
        """
        Run TTS fine-tuning using the configured parameters.
        
        Args:
            user: User identifier
            
        Returns:
            Dict with training results including output_dir and metrics
        """
        # Build command-line arguments for finetune_mtl.py
        script_path = self.chatterbox_scripts_dir / "finetune_mtl.py"
        
        cmd = [
            sys.executable,
            str(script_path),
            f"--output_dir={self.config.output_dir}",
            f"--metadata_file={self.config.metadata_file}",
            f"--dataset_dir={self.config.dataset_dir}",
            f"--train_split_name={self.config.train_split_name}",
            f"--eval_split_size={self.config.eval_split_size}",
            f"--num_train_epochs={self.config.num_train_epochs}",
            f"--per_device_train_batch_size={self.config.per_device_train_batch_size}",
            f"--per_device_eval_batch_size={self.config.per_device_eval_batch_size}",
            f"--gradient_accumulation_steps={self.config.gradient_accumulation_steps}",
            f"--learning_rate={self.config.learning_rate}",
            f"--warmup_steps={self.config.warmup_steps}",
            f"--weight_decay={self.config.weight_decay}",
            f"--max_grad_norm={self.config.max_grad_norm}",
            f"--logging_steps={self.config.logging_steps}",
            f"--evaluation_strategy={self.config.evaluation_strategy}",
            f"--save_strategy={self.config.save_strategy}",
            f"--save_total_limit={self.config.save_total_limit}",
            f"--report_to={self.config.report_to}",
            f"--language_id={self.config.language_id}",
            f"--seed={self.config.seed}",
            "--do_train",
            "--do_eval",
            "--dataloader_pin_memory=False",
            "--eval_on_start=True",
            "--label_names=labels_speech",
            "--text_column_name=text_scribe",
            "--dataloader_num_workers=0",
        ]
        
        if self.config.local_model_dir:
            cmd.append(f"--local_model_dir={self.config.local_model_dir}")
            
        if self.config.early_stopping_patience:
            cmd.append(f"--early_stopping_patience={self.config.early_stopping_patience}")
        
        # Run training
        print(f"Starting TTS training with command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Training failed: {result.stderr}")
            
        # Parse results
        output_dir = self.config.output_dir
        
        # Try to read training metrics if available
        metrics_file = Path(output_dir) / "train_results.json"
        metrics = {}
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
        
        return {
            "status": "completed",
            "output_dir": output_dir,
            "user": user,
            "metrics": metrics,
            "message": "TTS training completed successfully"
        }
    
    def run_hyperparameter_optimization(self, n_trials: int = 10) -> Dict[str, Any]:
        """
        Run Optuna hyperparameter optimization for TTS training.
        
        Args:
            n_trials: Number of trials to run
            
        Returns:
            Dict with best parameters and metrics
        """
        script_path = self.chatterbox_scripts_dir / "hyperparam_opt.py"
        
        cmd = [
            sys.executable,
            str(script_path),
            f"--metadata_file={self.config.metadata_file}",
            f"--dataset_dir={self.config.dataset_dir}",
            f"--output_dir={self.config.output_dir}_optuna",
            f"--n_trials={n_trials}",
            f"--language_id={self.config.language_id}",
            f"--eval_split_size={self.config.eval_split_size}",
        ]
        
        if self.config.local_model_dir:
            cmd.append(f"--local_model_dir={self.config.local_model_dir}")
        
        print(f"Starting hyperparameter optimization: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Optimization failed: {result.stderr}")
        
        # Parse best params from output directory
        best_params_file = Path(self.config.output_dir + "_optuna") / "best_params.json"
        if best_params_file.exists():
            with open(best_params_file, 'r') as f:
                best_params = json.load(f)
        else:
            best_params = {}
        
        return {
            "status": "completed",
            "best_params": best_params,
            "n_trials": n_trials
        }
    
    def evaluate(self, model_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate a trained TTS model.
        
        Args:
            model_dir: Directory containing trained model, defaults to output_dir
            
        Returns:
            Dict with evaluation metrics
        """
        model_dir = model_dir or self.config.output_dir
        
        # For now, return placeholder - can implement full evaluation later
        metrics_file = Path(model_dir) / "eval_results.json"
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                return json.load(f)
        
        return {
            "status": "completed",
            "model_dir": model_dir,
            "message": "Evaluation metrics not found"
        }


def main():
    """CLI entry point for testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="TTS Training Service CLI")
    parser.add_argument("--action", choices=["train", "optimize", "evaluate"], required=True)
    parser.add_argument("--metadata_file", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./models/checkpoints/tts_finetuned")
    parser.add_argument("--user", type=str, required=True, help="Profile username or folder name")
    parser.add_argument("--n_trials", type=int, default=10)
    
    args = parser.parse_args()
    
    config = TTSTrainingConfig(
        metadata_file=args.metadata_file,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir
    )
    
    service = TTSTrainingService(config)
    
    if args.action == "train":
        result = service.train(user=args.user)
        print(json.dumps(result, indent=2))
    elif args.action == "optimize":
        result = service.run_hyperparameter_optimization(n_trials=args.n_trials)
        print(json.dumps(result, indent=2))
    elif args.action == "evaluate":
        result = service.evaluate()
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
