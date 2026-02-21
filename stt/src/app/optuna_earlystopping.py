import optuna
from transformers import TrainerCallback

class OptunaEarlyStoppingCallback(TrainerCallback):
    """Callback to report metrics to Optuna and handle pruning."""
    
    def __init__(
        self, 
        trial: optuna.Trial, 
        metric_name: str = "eval_wer",
        max_wer_threshold: float = 200.0,  # Prune if WER exceeds this
        patience: int = 2,  # Prune if WER increases for N consecutive epochs
    ):
        self.trial = trial
        self.metric_name = metric_name
        self.max_wer_threshold = max_wer_threshold
        self.patience = patience
        self.step = 0
        self.best_value = float("inf")
        self.worse_count = 0
    
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        """Called after each evaluation - report to Optuna and check for pruning."""
        metric_value = metrics.get(self.metric_name)
        
        if metric_value is not None:
            self.trial.report(metric_value, step=self.step)
            self.step += 1
            
            if self.trial.should_prune():
                raise optuna.TrialPruned(f"Trial pruned by Optuna at step {self.step}")
            
            if metric_value > self.max_wer_threshold:
                raise optuna.TrialPruned(
                    f"Trial pruned: WER {metric_value:.1f}% exceeds threshold {self.max_wer_threshold}%"
                )
            
            if metric_value < self.best_value:
                self.best_value = metric_value
                self.worse_count = 0
            else:
                self.worse_count += 1
                if self.worse_count >= self.patience:
                    raise optuna.TrialPruned(
                        f"Trial pruned: WER increased for {self.patience} consecutive epochs"
                    )