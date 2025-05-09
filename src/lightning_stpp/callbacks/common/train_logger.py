
from lightning.pytorch.callbacks import Callback
import torch

class TrainLoggerCallback(Callback):
    """
    Logs training metrics every `log_every_n_steps` steps.
    Expects that the LightningModule logs metrics via `self.log(...)`.
    """
    def __init__(self, log_every_n_steps: int = 100):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        step = trainer.global_step
        if step > 0 and step % self.log_every_n_steps == 0:
            # Collect the metrics logged during this batch
            metrics = {
                key: val.item()
                for key, val in trainer.logger_connector.cached_results["log"].items()
                if isinstance(val, torch.Tensor)
            }
            # Also log the current learning rate
            opt = trainer.optimizers[0]
            metrics["lr"] = opt.param_groups[0]["lr"]
            trainer.logger.log_metrics(metrics, step=step)