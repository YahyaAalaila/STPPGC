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

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if step > 0 and step % self.log_every_n_steps == 0:
            # grab all metrics that have been logged so far
            metrics = {
                k: v.item() if isinstance(v, torch.Tensor) else v
                for k, v in trainer.callback_metrics.items()
            }
            # add the LR
            opt = trainer.optimizers[0]
            metrics["lr"] = opt.param_groups[0]["lr"]

            # log to each logger you’ve configured
            if hasattr(trainer, "loggers") and trainer.loggers:
                for logger in trainer.loggers:
                    logger.log_metrics(metrics, step=step)
            else:
                # fallback for single‐logger API
                trainer.logger.log_metrics(metrics, step=step)