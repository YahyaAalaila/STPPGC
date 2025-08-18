from lightning.pytorch.callbacks import Callback
class ValidationSchedulerCallback(Callback):
    """
    Runs validation every `every_n_epochs` epochs during training.
    """
    def __init__(self, every_n_epochs: int = 1):
        super().__init__()
        self.every_n_epochs = every_n_epochs

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if (epoch + 1) % self.every_n_epochs == 0:
            # `verbose=False` silences the extra printouts
            #trainer.validate(verbose=False)
            trainer.validate(ckpt_path=None, verbose=False)
