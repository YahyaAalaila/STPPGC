from lightning.pytorch.callbacks import Callback

class TestSchedulerCallback(Callback):
    """
    Runs full test evaluation every `every_n_epochs` epochs during training.
    """
    def __init__(self, every_n_epochs: int):
        super().__init__()
        self.every_n_epochs = every_n_epochs

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if (epoch + 1) % self.every_n_epochs == 0:
            trainer.test(pl_module, verbose=False)
