import pytorch_lightning as pl
from pytorch_lightning import Callback


class DynamicTrainCallback(Callback):
    def __init__(self, unfreeze_step=10000):
        super().__init__()
        self.unfreeze_step = unfreeze_step
        self.unfrozen = False  # Flag to track if the language model has been unfrozen

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Freeze LM parameters at the start of training, only train the adapter."""
        pl_module.partial_freeze_weights(
            pl_module.config.original_vocab_size, pl_module.config.lm_vocab_size
        )
        print("LM parameters frozen, training only the adapter.")

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx,
    ):
        """Check the step count after each batch; unfreeze LM parameters if the threshold is reached."""
        if trainer.global_step >= self.unfreeze_step and not self.unfrozen:
            for name, param in pl_module.llm.named_parameters():
                param.requires_grad = True

            for handle in pl_module.hook_handles:
                handle.remove()

            pl_module.llm.train()

            self.unfrozen = True
            print(
                f"Step {trainer.global_step}: Unfreezing LM parameters, continuing full model training."
            )
