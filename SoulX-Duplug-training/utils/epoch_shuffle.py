import pytorch_lightning as pl


class SetEpochCallback(pl.Callback):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def on_train_epoch_start(self, trainer, pl_module):
        epoch = trainer.current_epoch

        generator = getattr(trainer.train_dataloader, "generator", None)
        if generator is not None and hasattr(generator, "manual_seed"):
            generator.manual_seed(epoch + self.config.seed)
            pl_module.print(f"[SetEpochCallback] Set epoch={epoch} succeeded.")
        else:
            pl_module.print(f"[SetEpochCallback] Set epoch={epoch} failed.")

        # try:
        #     datamodule = trainer.datamodule
        #     trainer.train_dataloader = datamodule.train_dataloader_manual(
        #         seed=2 * epoch + self.config.seed
        #     )
        #     pl_module.print(f"[SetEpochCallback] Set epoch={epoch} succeeded.")
        # except Exception as e:
        #     pl_module.print(f"[SetEpochCallback] Set epoch={epoch} failed: {e}")
