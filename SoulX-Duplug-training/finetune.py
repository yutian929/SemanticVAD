import os, sys
import random
import wandb
import numpy as np
import torch
import pytorch_lightning as pl
import argparse
import hydra
from omegaconf import DictConfig, ListConfig, OmegaConf

from config.config import RunConfig
from models.nonstreaming_asr_data import Nonstreaming_ASR_DataModule
from models.nonstreaming_asr_model import Nonstreaming_ASR_Model
from models.streaming_asr_data import Streaming_ASR_DataModule
from models.streaming_asr_model import Streaming_ASR_Model
from models.state_prediction_data import State_Prediction_DataModule
from models.state_prediction_model import State_Prediction_Model

from pytorch_lightning.loggers import WandbLogger, CSVLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from utils.ema.ema import EMA, EMAModelCheckpoint
from utils.dynamic_train import DynamicTrainCallback
from utils.epoch_shuffle import SetEpochCallback


def debug(config):
    for arg_name, arg_value in OmegaConf.to_container(config, resolve=True).items():
        print(f"{arg_name}: {arg_value}")

    if config.model_config.task == "nonstreaming_asr":
        print("current task: non-streaming ASR")
        model = Nonstreaming_ASR_Model(config)
        data = Nonstreaming_ASR_DataModule(config)
    elif config.model_config.task == "streaming_asr":
        print("current task: streaming ASR")
        model = Streaming_ASR_Model(config)
        data = Streaming_ASR_DataModule(config)
    elif config.model_config.task == "state_prediction":
        print("current task: state prediction")
        model = State_Prediction_Model(config)
        data = State_Prediction_DataModule(config)
    else:
        raise ValueError(
            "task should be nonstreaming_asr, streaming_asr or state_prediction"
        )

    if config.train_config.wandb_run_name:
        logger = WandbLogger(
            project="duplex-asr",
            name=config.train_config.wandb_run_name,
            log_model=False,
            save_dir=config.train_config.wandb_save_dir,
        )
    else:
        logger = CSVLogger(save_dir=config.train_config.debug_log_dir)

    # callbacks
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [lr_monitor]

    # shuffle_callback = SetEpochCallback(config.train_config)
    # callbacks.append(shuffle_callback)

    # if config.train_config.adapter_first:
    #     print(
    #         f"only train adapter first, will unfreeze LM at step {config.train_config.unfreeze_step}"
    #     )
    #     dynamic_train_callback = DynamicTrainCallback(config.train_config.unfreeze_step)
    #     callbacks.append(dynamic_train_callback)

    if config.train_config.total_epochs > 0:
        trainer = pl.Trainer(
            max_epochs=config.train_config.total_epochs,
            accelerator="gpu",
            precision="16-mixed",
            # strategy="deepspeed_stage_2",
            devices=[0],
            log_every_n_steps=config.train_config.log_every_n_steps,
            val_check_interval=config.train_config.val_check_interval,
            accumulate_grad_batches=config.train_config.accumulate_grad_batches,
            enable_checkpointing=False,
            reload_dataloaders_every_n_epochs=1,
            logger=logger,
            callbacks=callbacks,
            sync_batchnorm=True,
        )
    else:
        trainer = pl.Trainer(
            max_steps=config.train_config.total_steps,
            accelerator="gpu",
            precision="16-mixed",
            # strategy="deepspeed_stage_2",
            devices=[0],
            log_every_n_steps=config.train_config.log_every_n_steps,
            val_check_interval=config.train_config.val_check_interval,
            accumulate_grad_batches=config.train_config.accumulate_grad_batches,
            enable_checkpointing=False,
            reload_dataloaders_every_n_epochs=1,
            logger=logger,
            callbacks=callbacks,
            sync_batchnorm=True,
        )

    trainer.fit(model=model, datamodule=data)


def train(config):
    assert config.train_config.wandb_run_name != ""
    assert config.train_config.wandb_save_dir != ""

    for arg_name, arg_value in OmegaConf.to_container(config, resolve=True).items():
        print(f"{arg_name}: {arg_value}")

    # prepare model and data
    if config.model_config.task == "nonstreaming_asr":
        print("current task: non-streaming ASR")
        model = Nonstreaming_ASR_Model(config)
        data = Nonstreaming_ASR_DataModule(config)
    elif config.model_config.task == "streaming_asr":
        print("current task: streaming ASR")
        model = Streaming_ASR_Model(config)
        data = Streaming_ASR_DataModule(config)
    elif config.model_config.task == "state_prediction":
        print("current task: state prediction")
        model = State_Prediction_Model(config)
        data = State_Prediction_DataModule(config)
    else:
        raise ValueError(
            "task should be nonstreaming_asr, streaming_asr or state_prediction"
        )

    # set wandb
    wandb_logger = WandbLogger(
        project="duplex-asr",
        name=config.train_config.wandb_run_name,
        log_model=False,
        save_dir=config.train_config.wandb_save_dir,
    )

    # callbacks
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [lr_monitor]

    # shuffle_callback = SetEpochCallback(config.train_config)
    # callbacks.append(shuffle_callback)

    # if config.train_config.adapter_first:
    #     print(
    #         f"only train adapter first, will unfreeze LM at step {config.train_config.unfreeze_step}"
    #     )
    #     config.train_config.strategy = "ddp_find_unused_parameters_true"
    #     dynamic_train_callback = DynamicTrainCallback(config.train_config.unfreeze_step)
    #     callbacks.append(dynamic_train_callback)

    if config.train_config.enable_ema:
        print("using EMA for better performance")
        checkpoint_callback = EMAModelCheckpoint(
            save_top_k=2,
            monitor="val_acc",
            mode="max",
            dirpath=config.train_config.default_root_dir,
            filename="{epoch:02d}_{step:06d}_model-{val_acc:.5f}",
            save_last=True,
            # save_on_train_epoch_end=True,
        )
        ema_callback = EMA(
            decay=config.train_config.ema_dacay,
            apply_ema_every_n_steps=config.train_config.ema_every_n_steps,
            start_step=config.train_config.ema_start_step,
            save_ema_weights_in_callback_state=True,
            evaluate_ema_weights_instead=True,
        )

        callbacks.append(checkpoint_callback)
        callbacks.append(ema_callback)
    else:
        checkpoint_callback = ModelCheckpoint(
            save_top_k=2,
            monitor="val_acc",
            mode="max",
            dirpath=config.train_config.default_root_dir,
            filename="{epoch:02d}_{step:06d}_model-{val_acc:.5f}",
            save_last=True,
            # save_on_train_epoch_end=True,
        )
        callbacks.append(checkpoint_callback)

    if config.train_config.total_epochs > 0:
        trainer = pl.Trainer(
            max_epochs=config.train_config.total_epochs,
            accelerator=config.train_config.accelerator,
            precision=config.train_config.precision,
            strategy=config.train_config.strategy,
            accumulate_grad_batches=config.train_config.accumulate_grad_batches,
            devices=config.train_config.num_gpu_per_node,
            num_nodes=config.train_config.num_node,
            log_every_n_steps=config.train_config.log_every_n_steps,
            val_check_interval=config.train_config.val_check_interval,
            default_root_dir=config.train_config.default_root_dir,
            reload_dataloaders_every_n_epochs=1,
            logger=wandb_logger,
            callbacks=callbacks,
            sync_batchnorm=config.train_config.sync_batchnorm,
        )
    else:
        trainer = pl.Trainer(
            max_steps=config.train_config.total_steps,
            accelerator=config.train_config.accelerator,
            precision=config.train_config.precision,
            strategy=config.train_config.strategy,
            accumulate_grad_batches=config.train_config.accumulate_grad_batches,
            devices=config.train_config.num_gpu_per_node,
            num_nodes=config.train_config.num_node,
            log_every_n_steps=config.train_config.log_every_n_steps,
            val_check_interval=config.train_config.val_check_interval,
            default_root_dir=config.train_config.default_root_dir,
            reload_dataloaders_every_n_epochs=1,
            logger=wandb_logger,
            callbacks=callbacks,
            sync_batchnorm=config.train_config.sync_batchnorm,
        )

    save_config_path = os.path.join(config.train_config.default_root_dir, "config.yaml")
    os.makedirs(os.path.dirname(save_config_path), exist_ok=True)
    with open(save_config_path, "w") as f:
        OmegaConf.save(config, f)

    if config.train_config.ckpt_path:
        # continue training from checkpoint
        trainer.fit(
            model=model, datamodule=data, ckpt_path=config.train_config.ckpt_path
        )
    else:
        trainer.fit(model=model, datamodule=data)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune model")
    parser.add_argument(
        "--config_path", type=str, required=True, help="Path to the config file"
    )
    args = parser.parse_args()

    default_config = RunConfig()
    run_config = OmegaConf.load(args.config_path)
    config = OmegaConf.merge(default_config, run_config)

    torch.backends.cudnn.enabled = False
    torch.cuda.empty_cache()

    # set seed
    pl.seed_everything(config.train_config.seed, workers=True)
    torch.cuda.manual_seed(config.train_config.seed)
    torch.manual_seed(config.train_config.seed)
    random.seed(config.train_config.seed)
    np.random.seed(config.train_config.seed)

    if config.train_config.stage == "train":
        train(config)
    elif config.train_config.stage == "debug":
        debug(config)
    else:
        raise ValueError("stage should be 'train' or 'debug'")


if __name__ == "__main__":
    main()
