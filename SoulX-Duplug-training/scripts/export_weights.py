import os, sys
import torch
import random
import numpy as np
import argparse
from omegaconf import OmegaConf
import pytorch_lightning as pl

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.config import RunConfig
from models.nonstreaming_asr_model import Nonstreaming_ASR_Model
from models.streaming_asr_model import Streaming_ASR_Model
from models.state_prediction_model import State_Prediction_Model
from transformers import WhisperFeatureExtractor


def parse_args():
    parser = argparse.ArgumentParser(description="Export Model Weights")

    parser.add_argument(
        "--config_path", type=str, required=True, help="Path to the config file"
    )

    parser.add_argument(
        "--output_path", type=str, required=True, help="Path to the output weights file"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    default_config = RunConfig()
    run_config = OmegaConf.load(args.config_path)
    config = OmegaConf.merge(default_config, run_config)

    pl.seed_everything(config.infer_config.seed)
    torch.manual_seed(config.infer_config.seed)
    np.random.seed(config.infer_config.seed)
    random.seed(config.infer_config.seed)

    if config.model_config.task == "nonstreaming_asr":
        print("current task: non-streaming ASR")
        model = Nonstreaming_ASR_Model(config)
    elif config.model_config.task == "streaming_asr":
        print("current task: streaming ASR")
        model = Streaming_ASR_Model(config)
    elif config.model_config.task == "state_prediction":
        print("current task: state prediction")
        model = State_Prediction_Model(config)

    model.feature_extractor = WhisperFeatureExtractor.from_pretrained(
        config.model_config.glm_tokenizer_path
    )

    torch.save(model.state_dict(), args.output_path)

    print("Model weights exported successfully to", args.output_path)


if __name__ == "__main__":
    main()

# python scripts/export_weights.py --config_path config/infer_config.yaml --output_path path/to/output_weights.pth
