<div align="center">
    <h1>
    SoulX-Duplug
    </h1>
    <p>
    Official code for enabling full-duplex speech interaction with<br>
    <b><em>SoulX-Duplug: Plug-and-Play Streaming State Prediction Module for Realtime Full-Duplex Speech Conversation</em></b>
    </p>
    <p>
    <img src="assets/SoulX-Duplug-logo.png" alt="SoulX-Duplug Logo" style="width: 200px; height: 200px;">
    </p>
    <p>
    </p>
    <!-- <a href="https://github.com/Soul-AILab/SoulX-Duplug"><img src="https://img.shields.io/badge/Platform-linux-lightgrey" alt="version"></a>
    <a href="https://github.com/Soul-AILab/SoulX-Duplug"><img src="https://img.shields.io/badge/Python-3.10-blue" alt="version"></a> -->
    <a href="https://soulx-duplug.sjtuxlance.com/"><img src="https://img.shields.io/badge/🌐%20Online-Demo-blue" alt="Online Demo"></a>
    <a href="https://arxiv.org/abs/2603.14877"><img src="https://img.shields.io/badge/arXiv-2603.14877-B31B1B?logo=arxiv&logoColor=white.svg" alt="arXiv"></a>
    <a href="https://huggingface.co/Soul-AILab/SoulX-Duplug-0.6B"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow" alt="HF-Model"></a>
    <a href="https://huggingface.co/datasets/Soul-AILab/SoulX-Duplug-Eval"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Eval-yellow" alt="HF-Eval"></a>
    <a href="https://github.com/Soul-AILab/SoulX-Duplug"><img src="https://img.shields.io/badge/License-Apache%202.0-green.svg" alt="Apache-2.0"></a>
</div>


# Training Scripts



## 🚀 News
- **[2026-07-17]** We have released the training code for SoulX-Duplug. This is a re-implemented version of our training pipeline, and you can use it to train the model from scratch or fine-tune it on your own data.
- **[2026-07-17]** Demo Notice: The online demo is temporarily unavailable. Please follow the deployment instructions in this repository if you would like to try SoulX-Duplug locally.
- **[2026-03-17]** Our paper on this project has been published! You can read it here: [SoulX-Duplug](https://arxiv.org/abs/2603.14877).
- **[2026-03-16]** SoulX-Duplug checkpoint and SoulX-Duplug-Eval are now available on Hugging Face! You can access it directly from [SoulX-Duplug-HF](https://huggingface.co/collections/Soul-AILab/soulx-duplug).


## 🛠️ Install

### Clone and Install
Here are instructions for installing on Linux.

- Clone the repo
```bash
git clone https://github.com/Soul-AILab/SoulX-Duplug.git
cd SoulX-Duplug
```

- Install system dependencies
```bash
sudo apt-get update
sudo apt-get install ffmpeg sox libsox-dev -y
```

- Install Conda: please see https://docs.conda.io/en/latest/miniconda.html

- Create Conda env
```bash
conda create -n soulx-duplug -y python=3.10
conda activate soulx-duplug
pip install -r requirements.txt
# If you are in mainland China, you can set the mirror as follows:
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
```
> `requirements.txt` is a pinned snapshot of a fully working environment. If the resolver reports version conflicts on some peripheral packages, install with `--no-deps` to reproduce the environment exactly:
> ```bash
> pip install -r requirements.txt --no-deps
> ```



### Model Download

Download via hf:
```bash
# If you are in mainland China, please first set the mirror:
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --resume-download Soul-AILab/SoulX-Duplug-0.6B --local-dir pretrained_models
```

Download via python:
```python
from huggingface_hub import snapshot_download
snapshot_download("Soul-AILab/SoulX-Duplug-0.6B", local_dir="pretrained_models") 
```

Download via git clone:
```bash
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install
git clone https://huggingface.co/Soul-AILab/SoulX-Duplug-0.6B pretrained_models
```

The released checkpoint also serves as a good starting point for fine-tuning: point `init_ckpt_path_lora` (or `init_ckpt_path`) in your config to the downloaded weights and continue training on your own data (see [Training](https://github.com/Soul-AILab/SoulX-Duplug/tree/training-code#%EF%B8%8F-training) below).



## 🏋️ Training

We release the re-implemented training code of SoulX-Duplug, covering all three tasks: **state prediction** (the core streaming turn-taking module), **streaming ASR** and **non-streaming ASR**.

### Prepare the Base Model

SoulX-Duplug is built on a Qwen3 backbone whose vocabulary is expanded with discrete audio tokens and streaming state tokens. Use `scripts/expand_vocab.py` to add these tokens to a base Qwen3 model and initialize the new embeddings:

```bash
python scripts/expand_vocab.py \
    --model_path pretrained_models/Qwen3-0.6B \
    --tokenizer_path pretrained_models/Qwen3-0.6B-expand_vocab_v2 \
    --output_path pretrained_models/Qwen3-0.6B-expand_vocab_v2 \
    --safe_serialization
```

This adds `51866` audio tokens (`<|audio_0|>` ... `<|audio_51865|>`) and a set of special task/state tokens (e.g. `<|task_duplex_predict|>`, `<|user_idle|>`, `<|user_complete|>`, `<|end_of_sentence|>`), and writes a `config_record.yaml` with the resulting vocabulary layout. The released `pretrained_models/Qwen3-0.6B-expand_vocab_v2` is already prepared this way, so you only need this step if you start from a fresh Qwen3 model.

### Data Format

Training data is loaded through 🤗 `datasets` via `load_dataset(train_data_path)`, where `train_data_path` is a **directory** containing your dataset (a single `.jsonl` file path is not accepted). We recommend converting your data to **Parquet** before training for faster loading:

```python
from datasets import load_dataset

ds = load_dataset("json", data_files="example_data_fisher.jsonl")
ds["train"].to_parquet("data/fisher/train.parquet")   # train_data_path -> data/fisher
```

`example_data_fisher.jsonl` is a small state-prediction example illustrating the data format.

### Configs

All configs live in `config/` and are merged on top of the dataclass defaults in `config/config.py`. Set `dataset_config.train_data_path` to your data directory before training.

- **`config/train_config.yaml`** — the main training config. Selects the task via `model_config.task` (`state_prediction` / `streaming_asr` / `nonstreaming_asr`), and controls the backbone (`model_name`, `llm_dim`), LoRA, the per-state loss weights, and the trainer setup (steps, precision, strategy, wandb).
- **`config/debug_config.yaml`** — a single-GPU config (`stage: debug`) for quick local sanity checks; logs to a local CSV logger instead of wandb.
- **`config/infer_config.yaml`** — config used by the offline inference / export scripts.

> The released checkpoint is the **0.6B** model. If you train a larger backbone, update `model_name` and `llm_dim` accordingly.

### Loss Weighting for State Prediction

State prediction is trained with several token-level objectives (text, EOS, and the user states: idle / non-idle / complete / incomplete / backchannel), combined as a weighted sum. The weights are set by the `*_loss_rate` fields in `train_config`. When `enable_switch_loss_rate` is on, samples whose `index` starts with `switch_loss_rate_label` (e.g. `fe`) use the `*_loss_rate_switched` variants instead, which lets you weight different data sources differently.

### Launch Training

Edit `launch.sh` to set your visible GPUs and (optionally) your `WANDB_API_KEY`, then run:

```bash
bash launch.sh
```

`launch.sh` runs the main entry point with `torchrun` for multi-GPU distributed training:

```bash
torchrun finetune.py --config_path config/train_config.yaml
```

For a quick single-GPU sanity check without wandb, use the debug config:

```bash
python finetune.py --config_path config/debug_config.yaml
```



## 📌 TODOs
- [x] Publish the technical report.
- [x] Release evaluation scripts.
- [x] Release training scripts.


## 🔖 Citation
If you find this work useful in your research, please consider citing:

```bibtex
@misc{yan2026soulxduplug,
      title={SoulX-Duplug: Plug-and-Play Streaming State Prediction Module for Realtime Full-Duplex Speech Conversation}, 
      author={Ruiqi Yan and Wenxi Chen and Zhanxun Liu and Ziyang Ma and Haopeng Lin and Hanlin Wen and Hanke Xie and Jun Wu and Yuzhe Liang and Yuxiang Zhao and Pengchao Feng and Jiale Qian and Hao Meng and Yuhang Dai and Shunshun Yin and Ming Tao and Lei Xie and Kai Yu and Xinsheng Wang and Xie Chen},
      year={2026},
      eprint={2603.14877},
      archivePrefix={arXiv},
      primaryClass={eess.AS},
      url={https://arxiv.org/abs/2603.14877}, 
}
```

## 📜 License
This project is licensed under the [Apache 2.0 License](LICENSE).


## 🙏 Acknowledgment
We thank the following open-source projects for their open-source contributions:

- [QwenLM](https://github.com/QwenLM)
- [GLM-4-Voice](https://github.com/zai-org/GLM-4-Voice)
- [chinese_text_normalization](https://github.com/speechio/chinese_text_normalization)
- [Paraformer](https://github.com/modelscope/FunASR/wiki/paraformer)
- [Sensevoice](https://github.com/FunAudioLLM/SenseVoice)
- [ChatTTS](https://github.com/2noise/ChatTTS)
- [SLAM-LLM](https://github.com/X-LANCE/SLAM-LLM)
