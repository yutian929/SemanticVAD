import pytorch_lightning as pl
import torch, torchaudio
import numpy as np
from torch.utils.data import DataLoader, DistributedSampler
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn.utils.rnn import pad_sequence


class ASR_Testset(torch.utils.data.Dataset):
    def __init__(self, config):
        super().__init__()
        self.config = config

        ds = load_dataset(
            config.dataset_config.test_data_path
        )  # load_from huggingface datasets
        self.data_list = ds["test"]

        self._resample_buffer: dict[int, torchaudio.transforms.Resample] = {}

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data_dict = self.data_list[index]

        wav = torch.from_numpy(
            np.frombuffer(data_dict["wav"], dtype=np.float32).copy().reshape(1, -1)
        )
        gt_text = data_dict["text"]
        sr = data_dict["sample_rate"]

        if sr != 16000:
            if sr not in self._resample_buffer:
                self._resample_buffer[sr] = torchaudio.transforms.Resample(
                    orig_freq=sr, new_freq=16000
                )
            wav = self._resample_buffer[sr](wav).squeeze()
            sr = 16000
        else:
            wav = wav.squeeze()

        return {"wav": wav, "gt_text": gt_text, "sr": sr}

    def collator(self, samples):
        return samples


class Nonstreaming_ASR_Dataset(torch.utils.data.Dataset):
    def __init__(self, config, split="train"):
        super().__init__()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_config.model_name)
        self.IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss
        self.pad_token_id = self.tokenizer.pad_token_id
        self.added_audio_token_start = self.config.model_config.added_audio_token_start

        ds = load_dataset(
            config.dataset_config.train_data_path
        )  # load_from huggingface datasets
        train_val_split = ds["train"].train_test_split(
            test_size=self.config.dataset_config.split_size,
            seed=self.config.train_config.seed,
        )
        if split == "train":
            self.data_list = train_val_split["train"]
        else:
            self.data_list = train_val_split["test"]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data_dict = self.data_list[index]

        gt_text = data_dict["text"]
        audio_token = data_dict["audio_token"]

        audio_token = torch.tensor(audio_token) + self.added_audio_token_start

        # if data_dict["sequence"].startswith("<|task_asr|><|punctuation_on|>"):
        #     prompt = "<|task_asr|><|punctuation_on|>"
        # else:
        #     prompt = "<|task_asr|><|punctuation_off|>"

        prompt = "<|task_asr|><|punctuation_off|>"

        prompt_token = torch.tensor(self.tokenizer.encode(prompt))
        audio_token = torch.cat((prompt_token, audio_token), dim=0)

        text = "<|im_start|>" + gt_text + "<|im_end|>"

        text_token = torch.tensor(self.tokenizer.encode(text))

        full_token = torch.cat((audio_token, text_token), dim=0)

        if full_token.shape[0] > self.config.dataset_config.max_token_length:
            full_token = full_token[: self.config.dataset_config.max_token_length]

        audio_mask = full_token >= self.added_audio_token_start

        label_token = full_token.clone()
        label_token[: len(audio_token) + 1] = self.IGNORE_INDEX

        return {
            "input_id": full_token,
            "label": label_token,
            "gt_text": gt_text,
            "audio_mask": audio_mask,
        }

    def collator(self, samples):
        assert samples is not None

        input_ids = pad_sequence(
            [s["input_id"] for s in samples], padding_value=self.pad_token_id
        ).transpose(0, 1)

        label_ids = pad_sequence(
            [s["label"] for s in samples], padding_value=self.IGNORE_INDEX
        ).transpose(0, 1)

        audio_masks = pad_sequence(
            [s["audio_mask"] for s in samples], padding_value=False
        ).transpose(0, 1)

        return {
            "input_ids": input_ids,
            "label_ids": label_ids,
            "gt_texts": [s["gt_text"] for s in samples],
            "audio_masks": audio_masks,
        }


class Nonstreaming_ASR_DataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.asr_train = Nonstreaming_ASR_Dataset(self.config, "train")
            self.asr_val = Nonstreaming_ASR_Dataset(self.config, "val")

        # Assign test dataset for use in dataloaders
        if stage == "test" or stage is None:
            self.asr_test = ASR_Testset(self.config)

    def train_dataloader(self):
        generator = torch.Generator()
        generator.manual_seed(self.config.train_config.seed)

        return DataLoader(
            self.asr_train,
            batch_size=self.config.dataset_config.batch_size,
            num_workers=self.config.dataset_config.num_workers,
            collate_fn=self.asr_train.collator,
            generator=generator,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        generator = torch.Generator()
        generator.manual_seed(self.config.train_config.seed)

        return DataLoader(
            self.asr_val,
            batch_size=self.config.dataset_config.batch_size,
            num_workers=self.config.dataset_config.num_workers,
            collate_fn=self.asr_val.collator,
            generator=generator,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.asr_test,
            batch_size=1,
            num_workers=self.config.dataset_config.num_workers,
            collate_fn=self.asr_test.collator,
            shuffle=False,
        )
