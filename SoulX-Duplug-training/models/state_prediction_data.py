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


class State_Prediction_Dataset(torch.utils.data.Dataset):
    def __init__(self, config, split="train"):
        super().__init__()
        self.config = config
        self.dataset_config = config.dataset_config
        self.model_config = config.model_config

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_config.model_name)
        self.IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss
        self.pad_token_id = self.tokenizer.pad_token_id
        self.added_audio_token_start = self.model_config.added_audio_token_start
        self.special_token_start = self.model_config.special_token_start
        self.asr_eos_token_id = self.model_config.asr_eos_token_id

        self.user_idle_token_id = self.model_config.user_idle_token_id
        self.user_nonidle_token_id = self.model_config.user_nonidle_token_id
        self.user_complete_token_id = self.model_config.user_complete_token_id
        self.user_incomplete_token_id = self.model_config.user_incomplete_token_id
        self.user_backchannel_token_id = self.model_config.user_backchannel_token_id
        self.assistant_backchannel_token_id = (
            self.model_config.assistant_backchannel_token_id
        )
        self.assistant_interrupt_token_id = (
            self.model_config.assistant_interrupt_token_id
        )

        ds = load_dataset(
            self.dataset_config.train_data_path
        )  # load_from huggingface datasets
        train_val_split = ds["train"].train_test_split(
            test_size=self.dataset_config.split_size, seed=self.config.train_config.seed
        )
        if split == "train":
            self.data_list = train_val_split["train"]
        else:
            self.data_list = train_val_split["test"]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data_dict = self.data_list[index]
        sequence = data_dict["sequence"]
        index = data_dict["index"]

        full_token = torch.tensor(self.tokenizer.encode(sequence))

        if full_token.shape[0] > self.dataset_config.max_token_length:
            full_token = full_token[: self.dataset_config.max_token_length]

        audio_mask = full_token >= self.added_audio_token_start
        special_token_mask = full_token >= self.special_token_start

        eos_mask = full_token == self.asr_eos_token_id
        user_idle_token_mask = full_token == self.user_idle_token_id
        user_nonidle_token_mask = full_token == self.user_nonidle_token_id
        user_complete_token_mask = full_token == self.user_complete_token_id
        user_incomplete_token_mask = full_token == self.user_incomplete_token_id
        user_backchannel_token_mask = full_token == self.user_backchannel_token_id

        label_token_text = full_token.clone()
        label_token_eos = full_token.clone()

        label_token_user_idle = full_token.clone()
        label_token_user_nonidle = full_token.clone()
        label_token_user_complete = full_token.clone()
        label_token_user_incomplete = full_token.clone()
        label_token_user_backchannel = full_token.clone()

        label_token_text[special_token_mask] = self.IGNORE_INDEX
        label_token_eos[~eos_mask] = self.IGNORE_INDEX

        label_token_user_idle[~user_idle_token_mask] = self.IGNORE_INDEX
        label_token_user_nonidle[~user_nonidle_token_mask] = self.IGNORE_INDEX
        label_token_user_complete[~user_complete_token_mask] = self.IGNORE_INDEX
        label_token_user_incomplete[~user_incomplete_token_mask] = self.IGNORE_INDEX
        label_token_user_backchannel[~user_backchannel_token_mask] = self.IGNORE_INDEX

        return {
            "index": index,
            "input_id": full_token,
            "audio_mask": audio_mask,
            # "label": label_token,
            "label_text": label_token_text,
            "label_eos": label_token_eos,
            "label_user_idle": label_token_user_idle,
            "label_user_nonidle": label_token_user_nonidle,
            "label_user_complete": label_token_user_complete,
            "label_user_incomplete": label_token_user_incomplete,
            "label_user_backchannel": label_token_user_backchannel,
        }

    def collator(self, samples):
        assert samples is not None

        switch_loss_rate = False
        if self.config.train_config.enable_switch_loss_rate:
            assert self.dataset_config.batch_size == 1
            if self.config.train_config.switch_loss_rate_label == "" or samples[0][
                "index"
            ].startswith(self.config.train_config.switch_loss_rate_label):
                switch_loss_rate = True

        input_ids = pad_sequence(
            [s["input_id"] for s in samples], padding_value=self.pad_token_id
        ).transpose(0, 1)

        audio_masks = pad_sequence(
            [s["audio_mask"] for s in samples], padding_value=False
        ).transpose(0, 1)

        label_text_ids = pad_sequence(
            [s["label_text"] for s in samples], padding_value=self.IGNORE_INDEX
        ).transpose(0, 1)

        label_eos_ids = pad_sequence(
            [s["label_eos"] for s in samples], padding_value=self.IGNORE_INDEX
        ).transpose(0, 1)

        label_user_idle_ids = pad_sequence(
            [s["label_user_idle"] for s in samples], padding_value=self.IGNORE_INDEX
        ).transpose(0, 1)

        label_user_nonidle_ids = pad_sequence(
            [s["label_user_nonidle"] for s in samples], padding_value=self.IGNORE_INDEX
        ).transpose(0, 1)

        label_user_complete_ids = pad_sequence(
            [s["label_user_complete"] for s in samples], padding_value=self.IGNORE_INDEX
        ).transpose(0, 1)

        label_user_incomplete_ids = pad_sequence(
            [s["label_user_incomplete"] for s in samples],
            padding_value=self.IGNORE_INDEX,
        ).transpose(0, 1)

        label_user_backchannel_ids = pad_sequence(
            [s["label_user_backchannel"] for s in samples],
            padding_value=self.IGNORE_INDEX,
        ).transpose(0, 1)

        # label_ids = pad_sequence(
        #     [s["label"] for s in samples], padding_value=self.IGNORE_INDEX
        # ).transpose(0, 1)

        return {
            "switch_loss_rate": switch_loss_rate,
            "input_ids": input_ids,
            "audio_masks": audio_masks,
            # "label_ids": label_ids,
            "label_text_ids": label_text_ids,
            "label_eos_ids": label_eos_ids,
            "label_user_idle_ids": label_user_idle_ids,
            "label_user_nonidle_ids": label_user_nonidle_ids,
            "label_user_complete_ids": label_user_complete_ids,
            "label_user_incomplete_ids": label_user_incomplete_ids,
            "label_user_backchannel_ids": label_user_backchannel_ids,
        }


class State_Prediction_DataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dataset_config = config.dataset_config
        self.epoch = 0

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.asr_train = State_Prediction_Dataset(self.config, "train")
            self.asr_val = State_Prediction_Dataset(self.config, "val")

        # dummy test set
        if stage == "test" or stage is None:
            self.asr_test = ASR_Testset(self.config)

    def train_dataloader(self):
        generator = torch.Generator()
        generator.manual_seed(self.config.train_config.seed + self.epoch)
        print(
            f"[train_dataloader] current seed: {self.config.train_config.seed + self.epoch}"
        )
        self.epoch += 1

        return DataLoader(
            self.asr_train,
            batch_size=self.dataset_config.batch_size,
            num_workers=self.dataset_config.num_workers,
            collate_fn=self.asr_train.collator,
            generator=generator,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.asr_val,
            batch_size=self.dataset_config.batch_size,
            num_workers=self.dataset_config.num_workers,
            collate_fn=self.asr_val.collator,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.asr_test,
            batch_size=1,
            num_workers=self.dataset_config.num_workers,
            collate_fn=self.asr_test.collator,
            shuffle=False,
        )
