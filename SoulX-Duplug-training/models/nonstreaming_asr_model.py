import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
import pytorch_lightning as pl

from models.glm_4_voice.speech_tokenizer.modeling_whisper import WhisperVQEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import WhisperFeatureExtractor
from utils.sparkvox.utils.scheduler import WarmupAnnealSteps
from models._train_heads import TokenHeadsMixin
import peft
from peft import LoraConfig, get_peft_model


# # from SLAM-LLM
# class EncoderProjector(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.audio_embed_dim = config.audio_embed_dim
#         self.llm_dim = config.llm_dim
#         self.linear1 = nn.Linear(self.audio_embed_dim, 2048)
#         self.relu = nn.ReLU()
#         self.linear2 = nn.Linear(2048, self.llm_dim)

#     def forward(self, x):
#         x = x.contiguous()  # (batch, seq_len, dim)
#         x = self.linear1(x)
#         x = self.relu(x)
#         x = self.linear2(x)
#         return x


class EncoderProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.audio_embed_dim = config.audio_embed_dim
        self.llm_dim = config.llm_dim
        self.linear1 = nn.Linear(self.audio_embed_dim, 2048)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(2048, 2048)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(2048, self.llm_dim)

    def forward(self, x):
        x = x.contiguous()  # (batch, seq_len, dim)
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        return x


class Nonstreaming_ASR_Model(TokenHeadsMixin, pl.LightningModule):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.model_config = config.model_config
        self.train_config = config.train_config
        self.lm_vocab_size = config.model_config.lm_vocab_size
        self.best_val_acc = 0.0
        self.save_hyperparameters(self.config)

        self.glm_tokenizer = WhisperVQEncoder.from_pretrained(
            config.model_config.glm_tokenizer_path
        )
        for name, param in self.glm_tokenizer.named_parameters():
            param.requires_grad = False
        self.glm_tokenizer.eval()

        if self.model_config.enable_projector:
            if self.global_rank == 0:
                print(f"setting up audio projector...")
            self.audio_projector = EncoderProjector(self.model_config)
        else:
            self.audio_projector = None

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_config.model_name)
        self.llm = AutoModelForCausalLM.from_pretrained(config.model_config.model_name)

        for name, param in self.llm.named_parameters():
            param.requires_grad = True
        self.llm.train()

        if self.model_config.embed_only:
            if self.global_rank == 0:
                print(f"only train partial embedding weights...")
            self.partial_freeze_weights(
                self.model_config.original_vocab_size, self.model_config.lm_vocab_size
            )

        if self.model_config.enable_lora:
            if self.global_rank == 0:
                print(f"setting up lora model...")
            peft_config = LoraConfig(
                task_type=self.model_config.lora_task_type,
                r=self.model_config.lora_r,
                lora_alpha=self.model_config.lora_alpha,
                lora_dropout=self.model_config.lora_dropout,
            )
            self.llm = get_peft_model(self.llm, peft_config)

        if hasattr(self.llm.model, "embed_tokens"):
            self.embed_tokens_func = self.llm.model.embed_tokens
        elif hasattr(self.llm.model.model, "embed_tokens"):
            self.embed_tokens_func = self.llm.model.model.embed_tokens
        else:
            self.embed_tokens_func = self.llm.model.model.model.embed_tokens

    def forward(self, batch):
        sequences, audio_masks, labels = batch

        if self.audio_projector:
            audio_tokens = sequences.clone()
            audio_tokens[audio_masks] -= self.model_config.added_audio_token_start
            audio_tokens[~audio_masks] = 0
            audio_embeds = self.glm_tokenizer.codebook(audio_tokens)
            audio_embeds = self.audio_projector(audio_embeds)

            sequences[audio_masks] = 0

            text_embeds = self.embed_tokens_func(sequences)

            audio_masks = audio_masks.unsqueeze(-1)
            inputs_embeds = audio_embeds * audio_masks + text_embeds * (~audio_masks)

            model_outputs = self.llm(inputs_embeds=inputs_embeds, labels=labels)
        else:
            model_outputs = self.llm(input_ids=sequences, labels=labels)

        return model_outputs

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        label_ids = batch["label_ids"]
        audio_masks = batch["audio_masks"]

        model_outputs = self((input_ids, audio_masks, label_ids))

        x_ori = model_outputs.logits

        loss = self._shifted_ce(x_ori, label_ids)

        # if torch.isnan(loss):
        #     print(f"[Warning]: NaN loss detected, Skip back propagation.")
        #     return None

        preds = torch.argmax(x_ori, -1)
        acc = self._shifted_acc(preds, label_ids)

        self.log("train_loss", loss, prog_bar=True, logger=True, rank_zero_only=True)

        self.log("train_acc", acc, prog_bar=True, logger=True, rank_zero_only=True)

        return loss

    def on_validation_epoch_start(self):
        self.val_loss = []
        self.val_acc = []

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        label_ids = batch["label_ids"]
        audio_masks = batch["audio_masks"]

        model_outputs = self((input_ids, audio_masks, label_ids))

        x_ori = model_outputs.logits

        loss = self._shifted_ce(x_ori, label_ids)

        preds = torch.argmax(x_ori, -1)
        acc = self._shifted_acc(preds, label_ids)

        self.val_loss.append(loss)
        self.val_acc.append(acc)

    def on_validation_epoch_end(self):
        avg_loss = torch.nanmean(torch.stack(self.val_loss))
        avg_acc = torch.nanmean(torch.stack(self.val_acc))

        self.log(f"val_loss", avg_loss, prog_bar=True, logger=True, sync_dist=True)

        self.log(f"val_acc", avg_acc, prog_bar=True, logger=True, sync_dist=True)

        self.best_val_acc = max(self.best_val_acc, avg_acc)
        self.log(
            f"best_val_acc",
            self.best_val_acc,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

    def test_step(self, batch, batch_idx):
        asr_text, asr_end, gt_text, duration = self.generate_from_audio(batch)
        print(f"ASR result: {asr_text}")

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.train_config.learning_rate,
            weight_decay=self.train_config.weight_decay,
            betas=self.train_config.betas,
            eps=self.train_config.eps,
        )

        if self.train_config.linear_lr:
            scheduler_dict = {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(
                    optimizer,
                    lr_lambda=lambda step: (
                        min(step / self.train_config.warmup_steps, 1)
                        if step < self.train_config.warmup_steps
                        else 1
                    ),
                ),
                "interval": "step",
                "frequency": 1,
            }
        else:
            scheduler_dict = {
                "scheduler": WarmupAnnealSteps(
                    optimizer,
                    warmup_step=self.train_config.warmup_steps,
                    anneal_steps=[self.train_config.anneal_steps],
                    anneal_rate=self.train_config.anneal_rate,
                    final_lr=self.train_config.min_lr,
                ),
                "interval": "step",
                "frequency": 1,
            }

        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}

    def partial_freeze_weights(self, original_vocabsize, total_vocabsize):
        self.hook_handles = []

        if self.global_rank == 0:
            print(
                f"Only training partial embedding layer, from {original_vocabsize} to {total_vocabsize}"
            )

        trainable_range = (original_vocabsize, total_vocabsize)

        # Define a hook to zero out the gradient for weights outside the trainable range during the backward pass
        def zero_out_gradient(grad):
            grad[: trainable_range[0], :] = 0
            grad[trainable_range[1] + 1 :, :] = 0
            return grad

        # Freeze all layers first
        for param in self.llm.parameters():
            param.requires_grad = False

        # Assuming the output layer is `lm_head`
        for param in self.llm.lm_head.parameters():
            # Compute the standard deviation for He initialization
            std_dev = (2.0 / param.size(1)) ** 0.5

            # Initialize the specific rows with He initialization
            param[original_vocabsize:total_vocabsize] = (
                torch.randn((trainable_range[1] - trainable_range[0], param.size(1)))
                * std_dev
            )
            param.requires_grad = True
            # Register the hook on the weight tensor
            handle = param.register_hook(zero_out_gradient)
            self.hook_handles.append(handle)

        if hasattr(self.llm.model, "model") and hasattr(
            self.llm.model.model, "embed_tokens"
        ):
            embed_tokens_module = self.llm.model.model.embed_tokens
        elif hasattr(self.llm.model, "embed_tokens"):
            embed_tokens_module = self.llm.model.embed_tokens
        else:
            raise AttributeError("Cannot find embed_tokens in self.llm.model")

        # For non-tied embedding layers, both the two embedding layers need to be hooked
        if self.llm.lm_head.weight.data_ptr() != embed_tokens_module.weight.data_ptr():
            for param in embed_tokens_module.parameters():
                std_dev = (2.0 / param.size(1)) ** 0.5
                param[original_vocabsize:total_vocabsize] = (
                    torch.randn(
                        (trainable_range[1] - trainable_range[0], param.size(1))
                    )
                    * std_dev
                )
                param.requires_grad = True
                handle = param.register_hook(zero_out_gradient)
                self.hook_handles.append(handle)

    @torch.no_grad()
    def generate_from_audio(self, batch, punctuation):
        assert len(batch) == 1

        wav = batch[0]["wav"]
        gt_text = batch[0]["gt_text"]
        sr = batch[0]["sr"]

        device = "cuda"

        wav = wav.cpu().numpy()
        time_step = 0
        audios = []
        while time_step * 16000 < wav.shape[0]:
            audio_segment = wav[time_step * 16000 : (time_step + 30) * 16000]
            audios.append(audio_segment)
            time_step += 30

        pooling_kernel_size = self.glm_tokenizer.config.pooling_kernel_size or 1
        stride = (
            self.glm_tokenizer.conv1.stride[0]
            * self.glm_tokenizer.conv2.stride[0]
            * pooling_kernel_size
            * self.feature_extractor.hop_length
        )
        all_speech_tokens = []

        for start in range(len(audios)):
            features = self.feature_extractor(
                audios[start : start + 1],
                sampling_rate=16000,
                return_attention_mask=True,
                return_tensors="pt",
                device=device,
                padding="longest",
                pad_to_multiple_of=stride,
            )
            features = features.to(device=device)
            outputs = self.glm_tokenizer(**features)
            speech_tokens = outputs.quantized_token_ids

            attention_mask = features.attention_mask[
                :,
                :: self.glm_tokenizer.conv1.stride[0]
                * self.glm_tokenizer.conv2.stride[0],
            ]
            attention_mask = attention_mask[
                :, :: self.glm_tokenizer.config.pooling_kernel_size
            ]
            assert attention_mask.shape == speech_tokens.shape

            for i in range(len(speech_tokens)):
                speech_token = speech_tokens[i][attention_mask[i].bool()].tolist()
                all_speech_tokens.extend(speech_token)

        asr_end = None

        if self.audio_projector:
            audio_tokens = torch.tensor(all_speech_tokens).to("cuda")
            input_audio_tokens = audio_tokens

            if punctuation:
                input_text_tokens = torch.tensor(
                    self.tokenizer.encode("<|task_asr|><|punctuation_on|>")
                ).to("cuda")
            else:
                input_text_tokens = torch.tensor(
                    self.tokenizer.encode("<|task_asr|><|punctuation_off|>")
                ).to("cuda")

            bos_token = torch.tensor(self.tokenizer.encode("<|im_start|>")).to("cuda")
            audio_embeds = self.glm_tokenizer.codebook(input_audio_tokens)
            audio_embeds = self.audio_projector(audio_embeds)

            text_embeds = self.embed_tokens_func(input_text_tokens)
            bos_embeds = self.embed_tokens_func(bos_token)

            input_embeds = torch.cat(
                (text_embeds, audio_embeds, bos_embeds), dim=0
            ).unsqueeze(0)

            # beam search
            outputs = self.llm.generate(
                inputs_embeds=input_embeds,
                max_new_tokens=self.model_config.max_token_length,
                num_beams=self.model_config.num_beam,
            )
            asr_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

        else:
            audio_tokens = torch.tensor(all_speech_tokens).to("cuda")
            input_audio_tokens = (
                audio_tokens + self.model_config.added_audio_token_start
            )

            if punctuation:
                input_text_tokens = torch.tensor(
                    self.tokenizer.encode("<|task_asr|><|punctuation_on|>")
                ).to("cuda")
            else:
                input_text_tokens = torch.tensor(
                    self.tokenizer.encode("<|task_asr|><|punctuation_off|>")
                ).to("cuda")

            bos_token = torch.tensor(self.tokenizer.encode("<|im_start|>")).to("cuda")

            input_ids = torch.cat(
                (input_text_tokens, input_audio_tokens, bos_token), dim=0
            ).unsqueeze(0)

            input_embeds = self.embed_tokens_func(input_ids)

            outputs = self.llm.generate(
                inputs_embeds=input_embeds,
                max_new_tokens=self.model_config.max_token_length,
                num_beams=self.model_config.num_beam,
            )
            asr_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

        return asr_text, asr_end, gt_text, wav.shape[0] / sr

    @torch.no_grad()
    def generate(self, batch):
        input_ids = batch["input_ids"]
        label_ids = batch["label_ids"]
        audio_masks = batch["audio_masks"]
        gt_texts = batch["gt_texts"]

        audio_tokens = input_ids[0][audio_masks[0]].to("cuda")
        input_audio_tokens = audio_tokens - self.model_config.added_audio_token_start
        input_text_tokens = torch.tensor(
            self.tokenizer.encode("<|task_asr|><|punctuation_on|>")
        ).to("cuda")
        bos_token = torch.tensor(self.tokenizer.encode("<|im_start|>")).to("cuda")
        audio_embeds = self.glm_tokenizer.codebook(input_audio_tokens)
        audio_embeds = self.audio_projector(audio_embeds)

        text_embeds = self.embed_tokens_func(input_text_tokens)
        bos_embeds = self.embed_tokens_func(bos_token)

        input_embeds = torch.cat(
            (text_embeds, audio_embeds, bos_embeds), dim=0
        ).unsqueeze(0)

        asr_end = None

        outputs = self.llm.generate(
            inputs_embeds=input_embeds,
            max_new_tokens=self.model_config.max_token_length,
            num_beams=self.model_config.num_beam,
        )
        asr_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

        return asr_text, asr_end, gt_texts[0], audio_tokens.shape[0] / 12.5

    @torch.no_grad()
    def batch_generate(self, batch, punctuation):
        wavs = [s["wav"] for s in batch]
        gt_texts = [s["gt_text"] for s in batch]
        srs = [s["sr"] for s in batch]

        device = "cuda"

        audios, indices = [], []
        with torch.no_grad():
            for i, wav in enumerate(wavs):
                wav = wav.cpu().numpy()
                time_step = 0
                while time_step * 16000 < wav.shape[0]:
                    audio_segment = wav[time_step * 16000 : (time_step + 30) * 16000]
                    audios.append(audio_segment)
                    indices.append(i)
                    time_step += 30

            pooling_kernel_size = self.glm_tokenizer.config.pooling_kernel_size or 1
            stride = (
                self.glm_tokenizer.conv1.stride[0]
                * self.glm_tokenizer.conv2.stride[0]
                * pooling_kernel_size
                * self.feature_extractor.hop_length
            )

            all_speech_tokens = [[] for _ in range(len(gt_texts))]

            batch_size = 16
            for start in range(0, len(audios), batch_size):
                features = self.feature_extractor(
                    audios[start : start + batch_size],
                    sampling_rate=16000,
                    return_attention_mask=True,
                    return_tensors="pt",
                    device=device,
                    padding="longest",
                    pad_to_multiple_of=stride,
                )
                features = features.to(device=device)
                outputs = self.glm_tokenizer(**features)
                speech_tokens = outputs.quantized_token_ids

                attention_mask = features.attention_mask[
                    :,
                    :: self.glm_tokenizer.conv1.stride[0]
                    * self.glm_tokenizer.conv2.stride[0],
                ]
                attention_mask = attention_mask[
                    :, :: self.glm_tokenizer.config.pooling_kernel_size
                ]
                assert attention_mask.shape == speech_tokens.shape
                for i in range(len(speech_tokens)):
                    idx = indices[start + i]
                    speech_token = speech_tokens[i][attention_mask[i].bool()].tolist()
                    all_speech_tokens[idx].extend(speech_token)

        asr_end = None

        if self.audio_projector:
            if punctuation:
                input_text_tokens = torch.tensor(
                    self.tokenizer.encode("<|task_asr|><|punctuation_on|>")
                ).to("cuda")
            else:
                input_text_tokens = torch.tensor(
                    self.tokenizer.encode("<|task_asr|><|punctuation_off|>")
                ).to("cuda")

            bos_token = torch.tensor(self.tokenizer.encode("<|im_start|>")).to("cuda")

            input_ids = []
            attention_masks = []
            audio_masks = []
            for audio_tokens in all_speech_tokens:
                audio_tokens = torch.tensor(audio_tokens).to("cuda")
                input_audio_tokens = (
                    audio_tokens + self.model_config.added_audio_token_start
                )

                input_id = torch.cat(
                    (input_text_tokens, input_audio_tokens, bos_token), dim=0
                )
                input_ids.append(input_id)

                audio_mask = input_id >= self.model_config.added_audio_token_start
                audio_masks.append(audio_mask)

                attention_mask = torch.ones_like(input_id, dtype=torch.long)
                attention_masks.append(attention_mask)

            input_ids = pad_sequence(
                input_ids,
                padding_value=self.model_config.pad_token_id,
                padding_side="left",
            ).transpose(0, 1)

            audio_masks = pad_sequence(
                audio_masks, padding_value=False, padding_side="left"
            ).transpose(0, 1)

            attention_masks = pad_sequence(
                attention_masks, padding_value=0, padding_side="left"
            ).transpose(0, 1)

            audio_tokens = input_ids.clone()
            audio_tokens[audio_masks] -= self.model_config.added_audio_token_start
            audio_tokens[~audio_masks] = 0
            audio_embeds = self.glm_tokenizer.codebook(audio_tokens)
            audio_embeds = self.audio_projector(audio_embeds)

            input_ids[audio_masks] = 0
            text_embeds = self.embed_tokens_func(input_ids)

            audio_masks = audio_masks.unsqueeze(-1)
            inputs_embeds = audio_embeds * audio_masks + text_embeds * (
                ~audio_masks
            ) * attention_masks.unsqueeze(-1)

            outputs = self.llm.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=self.model_config.max_token_length,
                num_beams=self.model_config.num_beam,
                attention_mask=attention_masks,
                # do_sample=False,
            )
            asr_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        else:
            if punctuation:
                input_text_tokens = torch.tensor(
                    self.tokenizer.encode("<|task_asr|><|punctuation_on|>")
                ).to("cuda")
            else:
                input_text_tokens = torch.tensor(
                    self.tokenizer.encode("<|task_asr|><|punctuation_off|>")
                ).to("cuda")

            bos_token = torch.tensor(self.tokenizer.encode("<|im_start|>")).to("cuda")

            input_ids = []
            attention_masks = []
            for audio_tokens in all_speech_tokens:
                audio_tokens = torch.tensor(audio_tokens).to("cuda")
                input_audio_tokens = (
                    audio_tokens + self.model_config.added_audio_token_start
                )

                input_id = torch.cat(
                    (input_text_tokens, input_audio_tokens, bos_token), dim=0
                )
                input_ids.append(input_id)

                attention_mask = torch.ones_like(input_id)
                attention_masks.append(attention_mask)

            input_ids = pad_sequence(
                input_ids,
                padding_value=self.model_config.pad_token_id,
                padding_side="left",
            ).transpose(0, 1)

            attention_masks = pad_sequence(
                attention_masks, padding_value=0, padding_side="left"
            ).transpose(0, 1)

            inputs_embeds = self.embed_tokens_func(input_ids)

            outputs = self.llm.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=self.model_config.max_token_length,
                num_beams=self.model_config.num_beam,
                attention_mask=attention_masks,
            )
            asr_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return (
            asr_text,
            asr_end,
            gt_texts,
            [wavs[i].shape[0] / srs[i] for i in range(len(srs))],
        )
