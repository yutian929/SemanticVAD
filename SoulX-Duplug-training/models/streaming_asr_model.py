import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
import torch.optim as optim

import re
import math
from tqdm import tqdm
import pytorch_lightning as pl
import peft
from peft import LoraConfig, get_peft_model

from models.glm_4_voice.speech_tokenizer.modeling_whisper import WhisperVQEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import WhisperFeatureExtractor
from utils.sparkvox.utils.scheduler import WarmupAnnealSteps
from models._train_heads import TokenHeadsMixin


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


class Streaming_ASR_Model(TokenHeadsMixin, pl.LightningModule):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.model_config = config.model_config
        self.train_config = config.train_config
        self.asr_eos_token_id = config.model_config.asr_eos_token_id
        self.eos_loss_rate = config.train_config.eos_loss_rate
        self.lm_vocab_size = config.model_config.lm_vocab_size
        self.best_val_acc = 0.0
        self.save_hyperparameters(self.config)

        self.sampling_rate = self.model_config.sampling_rate  # 16000
        self.token_samples = int(0.08 * self.sampling_rate)

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

        if self.model_config.init_ckpt_path:
            print(f"loading state dict from {self.model_config.init_ckpt_path}...")
            checkpoint = torch.load(
                self.model_config.init_ckpt_path,
                map_location=torch.device("cpu"),
                weights_only=False,
            )
            state_dict = (
                checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
            )
            try:
                self.load_state_dict(state_dict)
            except Exception as e:
                print(f"load_state_dict failed: {e}. Retry with strict=False")
                self.load_state_dict(state_dict, strict=False)

            del checkpoint

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

            if self.model_config.init_ckpt_path_lora:
                print(
                    f"loading state dict from {self.model_config.init_ckpt_path_lora}..."
                )
                checkpoint = torch.load(
                    self.model_config.init_ckpt_path_lora,
                    map_location=torch.device("cpu"),
                    weights_only=False,
                )
                state_dict = (
                    checkpoint["state_dict"]
                    if "state_dict" in checkpoint
                    else checkpoint
                )
                try:
                    self.load_state_dict(state_dict)
                except Exception as e:
                    print(f"load_state_dict failed: {e}. Retry with strict=False")
                    self.load_state_dict(state_dict, strict=False)

                del checkpoint

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
        audio_masks = batch["audio_masks"]
        label_ids = batch["label_ids"]
        label_text_ids = batch["label_text_ids"]
        label_eos_ids = batch["label_eos_ids"]

        model_outputs = self((input_ids, audio_masks, label_ids))

        x_ori = model_outputs.logits

        text_loss = self._shifted_ce(x_ori, label_text_ids)
        eos_loss = self._shifted_ce(x_ori, label_eos_ids)

        # loss = self._shifted_ce(x_ori, label_ids)

        loss = self.eos_loss_rate * eos_loss + (1 - self.eos_loss_rate) * text_loss

        preds = torch.argmax(x_ori, -1)
        acc = self._shifted_acc(preds, label_ids)
        text_acc = self._shifted_acc(preds, label_text_ids)
        eos_acc = self._shifted_acc(preds, label_eos_ids)

        self.log("train_loss", loss, prog_bar=True, logger=True, rank_zero_only=True)
        # self.log(
        #     "train_text_loss",
        #     text_loss,
        #     prog_bar=True,
        #     logger=True,
        #     rank_zero_only=True,
        # )
        # self.log(
        #     "train_eos_loss", eos_loss, prog_bar=True, logger=True, rank_zero_only=True
        # )

        self.log("train_acc", acc, prog_bar=True, logger=True, rank_zero_only=True)
        self.log(
            "train_text_acc", text_acc, prog_bar=True, logger=True, rank_zero_only=True
        )
        self.log(
            "train_eos_acc", eos_acc, prog_bar=True, logger=True, rank_zero_only=True
        )

        return loss

    def on_validation_epoch_start(self):
        self.val_loss = []
        self.val_loss_text = []
        self.val_loss_eos = []
        self.val_acc = []
        self.val_acc_text = []
        self.val_acc_eos = []

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        audio_masks = batch["audio_masks"]
        label_ids = batch["label_ids"]
        label_text_ids = batch["label_text_ids"]
        label_eos_ids = batch["label_eos_ids"]

        model_outputs = self((input_ids, audio_masks, label_ids))

        x_ori = model_outputs.logits

        text_loss = self._shifted_ce(x_ori, label_text_ids)
        eos_loss = self._shifted_ce(x_ori, label_eos_ids)

        loss = self._shifted_ce(x_ori, label_ids)

        # loss = (
        #     self.eos_loss_rate * eos_loss + (1 - self.eos_loss_rate) * text_loss
        # )

        preds = torch.argmax(x_ori, -1)
        acc = self._shifted_acc(preds, label_ids)
        text_acc = self._shifted_acc(preds, label_text_ids)
        eos_acc = self._shifted_acc(preds, label_eos_ids)

        self.val_loss.append(loss)
        self.val_loss_text.append(text_loss)
        self.val_loss_eos.append(eos_loss)
        self.val_acc.append(acc)
        self.val_acc_text.append(text_acc)
        self.val_acc_eos.append(eos_acc)

    def on_validation_epoch_end(self):
        avg_loss = torch.nanmean(torch.stack(self.val_loss))
        avg_loss_text = torch.nanmean(torch.stack(self.val_loss_text))
        avg_loss_eos = torch.nanmean(torch.stack(self.val_loss_eos))
        avg_acc = torch.nanmean(torch.stack(self.val_acc))
        avg_acc_text = torch.nanmean(torch.stack(self.val_acc_text))
        avg_acc_eos = torch.nanmean(torch.stack(self.val_acc_eos))

        self.log(f"val_loss", avg_loss, prog_bar=True, logger=True, sync_dist=True)
        self.log(
            f"val_loss_text", avg_loss_text, prog_bar=True, logger=True, sync_dist=True
        )
        self.log(
            f"val_loss_eos", avg_loss_eos, prog_bar=True, logger=True, sync_dist=True
        )
        self.log(f"val_acc", avg_acc, prog_bar=True, logger=True, sync_dist=True)
        self.log(
            f"val_acc_text", avg_acc_text, prog_bar=True, logger=True, sync_dist=True
        )
        self.log(
            f"val_acc_eos", avg_acc_eos, prog_bar=True, logger=True, sync_dist=True
        )

        self.best_val_acc = max(self.best_val_acc, avg_acc)
        self.log(
            f"best_val_acc",
            self.best_val_acc,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

    def test_step(self, batch, batch_idx):
        asr_text, asr_end, gt_text, duration = self.stream_generate(batch)
        print(f"ASR result: {asr_text}")

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.train_config.learning_rate,
            weight_decay=self.train_config.weight_decay,
            betas=self.train_config.betas,
            eps=self.train_config.eps,
        )

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

    def check_en(self, text):
        symbol_pattern = re.compile(
            r"[\u0020-\u002F\u003A-\u0040\u005B-\u0060\u007B-\u007E"
            r"\u2000-\u206F"
            r"\u3000-\u303F"
            r"\uFF00-\uFFEF]"
        )

        for char in reversed(text):
            if char.isdigit() or symbol_pattern.match(char):
                continue
            if char >= "\u4e00" and char <= "\u9fff":  # is chinese
                return False
            else:
                return True

        return True

    def repetition_penalty(self, logits, generated_ids, repetition_penalty):
        """
        Apply repetition penalty to the logits.
        """
        if repetition_penalty == 1.0:
            return logits

        # Gather the logits for generated_ids
        score = torch.gather(logits, -1, generated_ids.unsqueeze(0))

        # Apply penalty
        score = torch.where(
            score < 0, score * repetition_penalty, score / repetition_penalty
        )

        # Scatter the updated scores back into logits
        logits.scatter_(-1, generated_ids.unsqueeze(0), score)

        return logits

    @torch.no_grad()
    def generate(self, audio_tokens):
        audio_tokens = audio_tokens.to("cuda")
        input_audio_tokens = audio_tokens[:2]
        input_text_tokens = torch.tensor(
            self.tokenizer.encode("<|task_asr|><|punctuation_off|>")
        ).to("cuda")
        audio_pad_token = torch.tensor(self.tokenizer.encode("<|padding|>")).to("cuda")
        audio_embeds = self.glm_tokenizer.codebook(input_audio_tokens)
        audio_embeds = self.audio_projector(audio_embeds)

        text_embeds = self.embed_tokens_func(input_text_tokens)

        input_embeds = torch.cat((text_embeds, audio_embeds), dim=0).unsqueeze(0)
        audio_index = 2
        past_key_values = None
        asr_result = []
        asr_segment = []
        asr_end = False
        generated_ids = torch.zeros(
            [self.model_config.max_token_length], dtype=torch.long
        ).to("cuda")

        # TODO: change to beam search
        for step in tqdm(range(self.model_config.max_token_length), desc="Generating"):
            outputs = self.llm(
                inputs_embeds=input_embeds,
                past_key_values=past_key_values,
                use_cache=True,
            )

            logits = outputs.logits[0]
            logits = self.repetition_penalty(
                logits, generated_ids[:step], self.model_config.asr_repetition_penalty
            )
            past_key_values = outputs.past_key_values
            pred = torch.argmax(logits, -1)[-1]
            generated_ids[step] = pred

            if pred == self.asr_eos_token_id:
                if asr_segment:
                    try:
                        asr_segment_text = self.tokenizer.decode(
                            torch.stack(asr_segment, dim=0), skip_special_tokens=True
                        ).strip()

                        if self.check_en(asr_segment_text):
                            asr_segment_text += " "
                    except:
                        asr_segment_text = ""

                    asr_result.append(asr_segment_text)

                asr_segment = []
                if audio_index < audio_tokens.shape[0]:
                    input_audio_tokens = audio_tokens[audio_index : audio_index + 2]
                    audio_embeds = self.glm_tokenizer.codebook(input_audio_tokens)
                    audio_embeds = self.audio_projector(audio_embeds)

                    if audio_embeds.shape[0] != 2:
                        audio_pad_embeds = self.embed_tokens_func(audio_pad_token)
                        input_embeds = torch.cat(
                            (audio_embeds, audio_pad_embeds), dim=0
                        ).unsqueeze(0)
                    else:
                        input_embeds = audio_embeds.unsqueeze(0)

                    audio_index += 2
                else:
                    asr_end = True
                    break
            else:
                asr_segment.append(pred)
                input_embeds = self.embed_tokens_func(pred.unsqueeze(0)).unsqueeze(0)

        try:
            asr_text = "".join(asr_result).strip()
        except:
            asr_text = ""

        return asr_text, asr_end

    @torch.no_grad()
    def generate_from_audio(self, batch, punctuation):
        assert len(batch) == 1

        chunk_size = self.model_config.chunk_size  # 960
        chunk_token_len = chunk_size // 80

        wav = batch[0]["wav"]
        gt_text = batch[0]["gt_text"]
        sr = batch[0]["sr"]

        device = "cuda"

        # extract token
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

        # lm generate
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

        audio_pad_token = torch.tensor(self.tokenizer.encode("<|padding|>")).to("cuda")
        audio_eos_token = torch.tensor(self.tokenizer.encode("<|end_of_sentence|>")).to(
            "cuda"
        )

        all_audio_embeds = self.glm_tokenizer.codebook(input_audio_tokens)
        all_audio_embeds = self.audio_projector(all_audio_embeds)

        text_embeds = self.embed_tokens_func(input_text_tokens)
        audio_pad_embeds = self.embed_tokens_func(audio_pad_token)
        audio_eos_embeds = self.embed_tokens_func(audio_eos_token)

        chunk_num = math.ceil(all_audio_embeds.shape[0] / chunk_token_len)

        past_key_values = None
        asr_result = []
        asr_end = None

        # TODO: change to beam search
        # for chunk in tqdm(range(chunk_num), desc="Generating"):
        for chunk in range(chunk_num):
            asr_segment = []

            audio_embeds = all_audio_embeds[
                chunk * chunk_token_len : (chunk + 1) * chunk_token_len
            ]

            if audio_embeds.shape[0] != chunk_token_len:
                audio_embeds = torch.cat(
                    (
                        audio_embeds,
                        audio_pad_embeds.expand(
                            chunk_token_len - audio_embeds.shape[0], -1
                        ),
                    ),
                    dim=0,
                )

            if chunk:
                input_embeds = torch.cat(
                    (audio_eos_embeds, audio_embeds), dim=0
                ).unsqueeze(0)
            else:
                input_embeds = torch.cat((text_embeds, audio_embeds), dim=0).unsqueeze(
                    0
                )

            for i in range(self.model_config.max_chunk_token_length):
                outputs = self.llm(
                    inputs_embeds=input_embeds,
                    past_key_values=past_key_values,
                    use_cache=True,
                )

                logits = outputs.logits[0]
                past_key_values = outputs.past_key_values
                pred = torch.argmax(logits, -1)[-1]

                if pred == self.asr_eos_token_id:
                    break
                else:
                    asr_segment.append(pred)
                    input_embeds = self.embed_tokens_func(pred.unsqueeze(0)).unsqueeze(
                        0
                    )

            if asr_segment:
                try:
                    asr_segment_text = self.tokenizer.decode(
                        torch.stack(asr_segment, dim=0), skip_special_tokens=True
                    ).strip()

                    if self.check_en(asr_segment_text):
                        asr_segment_text += " "
                except:
                    asr_segment_text = ""

                asr_result.append(asr_segment_text)

        try:
            asr_text = "".join(asr_result).strip()
        except:
            asr_text = ""

        return asr_text, asr_end, gt_text, wav.shape[0] / sr

    @torch.no_grad()
    def stream_generate(self, batch, punctuation):
        assert len(batch) == 1

        chunk_size = self.model_config.chunk_size  # 960
        chunk_token_len = chunk_size // 80

        wav = batch[0]["wav"]
        gt_text = batch[0]["gt_text"]
        sr = batch[0]["sr"]

        device = "cuda"

        # extract token
        wav = wav.cpu().numpy()

        # TODO: set this in config
        sampling_rate = self.sampling_rate  # Fixed at 16kHz
        segment_duration = 0.16
        lookback_duration = 0.96
        lookahead_duration = 0.04
        valid_samples_per_segment = int(segment_duration * sampling_rate)
        lookback_samples_per_segment = int(lookback_duration * sampling_rate)
        lookahead_samples_per_segment = int(lookahead_duration * sampling_rate)
        segment_token_len = math.ceil(segment_duration / 0.08)
        discard_token_len = math.ceil(lookahead_duration / 0.08)
        audios = []
        valid_range = []

        for start_sample in range(0, wav.shape[0], valid_samples_per_segment):
            audio_segment = wav[
                max(0, start_sample - lookback_samples_per_segment) : start_sample
                + valid_samples_per_segment
                + lookahead_samples_per_segment
            ]
            audios.append(audio_segment)
            start_index = (
                start_sample - max(0, start_sample - lookback_samples_per_segment)
            ) // self.token_samples
            end_index = min(
                start_index + 2, math.ceil(audio_segment.shape[0] / self.token_samples)
            )
            valid_range.append((start_index, end_index))

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
                all_speech_tokens.extend(
                    speech_token[valid_range[start + i][0] : valid_range[start + i][1]]
                )

        # lm generate
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

        audio_pad_token = torch.tensor(self.tokenizer.encode("<|padding|>")).to("cuda")
        audio_eos_token = torch.tensor(self.tokenizer.encode("<|end_of_sentence|>")).to(
            "cuda"
        )

        all_audio_embeds = self.glm_tokenizer.codebook(input_audio_tokens)
        all_audio_embeds = self.audio_projector(all_audio_embeds)

        text_embeds = self.embed_tokens_func(input_text_tokens)
        audio_pad_embeds = self.embed_tokens_func(audio_pad_token)
        audio_eos_embeds = self.embed_tokens_func(audio_eos_token)

        chunk_num = math.ceil(all_audio_embeds.shape[0] / chunk_token_len)

        past_key_values = None
        asr_result = []
        asr_end = None

        # TODO: change to beam search
        # for chunk in tqdm(range(chunk_num), desc="Generating"):
        for chunk in range(chunk_num):
            asr_segment = []

            audio_embeds = all_audio_embeds[
                chunk * chunk_token_len : (chunk + 1) * chunk_token_len
            ]

            if audio_embeds.shape[0] != chunk_token_len:
                audio_embeds = torch.cat(
                    (
                        audio_embeds,
                        audio_pad_embeds.expand(
                            chunk_token_len - audio_embeds.shape[0], -1
                        ),
                    ),
                    dim=0,
                )

            if chunk:
                input_embeds = torch.cat(
                    (audio_eos_embeds, audio_embeds), dim=0
                ).unsqueeze(0)
            else:
                input_embeds = torch.cat((text_embeds, audio_embeds), dim=0).unsqueeze(
                    0
                )

            for i in range(self.model_config.max_chunk_token_length):
                outputs = self.llm(
                    inputs_embeds=input_embeds,
                    past_key_values=past_key_values,
                    use_cache=True,
                )

                logits = outputs.logits[0]
                past_key_values = outputs.past_key_values
                pred = torch.argmax(logits, -1)[-1]

                if pred == self.asr_eos_token_id:
                    break
                else:
                    asr_segment.append(pred)
                    input_embeds = self.embed_tokens_func(pred.unsqueeze(0)).unsqueeze(
                        0
                    )

            if asr_segment:
                try:
                    asr_segment_text = self.tokenizer.decode(
                        torch.stack(asr_segment, dim=0), skip_special_tokens=True
                    ).strip()

                    if self.check_en(asr_segment_text):
                        asr_segment_text += " "
                except:
                    asr_segment_text = ""

                asr_result.append(asr_segment_text)

        try:
            asr_text = "".join(asr_result).strip()
        except:
            asr_text = ""

        return asr_text, asr_end, gt_text, wav.shape[0] / sr
