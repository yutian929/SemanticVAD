import os, sys
import json
import numpy as np
import random
import math
import torch, torchaudio
import pytorch_lightning as pl
import argparse
from omegaconf import OmegaConf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.text_utils import split_cn_en, check_en, get_lcs_substrings
from utils.backchannel_utils import check_backchannel, remove_leading_backchannel
from utils.MyTn.textnorm import zh_norm, zh_remove_punc
from config.config import RunConfig
from models.state_prediction_model import State_Prediction_Model
from transformers import WhisperFeatureExtractor


def state_map(state: str):
    s_map = {
        "<|user_complete|>": "speak",
        "<|user_incomplete|>": "wait",
        "<|user_backchannel|>": "backchannel",
        "<|user_idle|>": "idle",
        "<|user_nonidle|>": "nonidle",
    }

    return s_map.get(state, "unknown")


def duplex_predict_160_cascade_asr(config, model, in_wav_path, cascade_asr):
    device = config.infer_config.device
    chunk_token_len_small = config.infer_config.input["chunk_token_len_small"]  # 2
    mistake_times = 0

    assert os.path.exists(in_wav_path)
    wav, sr = torchaudio.load(in_wav_path)

    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)

    if sr != 16000:
        if sr not in model._resample_buffer:
            model._resample_buffer[sr] = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=16000
            )
        wav = model._resample_buffer[sr](wav).squeeze()
        sr = 16000
    else:
        wav = wav.squeeze()

    wav = wav.cpu().numpy()
    wav = np.concatenate([wav, np.zeros(32000)], axis=0)

    sampling_rate = config.infer_config.input["sample_rate"]  # Fixed at 16kHz
    valid_samples_per_segment = config.infer_config.input["chunk_size"]
    lookback_samples_per_segment = config.infer_config.input["audio_back_size"]
    lookahead_samples_per_segment = config.infer_config.input["audio_ahead_size"]

    input_text_tokens = torch.tensor(
        model.tokenizer.encode("<|task_duplex_predict|><|punctuation_off|>")
    ).to("cuda")

    audio_pad_token = torch.tensor(model.tokenizer.encode("<|padding|>")).to("cuda")
    audio_bos_token = torch.tensor(model.tokenizer.encode("<|begin_of_sentence|>")).to(
        "cuda"
    )
    audio_eos_token = torch.tensor(model.tokenizer.encode("<|end_of_sentence|>")).to(
        "cuda"
    )

    action_speak_token = torch.tensor(model.tokenizer.encode("<|user_complete|>")).to(
        "cuda"
    )
    action_wait_token = torch.tensor(model.tokenizer.encode("<|user_incomplete|>")).to(
        "cuda"
    )
    non_idle_token = torch.tensor(model.tokenizer.encode("<|user_nonidle|>")).to("cuda")

    if hasattr(model.llm.model, "embed_tokens"):
        embed_fn = model.llm.model.embed_tokens
    elif hasattr(model.llm.model.model, "embed_tokens"):
        embed_fn = model.llm.model.model.embed_tokens
    else:
        embed_fn = model.llm.model.model.model.embed_tokens

    text_embeds = embed_fn(input_text_tokens)
    audio_pad_embeds = embed_fn(audio_pad_token)
    audio_bos_embeds = embed_fn(audio_bos_token)
    audio_eos_embeds = embed_fn(audio_eos_token)
    action_speak_embeds = embed_fn(action_speak_token)
    action_wait_embeds = embed_fn(action_wait_token)
    non_idle_embeds = embed_fn(non_idle_token)

    pooling_kernel_size = model.glm_tokenizer.config.pooling_kernel_size or 1
    stride = (
        model.glm_tokenizer.conv1.stride[0]
        * model.glm_tokenizer.conv2.stride[0]
        * pooling_kernel_size
        * model.feature_extractor.hop_length
    )

    past_state = {
        "input_embeds": text_embeds,
        "past_key_values": None,
        "delta_text": [],
        "cascade_text": "",
        "state": "",
        "mistake_len": 0,
        "checkpoint": None,
    }

    # Initialize buffer
    buffer = (
        np.random.randn(lookback_samples_per_segment + lookahead_samples_per_segment)
        * 0.0001
    ).astype(np.float32)

    cascade_buffer = (np.random.randn(int(3.2 * sampling_rate)) * 0.00001).astype(
        np.float32
    )

    state_list = []

    for start_sample in range(0, wav.shape[0], valid_samples_per_segment):
        audio_chunk = wav[start_sample : start_sample + valid_samples_per_segment]
        if len(audio_chunk) < valid_samples_per_segment:
            padding = np.zeros(
                valid_samples_per_segment - len(audio_chunk), dtype=np.float32
            )
            audio_chunk = np.concatenate([audio_chunk, padding])

        # Update buffer logic
        buffer = np.concatenate([buffer, audio_chunk])

        if (
            len(buffer)
            >= valid_samples_per_segment
            + lookback_samples_per_segment
            + lookahead_samples_per_segment
        ):
            audio_back = buffer[:lookback_samples_per_segment]
            process_chunk = buffer[
                lookback_samples_per_segment : lookback_samples_per_segment
                + valid_samples_per_segment
            ]
            audio_ahead = buffer[
                lookback_samples_per_segment
                + valid_samples_per_segment : lookback_samples_per_segment
                + valid_samples_per_segment
                + lookahead_samples_per_segment
            ]
            buffer = buffer[valid_samples_per_segment:]

            audio_segment = np.concatenate(
                [audio_back, process_chunk, audio_ahead], axis=0
            )
            cascade_buffer = np.concatenate([cascade_buffer, process_chunk])

            start_index = len(audio_back) // model.token_samples
            end_index = min(
                start_index + 2, math.ceil(audio_segment.shape[0] / model.token_samples)
            )
            valid_range = (start_index, end_index)

            features = model.feature_extractor(
                [audio_segment],
                sampling_rate=16000,
                return_attention_mask=True,
                return_tensors="pt",
                device=device,
                padding="longest",
                pad_to_multiple_of=stride,
            )
            features = features.to(device=device)
            outputs = model.glm_tokenizer(**features)
            speech_tokens = outputs.quantized_token_ids

            attention_mask = features.attention_mask[
                :,
                :: model.glm_tokenizer.conv1.stride[0]
                * model.glm_tokenizer.conv2.stride[0],
            ]
            attention_mask = attention_mask[
                :, :: model.glm_tokenizer.config.pooling_kernel_size
            ]
            assert attention_mask.shape == speech_tokens.shape
            assert len(speech_tokens) == 1

            speech_token = speech_tokens[0][attention_mask[0].bool()].tolist()
            audio_tokens = speech_token[valid_range[0] : valid_range[1]]

            input_audio_tokens = torch.tensor(audio_tokens).to("cuda")
            audio_embeds = model.glm_tokenizer.codebook(input_audio_tokens)
            audio_embeds = model.audio_projector(audio_embeds)

            if audio_embeds.shape[0] != chunk_token_len_small:
                audio_embeds = torch.cat(
                    (
                        audio_embeds,
                        audio_pad_embeds.expand(
                            chunk_token_len_small - audio_embeds.shape[0], -1
                        ),
                    ),
                    dim=0,
                )

            past_state["input_embeds"] = torch.cat(
                (past_state["input_embeds"], audio_embeds), dim=0
            ).unsqueeze(0)

            with torch.no_grad():
                outputs = model.llm(
                    inputs_embeds=past_state["input_embeds"],
                    past_key_values=past_state["past_key_values"],
                    use_cache=True,
                )
                logits = outputs.logits[0]
                current_kv = outputs.past_key_values
                pred = torch.argmax(logits, -1)[-1]

                delta_text = ""
                need_correction = False
                corrected_prev_delta = ""

                if (
                    pred != model.asr_eos_token_id
                    or past_state["state"] == "<|user_nonidle|>"
                ):
                    # if True:
                    full_text = remove_leading_backchannel(
                        cascade_asr.recognize(cascade_buffer, sampling_rate)
                    )
                    history_text = past_state.get("cascade_text", "")
                    norm_full_text = split_cn_en(
                        zh_norm(zh_remove_punc(full_text.strip()))
                    )
                    norm_history_text = split_cn_en(
                        zh_norm(zh_remove_punc(history_text.strip()))
                    )

                    if len(norm_full_text) >= 5 and len(norm_history_text) >= 5:
                        backup_norm_full_text = norm_full_text.copy()
                        backup_norm_history_text = norm_history_text.copy()
                        norm_full_text, norm_history_text = get_lcs_substrings(
                            norm_full_text, norm_history_text
                        )

                    prev_delta = (
                        past_state["delta_text"][-1] if past_state["delta_text"] else ""
                    )
                    prev_delta_split = split_cn_en(prev_delta)
                    len_prev = len(prev_delta_split)

                    if len_prev > len(norm_history_text):
                        mistake_times += 1
                        norm_full_text = backup_norm_full_text
                        norm_history_text = backup_norm_history_text

                    history_base = (
                        norm_history_text[:-len_prev]
                        if len_prev > 0
                        else norm_history_text
                    )

                    if len(norm_full_text) > len(norm_history_text):
                        current_segment_in_full = norm_full_text[
                            len(history_base) : len(history_base) + len_prev
                        ]
                        if current_segment_in_full == prev_delta_split:
                            # 3.1 Consistent
                            delta_text = "".join(
                                [
                                    (s + " ") if check_en(s) else s
                                    for s in norm_full_text[len(norm_history_text) :]
                                ]
                            ).strip()
                        else:
                            # 3.2 Inconsistent
                            need_correction = True
                            corrected_prev_delta = "".join(
                                [
                                    (s + " ") if check_en(s) else s
                                    for s in current_segment_in_full
                                ]
                            ).strip()
                            delta_text = "".join(
                                [
                                    (s + " ") if check_en(s) else s
                                    for s in norm_full_text[
                                        len(history_base) + len_prev :
                                    ]
                                ]
                            ).strip()

                    elif len(norm_full_text) == len(norm_history_text):
                        current_segment_in_full = norm_full_text[len(history_base) :]
                        if current_segment_in_full == prev_delta_split:
                            # 4.1 Consistent
                            delta_text = ""
                        else:
                            # 4.2 Inconsistent
                            need_correction = True
                            # maybe homophone
                            corrected_prev_delta = "".join(
                                [
                                    (s + " ") if check_en(s) else s
                                    for s in current_segment_in_full
                                ]
                            ).strip()
                            delta_text = ""

                    else:  # len(norm_full) < len(norm_history)
                        # 5
                        need_correction = True
                        remainder = norm_full_text[len(history_base) :]
                        corrected_prev_delta = ""
                        delta_text = "".join(
                            [(s + " ") if check_en(s) else s for s in remainder]
                        ).strip()

                    past_state["cascade_text"] = "".join(
                        [(s + " ") if check_en(s) else s for s in norm_full_text]
                    ).strip()
                    print(
                        f"[ASR] History: {''.join([(s + ' ') if check_en(s) else s for s in norm_history_text]).strip()}"
                    )
                    print(f"[ASR] Full: {past_state['cascade_text']}")
                    print(f"[Need Correction]: {need_correction}")
                    print(f"[Prev Delta]: {corrected_prev_delta}")
                    print(f"[Delta]: {delta_text}")

                if need_correction and past_state["checkpoint"] is not None:
                    print("--- Correction Triggered ---")
                    past_state["past_key_values"] = past_state["checkpoint"]

                    embeds_list = []
                    # 1. Corrected Prev Text
                    if corrected_prev_delta:
                        ids = model.tokenizer.encode(
                            corrected_prev_delta, add_special_tokens=False
                        )
                        if ids:
                            t_ids = torch.tensor(ids, dtype=torch.long).to(device)
                            embeds_list.append(embed_fn(t_ids))

                    # 2. EOS
                    embeds_list.append(audio_eos_embeds)
                    # 3. NonIdle
                    embeds_list.append(non_idle_embeds)
                    # 4. Audio (Current Chunk)
                    embeds_list.append(audio_embeds)

                    correction_input = torch.cat(embeds_list, dim=0).unsqueeze(0)

                    outputs = model.llm(
                        inputs_embeds=correction_input,
                        past_key_values=past_state["past_key_values"],
                        use_cache=True,
                    )
                    past_state["past_key_values"] = outputs.past_key_values

                    if past_state["delta_text"]:
                        past_state["delta_text"][-1] = corrected_prev_delta
                else:
                    past_state["past_key_values"] = current_kv

                # Save Checkpoint for NEXT chunk (KV after Audio, before Text)
                past_state["checkpoint"] = past_state["past_key_values"]

                past_state["delta_text"].append(delta_text)
                max_len = int(3.2 * sampling_rate)
                if len(cascade_buffer) > max_len:
                    cascade_buffer = cascade_buffer[-max_len:]
                    delta_list = past_state["delta_text"][-20:]
                    total_tokens = 0
                    for t in delta_list:
                        total_tokens += len(split_cn_en(t))

                    cascade_tokens = split_cn_en(past_state["cascade_text"])
                    if total_tokens > 0:
                        keep_tokens = cascade_tokens[-total_tokens:]
                    else:
                        keep_tokens = []

                    update_text = "".join(
                        [(s + " ") if check_en(s) else s for s in keep_tokens]
                    ).strip()
                    past_state["cascade_text"] = update_text

                input_embeds_next = audio_eos_embeds.unsqueeze(0)
                if delta_text:
                    ids = model.tokenizer.encode(delta_text, add_special_tokens=False)
                    input_ids = torch.tensor(ids, dtype=torch.long).to(device)
                    if len(input_ids) > 0:
                        embeds = embed_fn(input_ids).unsqueeze(0)
                        input_embeds_next = torch.cat(
                            (embeds, input_embeds_next), dim=1
                        )

                if delta_text or corrected_prev_delta:
                    print(
                        f"[Concat]: {''.join([(s + ' ') if check_en(s) else s for s in past_state['delta_text']]).strip()}"
                    )

                past_state["input_embeds"] = input_embeds_next
                outputs = model.llm(
                    inputs_embeds=past_state["input_embeds"],
                    past_key_values=past_state["past_key_values"],
                    use_cache=True,
                )
                logits = outputs.logits[0]
                pred = torch.argmax(logits, -1)[-1]
                state = model.tokenizer.decode(pred)

                if state == "<|user_nonidle|>" and not delta_text:
                    past_state["mistake_len"] += 1
                else:
                    past_state["mistake_len"] = 0

                if (
                    past_state["state"] == "<|user_nonidle|>"
                    and state == "<|user_idle|>"
                ) or past_state["mistake_len"] >= config.infer_config.max_mistake_num:
                    mistake_times += 1
                    if (
                        logits[-1, model.config.model_config.user_complete_token_id]
                        > logits[-1, model.config.model_config.user_incomplete_token_id]
                    ):
                        state = "<|user_complete|>"
                        past_state["input_embeds"] = action_speak_embeds
                    else:
                        state = "<|user_incomplete|>"
                        past_state["input_embeds"] = action_wait_embeds
                else:
                    past_state["input_embeds"] = embed_fn(pred.unsqueeze(0))

                past_state["past_key_values"] = outputs.past_key_values
                past_state["state"] = state
                print(
                    f"[{start_sample/16000:.3f}-{(start_sample+valid_samples_per_segment)/16000:.3f}]: {state}"
                )

                if config.infer_config.single_round and (
                    state == "<|user_complete|>" or state == "<|user_incomplete|>"
                ):
                    past_state = {
                        "input_embeds": text_embeds,
                        "past_key_values": None,
                        "delta_text": [],
                        "cascade_text": "",
                        "state": "",
                        "mistake_len": 0,
                        "checkpoint": None,
                    }

                    buffer = (
                        np.random.randn(
                            lookback_samples_per_segment + lookahead_samples_per_segment
                        )
                        * 0.0001
                    ).astype(np.float32)

                    cascade_buffer = (
                        np.random.randn(int(3.2 * sampling_rate)) * 0.00001
                    ).astype(np.float32)

                state_list.append(
                    {
                        "state": state_map(state),
                        "timestamp": [
                            start_sample / 16000.0,
                            (start_sample + valid_samples_per_segment) / 16000.0,
                        ],
                    }
                )

    json_filename = in_wav_path.replace(".wav", "_states.json")
    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(state_list, f, ensure_ascii=False, indent=2)

    return mistake_times


def duplex_predict_160(config, model, in_wav_path):
    device = config.infer_config.device
    chunk_token_len_small = config.infer_config.input["chunk_token_len_small"]  # 2
    mistake_times = 0

    assert os.path.exists(in_wav_path)
    wav, sr = torchaudio.load(in_wav_path)

    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)

    if sr != 16000:
        if sr not in model._resample_buffer:
            model._resample_buffer[sr] = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=16000
            )
        wav = model._resample_buffer[sr](wav).squeeze()
        sr = 16000
    else:
        wav = wav.squeeze()

    wav = wav.cpu().numpy()
    wav = np.concatenate([wav, np.zeros(32000)], axis=0)

    sampling_rate = config.infer_config.input["sample_rate"]  # Fixed at 16kHz
    valid_samples_per_segment = config.infer_config.input["chunk_size"]
    lookback_samples_per_segment = config.infer_config.input["audio_back_size"]
    lookahead_samples_per_segment = config.infer_config.input["audio_ahead_size"]

    input_text_tokens = torch.tensor(
        model.tokenizer.encode("<|task_duplex_predict|><|punctuation_off|>")
    ).to("cuda")

    audio_pad_token = torch.tensor(model.tokenizer.encode("<|padding|>")).to("cuda")
    audio_bos_token = torch.tensor(model.tokenizer.encode("<|begin_of_sentence|>")).to(
        "cuda"
    )
    audio_eos_token = torch.tensor(model.tokenizer.encode("<|end_of_sentence|>")).to(
        "cuda"
    )

    action_speak_token = torch.tensor(model.tokenizer.encode("<|user_complete|>")).to(
        "cuda"
    )
    action_wait_token = torch.tensor(model.tokenizer.encode("<|user_incomplete|>")).to(
        "cuda"
    )

    if hasattr(model.llm.model, "embed_tokens"):
        embed_fn = model.llm.model.embed_tokens
    elif hasattr(model.llm.model.model, "embed_tokens"):
        embed_fn = model.llm.model.model.embed_tokens
    else:
        embed_fn = model.llm.model.model.model.embed_tokens

    text_embeds = embed_fn(input_text_tokens)
    audio_pad_embeds = embed_fn(audio_pad_token)
    audio_bos_embeds = embed_fn(audio_bos_token)
    audio_eos_embeds = embed_fn(audio_eos_token)
    action_speak_embeds = embed_fn(action_speak_token)
    action_wait_embeds = embed_fn(action_wait_token)

    pooling_kernel_size = model.glm_tokenizer.config.pooling_kernel_size or 1
    stride = (
        model.glm_tokenizer.conv1.stride[0]
        * model.glm_tokenizer.conv2.stride[0]
        * pooling_kernel_size
        * model.feature_extractor.hop_length
    )

    input_embeds = text_embeds
    past_key_values = None
    pre_state = None
    asr_result = []
    state_list = []
    mistake_len = 0

    # Initialize buffer
    buffer = (
        np.random.randn(lookback_samples_per_segment + lookahead_samples_per_segment)
        * 0.0001
    ).astype(np.float32)

    # simulute streaming
    for start_sample in range(0, wav.shape[0], valid_samples_per_segment):
        audio_chunk = wav[start_sample : start_sample + valid_samples_per_segment]
        if len(audio_chunk) < valid_samples_per_segment:
            padding = np.zeros(
                valid_samples_per_segment - len(audio_chunk), dtype=np.float32
            )
            audio_chunk = np.concatenate([audio_chunk, padding])

        # Update buffer logic
        buffer = np.concatenate([buffer, audio_chunk])

        if (
            len(buffer)
            >= valid_samples_per_segment
            + lookback_samples_per_segment
            + lookahead_samples_per_segment
        ):
            audio_back = buffer[:lookback_samples_per_segment]
            process_chunk = buffer[
                lookback_samples_per_segment : lookback_samples_per_segment
                + valid_samples_per_segment
            ]
            audio_ahead = buffer[
                lookback_samples_per_segment
                + valid_samples_per_segment : lookback_samples_per_segment
                + valid_samples_per_segment
                + lookahead_samples_per_segment
            ]
            buffer = buffer[valid_samples_per_segment:]

            audio_segment = np.concatenate(
                [audio_back, process_chunk, audio_ahead], axis=0
            )

            start_index = len(audio_back) // model.token_samples
            end_index = min(
                start_index + 2, math.ceil(audio_segment.shape[0] / model.token_samples)
            )

            valid_range = (start_index, end_index)

            features = model.feature_extractor(
                [audio_segment],
                sampling_rate=16000,
                return_attention_mask=True,
                return_tensors="pt",
                device=device,
                padding="longest",
                pad_to_multiple_of=stride,
            )
            features = features.to(device=device)
            outputs = model.glm_tokenizer(**features)
            speech_tokens = outputs.quantized_token_ids

            attention_mask = features.attention_mask[
                :,
                :: model.glm_tokenizer.conv1.stride[0]
                * model.glm_tokenizer.conv2.stride[0],
            ]
            attention_mask = attention_mask[
                :, :: model.glm_tokenizer.config.pooling_kernel_size
            ]
            assert attention_mask.shape == speech_tokens.shape
            assert len(speech_tokens) == 1

            speech_token = speech_tokens[0][attention_mask[0].bool()].tolist()
            audio_tokens = speech_token[valid_range[0] : valid_range[1]]

            # lm generate
            input_audio_tokens = torch.tensor(audio_tokens).to("cuda")

            audio_embeds = model.glm_tokenizer.codebook(input_audio_tokens)
            audio_embeds = model.audio_projector(audio_embeds)

            if audio_embeds.shape[0] != chunk_token_len_small:
                audio_embeds = torch.cat(
                    (
                        audio_embeds,
                        audio_pad_embeds.expand(
                            chunk_token_len_small - audio_embeds.shape[0], -1
                        ),
                    ),
                    dim=0,
                )

            input_embeds = torch.cat((input_embeds, audio_embeds), dim=0).unsqueeze(0)

            asr_segment = []

            for i in range(model.config.model_config.max_chunk_token_length):
                outputs = model.llm(
                    inputs_embeds=input_embeds,
                    past_key_values=past_key_values,
                    use_cache=True,
                )

                logits = outputs.logits[0]
                past_key_values = outputs.past_key_values
                pred = torch.argmax(logits, -1)[-1]

                if pred == model.asr_eos_token_id:
                    break
                else:
                    asr_segment.append(pred)
                    input_embeds = embed_fn(pred.unsqueeze(0)).unsqueeze(0)

            asr_segment_text = ""
            if asr_segment:
                try:
                    asr_segment_text = model.tokenizer.decode(
                        torch.stack(asr_segment, dim=0), skip_special_tokens=True
                    ).strip()

                    if check_en(asr_segment_text):
                        asr_segment_text += " "
                except:
                    asr_segment_text = ""

                asr_result.append(asr_segment_text)

            print(f"[user segment]: {asr_segment_text}")

            input_embeds = audio_eos_embeds.unsqueeze(0)

            outputs = model.llm(
                inputs_embeds=input_embeds,
                past_key_values=past_key_values,
                use_cache=True,
            )

            logits = outputs.logits[0]
            pred = torch.argmax(logits, -1)[-1]
            state = model.tokenizer.decode(pred)
            input_embeds = embed_fn(pred.unsqueeze(0))

            if state == "<|user_nonidle|>" and not asr_segment_text:
                mistake_len += 1
            else:
                mistake_len = 0

            if (
                pre_state == "<|user_nonidle|>" and state == "<|user_idle|>"
            ) or mistake_len >= config.infer_config.max_mistake_num:
                mistake_times += 1
                if (
                    logits[-1, model.config.model_config.user_complete_token_id]
                    > logits[-1, model.config.model_config.user_incomplete_token_id]
                ):
                    state = "<|user_complete|>"
                    input_embeds = action_speak_embeds
                else:
                    state = "<|user_incomplete|>"
                    input_embeds = action_wait_embeds
            else:
                input_embeds = embed_fn(pred.unsqueeze(0))

            past_key_values = outputs.past_key_values
            pre_state = state

            if config.infer_config.single_round and (
                state == "<|user_complete|>" or state == "<|user_incomplete|>"
            ):
                input_embeds = text_embeds
                past_key_values = None
                asr_result = []
                buffer = (
                    np.random.randn(
                        lookback_samples_per_segment + lookahead_samples_per_segment
                    )
                    * 0.0001
                ).astype(np.float32)

            print(
                f"[{start_sample/16000:.3f}-{(start_sample+valid_samples_per_segment)/16000:.3f}]: {state}"
            )

            state_list.append(
                {
                    "state": state_map(state),
                    "timestamp": [
                        start_sample / 16000.0,
                        (start_sample + valid_samples_per_segment) / 16000.0,
                    ],
                }
            )

    json_filename = in_wav_path.replace(".wav", "_states.json")
    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(state_list, f, ensure_ascii=False, indent=2)

    return mistake_times


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_path", type=str, required=True, help="Path to the config file"
    )

    parser.add_argument(
        "--eval_dir", type=str, required=True, help="Path to the evaluation directory"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    default_config = RunConfig()
    run_config = OmegaConf.load(args.config_path)
    config = OmegaConf.merge(default_config, run_config)

    # set seed
    pl.seed_everything(config.infer_config.seed)
    torch.cuda.manual_seed(config.infer_config.seed)
    torch.manual_seed(config.infer_config.seed)
    np.random.seed(config.infer_config.seed)
    random.seed(config.infer_config.seed)

    model = State_Prediction_Model(config)
    model.feature_extractor = WhisperFeatureExtractor.from_pretrained(
        config.model_config.glm_tokenizer_path
    )

    model.eval().to("cuda")

    wav_files = []
    eval_dir = args.eval_dir

    if eval_dir and os.path.exists(eval_dir):
        if os.path.isdir(eval_dir):
            for root, dirs, files in os.walk(eval_dir):
                for file in files:
                    if file.endswith(".wav"):
                        wav_files.append(os.path.join(root, file))
        else:
            if eval_dir.endswith(".wav"):
                wav_files.append(eval_dir)

    print(f"Found {len(wav_files)} wav files to process.")

    if config.model_config.get("enable_cascade_asr", False):
        from models.asr import ParaformerASR, SensevoiceASR

        if config.infer_config.asr.get("model_name") == "paraformer":
            cascade_asr = ParaformerASR()
        else:
            cascade_asr = SensevoiceASR(
                language=config.infer_config.asr.get("language", "auto")
            )

    mistake_times = 0
    with torch.no_grad():
        for wav_path in wav_files:
            print(f"Processing: {wav_path}")
            if config.model_config.get("enable_cascade_asr", False):
                mistake_times += duplex_predict_160_cascade_asr(
                    config, model, wav_path, cascade_asr
                )
            else:
                mistake_times += duplex_predict_160(config, model, wav_path)
            print(f"mistake_times: {mistake_times}")


if __name__ == "__main__":
    main()
