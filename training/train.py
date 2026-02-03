# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import io
import os
import copy
import random
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List
import tqdm
import itertools
import pandas as pd
import torchvision.transforms as transforms
import torch

import transformers
import tokenizers
from PIL import Image

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, \
    DEFAULT_IM_END_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN
from torch.utils.data import Dataset, DataLoader
from llava.train.llava_trainer import LLaVATrainer, LLAVADPOTrainer, get_batch_logps, concate_pad, concate_pad_three

from llava import conversation as conversation_lib
from llava.model import *
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


from packaging import version

IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default='flat')
    mm_vision_select_feature: Optional[str] = field(default="patch")


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'

    offline_ref_logits: str = ''
    use_image_type: str = "diffusion"  # black, crop, rotate, random
    diffusion_step: int = 500
    eval_data_path: Optional[str] = None


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
                "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)

    task: str = field(
        default='LM',
        metadata={
            'help': 'LM for language modeling. DPO for direct preference optimization'
        }
    )
    dpo_token_weighted: bool = False
    dpo_token_weight: float = 1.1
    dpo_use_average: bool = field(default=False, metadata={"help": "Use per-token average log-prob instead of sum for DPO computations to normalize sequence length effects."})
    use_cross_modal_loss: bool = False
    use_cross_modal_gama: float = 0.3
    use_cross_modal_loss_ref: bool = field(default=True, metadata={"help": "Enable cross-modal reference term L_DPO_r = pi(chosen) - r(chosen)"})
    use_cross_modal_loss_vis: bool = field(default=True, metadata={"help": "Enable cross-modal visual term L_DPO_v = -(pi(win_diffusion) - u(win))"})
    dpo_beta: float = field(default=0.1, metadata={"help": "Temperature beta for DPO loss"})
    tok_beta: float = 0.1
    use_tdpo: bool = True

    # SVCO joint optimization controls
    use_svco_loss: bool = field(default=False, metadata={"help": "Enable SVCO loss and build joint objective L = L_CHiP + lambda * L_SVCO"})
    svco_lambda: float = field(default=0.0, metadata={"help": "Weight for SVCO loss in the joint objective"})
    svco_reference_free: bool = field(default=False, metadata={"help": "If true, drop reference terms in SVCO (reference-free variant)"})
    svco_beta_img_win_vs_no_img: float = field(default=0.1, metadata={"help": "Beta for (res_win|img_win) > (res_win|no_img)"})
    svco_beta_no_img_vs_img_lose: float = field(default=0.1, metadata={"help": "Beta for (res_win|no_img) > (res_win|img_lose)"})
    svco_beta_img_lose_vs_no_img: float = field(default=0.1, metadata={"help": "Beta for (res_lose|img_lose) > (res_lose|no_img)"})
    svco_beta_no_img_vs_img_win: float = field(default=0.1, metadata={"help": "Beta for (res_lose|no_img) > (res_lose|img_win)"})

    # Asymmetric VCO (two-term) controls
    use_vco_loss: bool = field(default=False, metadata={"help": "Enable asymmetric VCO loss using only Attend and Reject terms (two comparisons)"})
    vco_lambda: float = field(default=0.0, metadata={"help": "Weight for VCO loss in the joint objective"})
    vco_reference_free: bool = field(default=False, metadata={"help": "If true, drop reference terms in VCO (reference-free variant)"})

    # -------------------- NEW TOGGLES/WEIGHTS --------------------
    use_chip_loss: bool = field(default=True, metadata={"help": "Enable CHiP (cross-modal DPO) loss"})
    use_new_visual_loss: bool = field(default=False, metadata={"help": "Enable the proposed new visual contrastive loss (L_new)"})
    new_visual_lambda: float = field(default=0.0, metadata={"help": "Weight for L_new in the joint objective"})
    new_visual_beta: float = field(default=0.5, metadata={"help": "Temperature beta for L_new"})
    # -------------------------------------------------------------

    # -------------------- Anchor Loss --------------------
    use_anchor_loss: bool = field(default=False, metadata={"help": "Enable anchor loss to regularize DPO"})
    anchor_loss_beta: float = field(default=0.1, metadata={"help": "Beta for anchor loss (primary view)"})
    use_dual_anchor_loss: bool = field(default=True, metadata={"help": "If true and use_anchor_loss, also add anchor on the second view (e.g., diffusion image)."})
    anchor_loss_beta2: float = field(default=0.1, metadata={"help": "Beta for second-view anchor loss"})
    anchor_loss_margin_delta: float = field(default=0.0, metadata={"help": "Margin delta inside logsigmoid for both anchor terms"})
    anchor_loss_lambda: float = field(default=1.0, metadata={"help": "Overall weight for the (sum of) anchor loss terms"})
    # -------------------------------------------------------------

    # -------------------- Adaptive Beta --------------------
    use_adaptive_beta: bool = field(default=False, metadata={"help": "Enable adaptive beta based on reward distribution"})
    adaptive_beta_min: float = field(default=0.01, metadata={"help": "Minimum beta value"})
    adaptive_beta_max: float = field(default=0.2, metadata={"help": "Maximum beta value"})
    adaptive_beta_base: float = field(default=0.1, metadata={"help": "Base beta value for mean preservation"})
    adaptive_beta_p25_percentile: float = field(default=25.0, metadata={"help": "25th percentile for gap distribution"})
    adaptive_beta_p75_percentile: float = field(default=75.0, metadata={"help": "75th percentile for gap distribution"})
    adaptive_beta_outlier_percentile: float = field(default=99.5, metadata={"help": "Outlier filtering percentile"})

    # -------------------- Dynamic Beta (online) --------------------
    use_dynamic_beta: bool = field(default=False, metadata={"help": "Enable online dynamic beta computed per step"})
    dynamic_beta_temp: float = field(default=1.0, metadata={"help": "Temperature scaling τ for sigmoid mapping"})
    dynamic_beta_lambda1: float = field(default=1.0, metadata={"help": "Weight λ1 for visual gap term 1 (chosen vs ref)"})
    dynamic_beta_lambda2: float = field(default=1.0, metadata={"help": "Weight λ2 for visual gap term 2 (diffusion vs uncond_ref)"})
    dynamic_beta_alpha: float = field(default=1.0, metadata={"help": "Blending weight α when alpha_schedule='const' (0=static only,1=dynamic only)"})
    dynamic_beta_alpha_schedule: str = field(default='const', metadata={"help": "Alpha schedule type: const | lin"})
    dynamic_beta_switch_ratio: float = field(default=0.3, metadata={"help": "For lin schedule, fraction of total steps when α reaches 1"})
    # -------------------------------------------------------------
    # -------------------------------------------------------------


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx + 2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(
        sources: Sequence[str],
        data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN,
                                                                  '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources


def preprocess_llama_2(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack(
            [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v1(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack(
            [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack(
            [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])]  # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx + 2]))  # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            if i != 0 and getattr(tokenizer, 'legacy', False) and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len += 1
                instruction_len += 1

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
        sources: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(
        sources: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
        has_image: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer, has_image=has_image)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)

    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        if data_path.endswith('.json'):
            raw_data = json.load(open(data_path, "r"))
            # Check if data needs conversion (has 'text' field instead of 'conversations')
            # This format is used by DPO training and should be supported for SFT too
            if raw_data and isinstance(raw_data[0], dict) and 'text' in raw_data[0] and 'conversations' not in raw_data[0]:
                # Convert from DPO format to SFT format (same as parquet handling)
                list_data_dict = []
                for sample in raw_data:
                    text = json.loads(sample['text'])
                    question = {'from': 'human', 'value': f"<image>\n{text['question']}"}
                    chosen = {'from': 'gpt', 'value': text['chosen']}
                    source = {
                        "id": sample.get('idx', len(list_data_dict)),
                        'conversations': [question, chosen]
                    }
                    # Handle image field: can be dict with 'path' or string
                    if isinstance(sample.get('image'), dict):
                        source['image'] = sample['image'].get('path', '')
                    elif isinstance(sample.get('image'), str):
                        source['image'] = sample['image']
                    elif 'image' in sample:
                        source['image'] = sample['image']
                    list_data_dict.append(source)
            else:
                list_data_dict = raw_data
        elif data_path.endswith('.parquet'):
            list_data_dict = []
            for sample in pd.read_parquet(data_path).to_dict(orient='records'):
                text = json.loads(sample['text'])
                question = {'from': 'human', 'value': f"<image>\n{text['question']}"}
                chosen = {'from': 'gpt', 'value': text['chosen']}
                source = {
                    "id": sample['idx'],
                    'conversations': [
                        question, chosen
                    ]
                }
                if 'bytes' in sample['image']:
                    img_io = io.BytesIO(sample['image']['bytes'])
                    img_io.seek(0)
                    source['image'] = img_io
                if 'weight' in sample:
                    source['weight'] = float(sample['weight'])
                else:
                    source['weight'] = 1.0
                list_data_dict.append(source)
        rank0_print("Formatting inputs...Skip in lazy mode ")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            processor = self.data_args.image_processor
            if isinstance(image_file, str):
                image_folder = self.data_args.image_folder
                image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            else:
                image = Image.open(sources[0]['image']).convert('RGB')
            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result

                image = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            else:
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]))
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        
        if 'weight' in self.list_data_dict[i]:
            data_dict['weight'] = torch.tensor(self.list_data_dict[i]['weight'], dtype=torch.float32)
        else:
            data_dict['weight'] = torch.tensor(1.0, dtype=torch.float32)
        
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        
        weights = [instance.get('weight', torch.tensor(1.0)) for instance in instances]
        if isinstance(weights[0], torch.Tensor):
            weights = torch.stack(weights)
        else:
            weights = torch.tensor(weights, dtype=torch.float32)
        
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            weights=weights,
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                          data_path=data_args.data_path,
                                          data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def add_diffusion_noise(image_tensor, noise_step):
    """Add diffusion noise to image for generating perturbed image (Section 4.4)."""
    num_steps = 1000
    betas = torch.linspace(-6, 6, num_steps)
    betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5
    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, dim=0)
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

    def q_x(x_0, t):
        noise = torch.randn_like(x_0)
        return alphas_bar_sqrt[t] * x_0 + one_minus_alphas_bar_sqrt[t] * noise

    return q_x(image_tensor.clone(), int(noise_step))


class DPODataset(Dataset):
    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments,
                 training_args=None):
        super(DPODataset, self).__init__()

        if data_path.endswith('.json'):
            list_data_dict = json.load(open(data_path, "r"))
        elif data_path.endswith('.parquet'):
            list_data_dict = pd.read_parquet(data_path).to_dict(orient='records')
        rank0_print(f"Formatting dpo inputs...Skip in lazy mode.")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.training_args = training_args
        self.ref_model_logits = {}
        self.adaptive_beta_cache = None

        # Compute adaptive beta if enabled
        if training_args and getattr(training_args, 'use_adaptive_beta', False):
            self._compute_adaptive_beta()

        if self.data_args.offline_ref_logits and os.path.exists(self.data_args.offline_ref_logits):
            df = pd.read_parquet(self.data_args.offline_ref_logits)
            assert len(df) == len(self.list_data_dict)
            for row in df.itertuples(index=False):
                self.ref_model_logits[row.idx] = row.offline_ref_logits

    def _compute_adaptive_beta(self):
        """Compute adaptive beta values for all samples based on gap (reward difference) distribution.
        
        According to DPO theory:
        - Large gap (easy samples) -> Large beta (conservative, prevent overfitting)
        - Small gap (hard samples) -> Small beta (aggressive, allow exploration)
        """
        import numpy as np
        
        rank0_print("Computing adaptive beta values based on gap distribution...")
        
        # Extract gaps (reward differences) from org_infos
        gaps = []
        for sample in self.list_data_dict:
            if isinstance(sample, dict) and 'text' in sample:
                try:
                    text_data = json.loads(sample['text'])
                    if 'org_infos' in text_data:
                        chosen_reward = text_data['org_infos']['scores']['chosen']['reward']
                        rejected_reward = text_data['org_infos']['scores']['rejected']['reward']
                        gap = chosen_reward - rejected_reward  # This is the gap
                        gaps.append(gap)
                    else:
                        gaps.append(0.0)  # Default for samples without org_infos
                except (json.JSONDecodeError, KeyError):
                    gaps.append(0.0)
            else:
                gaps.append(0.0)
        
        gaps = np.array(gaps)
        
        # Filter outliers
        outlier_threshold = np.percentile(gaps, self.training_args.adaptive_beta_outlier_percentile)
        valid_mask = gaps <= outlier_threshold
        valid_gaps = gaps[valid_mask]
        
        rank0_print(f"Original samples: {len(gaps)}, After outlier filtering: {len(valid_gaps)}")
        
        # Compute percentiles
        p25 = np.percentile(valid_gaps, self.training_args.adaptive_beta_p25_percentile)
        p75 = np.percentile(valid_gaps, self.training_args.adaptive_beta_p75_percentile)
        gap_max = valid_gaps.max()
        gap_min = valid_gaps.min()
        
        rank0_print(f"Gap stats - Min: {gap_min:.6f}, Max: {gap_max:.6f}, P25: {p25:.6f}, P75: {p75:.6f}")
        
        # Compute beta for each sample using piecewise mapping
        adaptive_betas = []
        for gap in gaps:
            if gap <= outlier_threshold:  # Non-outlier
                beta = self._compute_piecewise_beta(gap, p25, p75, gap_max, gap_min)
            else:
                beta = self.training_args.adaptive_beta_base  # Use base beta for outliers
            
            adaptive_betas.append(beta)
        
        self.adaptive_beta_cache = adaptive_betas
        
        # Log beta distribution
        adaptive_betas_array = np.array(adaptive_betas)
        rank0_print(f"Adaptive beta stats - Min: {adaptive_betas_array.min():.3f}, Max: {adaptive_betas_array.max():.3f}, Mean: {adaptive_betas_array.mean():.3f}")
        
        # Count samples in each beta range
        low_beta = np.sum(adaptive_betas_array < 0.08)
        mid_beta = np.sum((adaptive_betas_array >= 0.08) & (adaptive_betas_array < 0.12))
        high_beta = np.sum(adaptive_betas_array >= 0.12)
        rank0_print(f"Beta distribution - Low (<0.08): {low_beta}, Mid (0.08-0.12): {mid_beta}, High (>=0.12): {high_beta}")

    def _compute_piecewise_beta(self, gap, p25_gap, p75_gap, gap_max, gap_min):
        """Compute beta using piecewise mapping strategy based on gap.
        
        According to DPO theory:
        - Large gap (easy samples) -> Large beta (conservative, prevent overfitting)
        - Small gap (hard samples) -> Small beta (aggressive, allow exploration)
        """
        beta_min = self.training_args.adaptive_beta_min
        beta_max = self.training_args.adaptive_beta_max
        base_beta = self.training_args.adaptive_beta_base
        
        if gap <= p25_gap:
            # Small gap (hard samples) -> Small beta (aggressive)
            # Map from [gap_min, p25_gap] to [beta_min, base_beta]
            if p25_gap > gap_min:
                return beta_min + ((gap - gap_min) / (p25_gap - gap_min)) * (base_beta - beta_min)
            else:
                return base_beta
        elif gap >= p75_gap:
            # Large gap (easy samples) -> Large beta (conservative)
            # Map from [p75_gap, gap_max] to [base_beta, beta_max]
            if gap_max > p75_gap:
                return base_beta + ((gap - p75_gap) / (gap_max - p75_gap)) * (beta_max - base_beta)
            else:
                return beta_max  # All large gaps get max beta
        else:
            # Medium gap -> Medium beta
            # Map from [p25_gap, p75_gap] to [base_beta, 0.12]
            medium_beta_max = 0.12  # Medium beta upper bound
            return base_beta + ((gap - p25_gap) / (p75_gap - p25_gap)) * (medium_beta_max - base_beta)

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            text = json.loads(sample['text'])
            cur_len = max(len(text['chosen'].split()), len(text['rejected'].split()))
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i):
        sample: dict = self.list_data_dict[i]
        random_sample: dict = self.list_data_dict[random.choice(range(len(self.list_data_dict)))]

        text = json.loads(sample['text'])
        question = {'from': 'human', 'value': f"<image>\n{text['question']}"}
        chosen = {'from': 'gpt', 'value': text['chosen']}
        rejected = {'from': 'gpt', 'value': text['rejected']}
        source = {
            'image': sample['image']['path'],
            'random_image': random_sample['image']['path'],
            "question": question,
            "chosen": chosen,
            "rejected": rejected,
            "idx": sample['idx'],
            "data_source": sample['ds_name'],
        }
        # Prefer in-memory bytes only when present and valid; otherwise keep file paths
        if isinstance(sample.get('image', {}), dict):
            _img_bytes = sample['image'].get('bytes', None)
            if isinstance(_img_bytes, (bytes, bytearray)) and len(_img_bytes) > 0:
                img_io = io.BytesIO(_img_bytes)
                img_io.seek(0)
                source['image'] = img_io
        if isinstance(random_sample.get('image', {}), dict):
            _rand_bytes = random_sample['image'].get('bytes', None)
            if isinstance(_rand_bytes, (bytes, bytearray)) and len(_rand_bytes) > 0:
                random_img_io = io.BytesIO(_rand_bytes)
                random_img_io.seek(0)
                source['random_image'] = random_img_io

        win_conv = copy.deepcopy([source['question'], source["chosen"]])
        rej_conv = copy.deepcopy([source['question'], source["rejected"]])

        if 'image' in source:
            processor = self.data_args.image_processor
            image = Image.open(source['image']).convert('RGB')
            if self.data_args.use_image_type == 'crop':
                height, width = image.size
                resize_height = min(height, 240)
                resize_width = min(width, 320)
                # RandomCrop
                RandomCrop = transforms.RandomCrop(size=(resize_width, resize_height))
                crop_image = RandomCrop(image)
            if self.data_args.use_image_type == 'rotate':
                RR = transforms.RandomRotation(degrees=(10, 80))
                rotate_image = RR(image)
            if self.data_args.use_image_type == 'random':
                random_image = Image.open(source['random_image']).convert('RGB')
            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result

                image = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
                if self.data_args.use_image_type == 'crop':
                    crop_image = expand2square(crop_image, tuple(int(x * 255) for x in processor.image_mean))
                if self.data_args.use_image_type == 'rotate':
                    rotate_image = expand2square(rotate_image, tuple(int(x * 255) for x in processor.image_mean))
                if self.data_args.use_image_type == 'random':
                    random_image = expand2square(random_image, tuple(int(x * 255) for x in processor.image_mean))
            image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            
            # Prefer a sample-provided negative/contrast image as the second view when available
            diffusion_image = None
            if 'neg_image' in sample and isinstance(sample['neg_image'], str) and os.path.exists(sample['neg_image']):
                neg_image = Image.open(sample['neg_image']).convert('RGB')
                if self.data_args.image_aspect_ratio == 'pad':
                    def expand2square(pil_img, background_color):
                        width, height = pil_img.size
                        if width == height:
                            return pil_img
                        elif width > height:
                            result = Image.new(pil_img.mode, (width, width), background_color)
                            result.paste(pil_img, (0, (width - height) // 2))
                            return result
                        else:
                            result = Image.new(pil_img.mode, (height, height), background_color)
                            result.paste(pil_img, ((height - width) // 2, 0))
                            return result
                
                neg_image = expand2square(neg_image, tuple(int(x * 255) for x in processor.image_mean))
                diffusion_image = processor.preprocess(neg_image, return_tensors='pt')['pixel_values'][0]
            
            # Fallback to the original augmentation-based second view if not provided by sample
            if diffusion_image is None:
                if self.data_args.use_image_type == 'diffusion':
                    diffusion_image = add_diffusion_noise(image, self.data_args.diffusion_step)
                elif self.data_args.use_image_type == 'black':
                    diffusion_image = torch.zeros_like(image)
                elif self.data_args.use_image_type == 'crop':
                    diffusion_image = processor.preprocess(crop_image, return_tensors='pt')['pixel_values'][0]
                elif self.data_args.use_image_type == 'rotate':
                    diffusion_image = processor.preprocess(rotate_image, return_tensors='pt')['pixel_values'][0]
                elif self.data_args.use_image_type == 'random':
                    diffusion_image = processor.preprocess(random_image, return_tensors='pt')['pixel_values'][0]
            
            win_conv = preprocess_multimodal(
                [win_conv],
                self.data_args)[0]
            rej_conv = preprocess_multimodal(
                [rej_conv],
                self.data_args)[0]
        rej_data_dict = preprocess([rej_conv], self.tokenizer, has_image='image' in source)
        rej_data_dict = dict(input_ids=rej_data_dict["input_ids"][0],
                             labels=rej_data_dict["labels"][0])

        win_data_dict = preprocess([win_conv], self.tokenizer, has_image='image' in source)
        win_data_dict = dict(input_ids=win_data_dict["input_ids"][0],
                             labels=win_data_dict["labels"][0],
                             idx=sample['idx'])

        # image exist in the data
        if 'image' in source:
            rej_data_dict['image'] = win_data_dict['image'] = image
            rej_data_dict['diffusion_image'] = win_data_dict['diffusion_image'] = diffusion_image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            rej_data_dict['image'] = win_data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])

        # Build no-image sequences for SVCO (always safe; ignored if not used)
        rej_noimg = preprocess([copy.deepcopy([{'from': 'human', 'value': text['question']}, {'from': 'gpt', 'value': text['rejected']}])], self.tokenizer, has_image=False)
        win_noimg = preprocess([copy.deepcopy([{'from': 'human', 'value': text['question']}, {'from': 'gpt', 'value': text['chosen']}])], self.tokenizer, has_image=False)
        rej_data_dict["noimg_input_ids"] = rej_noimg["input_ids"][0]
        rej_data_dict["noimg_labels"] = rej_noimg["labels"][0]
        win_data_dict["noimg_input_ids"] = win_noimg["input_ids"][0]
        win_data_dict["noimg_labels"] = win_noimg["labels"][0]

        if self.data_args.offline_ref_logits and os.path.exists(self.data_args.offline_ref_logits):
            win_data_dict["offline_ref_logits"] = self.ref_model_logits[sample['idx']]

        # Add adaptive beta if enabled
        if self.adaptive_beta_cache is not None:
            adaptive_beta = self.adaptive_beta_cache[i]
            rej_data_dict["adaptive_beta"] = adaptive_beta
            win_data_dict["adaptive_beta"] = adaptive_beta

        # Add sample weight if exists (for weighted loss)
        if 'weight' in sample:
            weight = float(sample['weight'])
            rej_data_dict["weight"] = weight
            win_data_dict["weight"] = weight

        return rej_data_dict, win_data_dict


@dataclass
class DataCollatorForDPODataset(object):
    tokenizer: transformers.PreTrainedTokenizer

    def SFT_collator_fn(self, instances, pad_token_id):
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX)
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(pad_token_id),
        )

        # Optionally collate no-image sequences (for SVCO)
        if 'noimg_input_ids' in instances[0]:
            noimg_input_ids = [instance['noimg_input_ids'] for instance in instances]
            noimg_labels = [instance['noimg_labels'] for instance in instances]
            noimg_input_ids = torch.nn.utils.rnn.pad_sequence(
                noimg_input_ids,
                batch_first=True,
                padding_value=pad_token_id)
            noimg_labels = torch.nn.utils.rnn.pad_sequence(
                noimg_labels,
                batch_first=True,
                padding_value=IGNORE_INDEX)
            batch['noimg_input_ids'] = noimg_input_ids
            batch['noimg_labels'] = noimg_labels
            batch['noimg_attention_mask'] = noimg_input_ids.ne(pad_token_id)

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images
            batch['diffusion_image'] = torch.stack([instance['diffusion_image'] for instance in instances])
        return batch

    def preference_collator_fn(self, instances, pad_token_id):
        rej_instances, win_instances = list(zip(*instances))
        rej_batch = self.SFT_collator_fn(rej_instances, pad_token_id)
        win_batch = self.SFT_collator_fn(win_instances, pad_token_id)

        concatenated_input_ids = concate_pad(win_batch['input_ids'], rej_batch['input_ids'], pad_token_id)
        concatenated_labels = concate_pad(win_batch['labels'], rej_batch['labels'], -100)
        concatenated_attention_mask = concatenated_input_ids.ne(pad_token_id)
        concatenated_input_ids_3 = concate_pad_three(win_batch['input_ids'], rej_batch['input_ids'],
                                                     win_batch['input_ids'], pad_token_id)
        concatenated_labels_3 = concate_pad_three(win_batch['labels'], rej_batch['labels'], win_batch['labels'], -100)
        concatenated_attention_mask_3 = concatenated_input_ids_3.ne(pad_token_id)

        batch = dict(
            concatenated_input_ids=concatenated_input_ids,
            concatenated_labels=concatenated_labels,
            concatenated_attention_mask=concatenated_attention_mask,
            concatenated_input_ids_3=concatenated_input_ids_3,
            concatenated_labels_3=concatenated_labels_3,
            concatenated_attention_mask_3=concatenated_attention_mask_3,
            win_input_ids=win_batch['input_ids'],
            rej_input_ids=rej_batch['input_ids'],
            images=win_batch['images'],
            diffusion_image=win_batch['diffusion_image'],
            idx=win_instances[0]['idx']
        )
        if 'offline_ref_logits' in win_instances[0]:
            batch["offline_ref_logits"] = win_instances[0]['offline_ref_logits']

        # Handle adaptive beta if present
        if 'adaptive_beta' in win_instances[0]:
            batch["adaptive_beta"] = torch.tensor([win_instances[0]['adaptive_beta'], rej_instances[0]['adaptive_beta']])

        # Handle sample weight if present (for weighted loss)
        if 'weight' in win_instances[0]:
            weights = [inst['weight'] for inst in win_instances]
            batch["weights"] = torch.tensor(weights, dtype=torch.float32)

        # Build SVCO batches when enabled by args later (fields are optional and harmless if unused)
        def _pad_to_length(tensors: List[torch.Tensor], pad_value: int) -> torch.Tensor:
            if len(tensors) == 0:
                return None
            max_len = max(t.shape[1] for t in tensors)
            out_list = []
            for t in tensors:
                if t.shape[1] == max_len:
                    out_list.append(t)
                    continue
                pad_len = max_len - t.shape[1]
                pad_tensor = torch.full((t.shape[0], pad_len), pad_value, dtype=t.dtype, device=t.device)
                out_list.append(torch.cat([t, pad_tensor], dim=1))
            return torch.cat(out_list, dim=0)

        # Only construct SVCO tensors if both no-image inputs exist (robust to legacy datasets)
        if ('noimg_input_ids' in win_batch) and ('noimg_input_ids' in rej_batch):
            # has-image 4-way: [img_win|res_win, img_lose|res_win, img_win|res_lose, img_lose|res_lose]
            svco_hasimg_input_ids = _pad_to_length([
                win_batch['input_ids'],
                win_batch['input_ids'],
                rej_batch['input_ids'],
                rej_batch['input_ids'],
            ], pad_token_id)
            svco_hasimg_labels = _pad_to_length([
                win_batch['labels'],
                win_batch['labels'],
                rej_batch['labels'],
                rej_batch['labels'],
            ], -100)
            svco_hasimg_attention_mask = svco_hasimg_input_ids.ne(pad_token_id)

            # images order must align with input_ids order above
            svco_hasimg_images = torch.cat([
                win_batch['images'],
                win_batch['diffusion_image'],
                win_batch['images'],
                win_batch['diffusion_image'],
            ], dim=0)

            # no-image 2-way: [no_img|res_win, no_img|res_lose]
            svco_noimg_input_ids = _pad_to_length([
                win_batch['noimg_input_ids'],
                rej_batch['noimg_input_ids'],
            ], pad_token_id)
            svco_noimg_labels = _pad_to_length([
                win_batch['noimg_labels'],
                rej_batch['noimg_labels'],
            ], -100)
            svco_noimg_attention_mask = svco_noimg_input_ids.ne(pad_token_id)
            zeros_like_images = torch.zeros_like(win_batch['images'])
            svco_noimg_images = torch.cat([
                zeros_like_images,
                zeros_like_images,
            ], dim=0)

            batch.update({
                'svco_hasimg_input_ids': svco_hasimg_input_ids,
                'svco_hasimg_labels': svco_hasimg_labels,
                'svco_hasimg_attention_mask': svco_hasimg_attention_mask,
                'svco_hasimg_images': svco_hasimg_images,
                'svco_noimg_input_ids': svco_noimg_input_ids,
                'svco_noimg_labels': svco_noimg_labels,
                'svco_noimg_attention_mask': svco_noimg_attention_mask,
                'svco_noimg_images': svco_noimg_images,
            })

        return batch

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        batch = self.preference_collator_fn(instances, self.tokenizer.pad_token_id)
        return batch


def get_offline_ref_logits(model, dataloader):
    ref_logp_list = []

    with torch.inference_mode():
        for batch in tqdm.tqdm(dataloader):
            ref_output, new_labels = model(
                input_ids=batch['concatenated_input_ids'].cuda(),
                labels=batch['concatenated_labels'].cuda(),
                attention_mask=batch['concatenated_attention_mask'].cuda(),
                images=torch.cat([batch['images'], batch['images']], dim=0).to(dtype=torch.bfloat16).cuda(),
                return_new_labels=True
            )
            ref_logps, _ = get_batch_logps(ref_output.logits, new_labels)
            ref_logp_list.append({'idx': batch['idx'], 'offline_ref_logits': ref_logps.tolist()})
    return ref_logp_list,


def make_dpo_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args, training_args=None) -> Dict:
    train_dataset = DPODataset(tokenizer=tokenizer,
                               data_path=data_args.data_path,
                               data_args=data_args,
                               training_args=training_args)
    print(f'Train data size is {len(train_dataset)}', flush=True)
    data_collator = DataCollatorForDPODataset(
        tokenizer=tokenizer,
    )
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size,
                                                      self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


def get_ref_model(model_name_or_path, training_args, model_args):
    from transformers import AutoTokenizer

    rank0_print('init ref_model')

    dtype = torch.bfloat16 if training_args.bf16 else torch.float16
    ref_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
    ref_model = LlavaLlamaForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        # torch_dtype=dtype,
    ).to(device=training_args.device)

    mm_use_im_start_end = getattr(ref_model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(ref_model.config, "mm_use_im_patch_token", True)
    if mm_use_im_patch_token:
        ref_tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        ref_tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    ref_model.resize_token_embeddings(len(ref_tokenizer))
    ref_model.model.initialize_vision_modules(
        model_args=model_args,
        fsdp=training_args.fsdp
    )
    vision_tower = ref_model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(dtype=dtype, device=training_args.device)
    parameter_names = [n for n, _ in ref_model.named_parameters()]
    for param_name in parameter_names:
        param = ref_model.get_parameter(param_name)
        param.requires_grad = False
    ref_model = ref_model.eval()

    rank0_print('init ref_model done')

    return ref_model


def train(attn_implementation=None):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type
            )
        ))

    if model_args.vision_tower is not None:
        if 'mpt' in model_args.model_name_or_path:
            config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
            config.attn_config['attn_impl'] = training_args.mpt_attn_impl
            model = LlavaMptForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
        else:
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                **bnb_model_from_pretrained_args
            )
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            **bnb_model_from_pretrained_args
        )
    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype = (
            torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    if 'mpt' in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right"
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )

        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    if training_args.task == 'DPO':
        # Ensure evaluation uses custom compute_loss (detect labels via this key)
        try:
            training_args.label_names = ["concatenated_labels_3"]
        except Exception:
            pass
        if data_args.offline_ref_logits:
            # get ref logits
            if not os.path.exists(data_args.offline_ref_logits):
                data_module = make_dpo_data_module(tokenizer=tokenizer,
                                                   data_args=data_args)

                ref_model = get_ref_model(model_args.model_name_or_path, training_args, model_args)
                dataloader = DataLoader(
                    data_module['train_dataset'], batch_size=1, shuffle=False,
                    collate_fn=data_module['data_collator'], num_workers=5,
                    sampler=InferenceSampler(len(data_module['train_dataset'])))
                outputs = get_offline_ref_logits(ref_model, dataloader)

                world_size = torch.distributed.get_world_size()
                merged_outputs = [[None for _ in range(world_size)] for i in range(len(outputs))]
                for i in range(len(outputs)):
                    torch.distributed.all_gather_object(merged_outputs[i], outputs[i])
                    merged_outputs[i] = [_ for _ in itertools.chain.from_iterable(merged_outputs[i])]

                ref_logp_list = merged_outputs[0]
                df = pd.DataFrame(ref_logp_list)
                if torch.distributed.get_rank() == 0:
                    df.to_parquet(data_args.offline_ref_logits)

                torch.distributed.barrier()
                del ref_model
                del data_module
            ref_model = None
        else:
            ref_model = get_ref_model(model_args.model_name_or_path, training_args, model_args)
        data_module = make_dpo_data_module(tokenizer=tokenizer, data_args=data_args, training_args=training_args)
        # ===== New: optional eval dataset for DPO =====
        if getattr(data_args, 'eval_data_path', None):
            try:
                eval_dataset = DPODataset(tokenizer=tokenizer,
                                          data_path=data_args.eval_data_path,
                                          data_args=data_args)
                data_module['eval_dataset'] = eval_dataset
                if torch.distributed.get_rank() == 0:
                    print(f"[INFO] Loaded eval_dataset for DPO: {len(eval_dataset)} samples from {data_args.eval_data_path}")
            except Exception as e:
                if torch.distributed.get_rank() == 0:
                    print(f"[WARN] Failed to load eval_data_path for DPO ({data_args.eval_data_path}): {e}")

        trainer = LLAVADPOTrainer(ref_model=ref_model,
                                  model=model,
                                  tokenizer=tokenizer,
                                  args=training_args,
                                  **data_module)
    else:
        data_module = make_supervised_data_module(tokenizer=tokenizer,
                                                  data_args=data_args)
        # ===== New: optional eval dataset for SFT =====
        if getattr(data_args, 'eval_data_path', None):
            try:
                eval_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                                     data_path=data_args.eval_data_path,
                                                     data_args=data_args)
                data_module['eval_dataset'] = eval_dataset
                if torch.distributed.get_rank() == 0:
                    print(f"[INFO] Loaded eval_dataset for SFT: {len(eval_dataset)} samples from {data_args.eval_data_path}")
            except Exception as e:
                if torch.distributed.get_rank() == 0:
                    print(f"[WARN] Failed to load eval_data_path for SFT ({data_args.eval_data_path}): {e}")

        trainer = LLaVATrainer(model=model,
                               tokenizer=tokenizer,
                               args=training_args,
                               **data_module)

    trainer.train()
    
    # Ensure output directory exists before saving
    if training_args.local_rank == 0 or training_args.local_rank == -1:
        os.makedirs(training_args.output_dir, exist_ok=True)
    
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
