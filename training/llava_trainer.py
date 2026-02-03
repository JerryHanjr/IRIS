import os
import math
import torch
import deepspeed
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
from torch.utils.data import Sampler

from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    logger,
)
from typing import Optional, Dict, List, Union, Tuple
from llava.train.diff_lib import get_diff_ids


def concate_pad(tensorA, tensorB, padding_value):
    out = torch.nn.utils.rnn.pad_sequence(
        list(tensorA) + list(tensorB),
        batch_first=True,
        padding_value=padding_value)
    return out

def concate_pad_three(tensorA, tensorB, tensorC, padding_value):
    out = torch.nn.utils.rnn.pad_sequence(
        list(tensorA) + list(tensorB) + list(tensorC),
        batch_first=True,
        padding_value=padding_value)
    return out

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


class LLaVATrainer(Trainer):

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if self.args.mm_projector_lr is not None:
                projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")
        
            if torch.distributed.get_rank() == 0:
                # print(f'LR schduler is ', str(self.scheduler))
                print(f'optimizer: ', str(self.optimizer))
                print('optimizer_cls: ', optimizer_cls)
                print('optimizer_kwargs: ', optimizer_kwargs)
                print('accelerator.state: ', self.accelerator.state)
                print('self.is_deepspeed_enabled:', self.is_deepspeed_enabled)
                print('self.is_fsdp_enabled:', self.is_fsdp_enabled)

        return self.optimizer

    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ['mm_projector', 'vision_resampler']
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        else:
            super(LLaVATrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            super(LLaVATrainer, self)._save(output_dir, state_dict)

    def compute_loss(self, model, inputs, return_outputs=False):
        """Override compute_loss to support sample-level weighting."""
        weights = inputs.pop('weights', None)
        
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        
        outputs = model(**inputs)
        
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            logits = outputs.get("logits")
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss.view(shift_labels.size())
            
            from llava.constants import IGNORE_INDEX
            mask = (shift_labels != IGNORE_INDEX).float()
            loss = (loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            
            if weights is not None:
                if weights.device != loss.device:
                    weights = weights.to(loss.device)
                loss = loss * weights
            
            loss = loss.mean()
        else:
            loss = outputs.get("loss")
        
        return (loss, outputs) if return_outputs else loss

def chip_get_batch_logps(logits: torch.FloatTensor,
                        reference_logits: torch.FloatTensor,
                        uncond_ref_logits: torch.FloatTensor,
                        labels: torch.LongTensor,
                        average_log_prob: bool = False):
    """Compute the kl divergence/log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        reference_logits: Logits of the reference model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        uncond_ref_logits: Logits of the reference model (unconditional unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

    Returns:
        Several tensors of shape (batch_size,) containing the average/sum kl divergence/log probabilities of the given labels under the given logits.
    """
    # Fix: dynamically compute batch size based on reference_logits (supports batch_size > 1)
    ref_batch = reference_logits.shape[0]
    labels = labels[:ref_batch, :].clone()
    logits = logits[:ref_batch, :, :]
    assert logits.shape[:-1] == labels.shape, (logits.shape[:-1], labels.shape)
    assert reference_logits.shape[:-1] == labels.shape, (reference_logits.shape[:-1], labels.shape)
    assert uncond_ref_logits.shape[:-1] == labels.shape, (uncond_ref_logits.shape[:-1], labels.shape)

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    reference_logits = reference_logits[:, :-1, :]
    uncond_ref_logits = uncond_ref_logits[:, :-1, :]

    loss_mask = (labels != -100)

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == -100] = 0

    vocab_logps = logits.log_softmax(-1)

    reference_vocab_ps = reference_logits.softmax(-1)
    reference_vocab_logps = reference_vocab_ps.log()

    uncond_ref_vocab_logps = uncond_ref_logits.log_softmax(-1)

    per_position_kl = (reference_vocab_ps * (reference_vocab_logps - vocab_logps)).sum(-1)
    per_policy_token_logps = torch.gather(vocab_logps, dim=2, index=labels.unsqueeze(2)).squeeze(2)
    per_reference_token_logps = torch.gather(reference_vocab_logps, dim=2, index=labels.unsqueeze(2)).squeeze(2)
    per_uncond_ref_token_logps = torch.gather(uncond_ref_vocab_logps, dim=2, index=labels.unsqueeze(2)).squeeze(2)

    if average_log_prob:
        return (per_position_kl * loss_mask).sum(-1) / loss_mask.sum(-1), \
                (per_policy_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1), \
                (per_reference_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1), \
                (per_uncond_ref_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1), \
                per_policy_token_logps, per_reference_token_logps, per_uncond_ref_token_logps
    else:
        return (per_position_kl * loss_mask).sum(-1), \
            (per_policy_token_logps * loss_mask).sum(-1), \
            (per_reference_token_logps * loss_mask).sum(-1), \
            (per_uncond_ref_token_logps * loss_mask).sum(-1), \
            per_policy_token_logps, per_reference_token_logps, per_uncond_ref_token_logps
    
def get_batch_logps(logits: torch.FloatTensor, labels: torch.LongTensor, return_per_token_logp=False, return_all=False) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    """
    assert logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = (labels != -100)

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == -100] = 0

    per_policy_token_logps = torch.gather(logits.log_softmax(-1), dim=2,
                                   index=labels.unsqueeze(2)).squeeze(2)

    log_prob = (per_policy_token_logps * loss_mask).sum(-1)
    average_log_prob = log_prob / loss_mask.sum(-1)

    if return_per_token_logp:
        return per_policy_token_logps

    if return_all:
        return per_policy_token_logps, log_prob, average_log_prob

    return log_prob, average_log_prob

def get_eval_ds_config(offload=None, stage=3):
    from accelerate.state import AcceleratorState

    deepspeed_states = AcceleratorState().deepspeed_plugin

    device = "cpu" if offload else "none"
    zero_opt_dict = {
        "stage": stage,
        "stage3_param_persistence_threshold": 1e4,
        "offload_param": {
            "device": device
        }
    }
    return {
        "train_micro_batch_size_per_gpu": deepspeed_states.deepspeed_config['train_micro_batch_size_per_gpu'],
        "steps_per_print": 10,
        "zero_optimization": zero_opt_dict,
        "bf16": {
            "enabled": True
        },
        "gradient_clipping": 20.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False
    }


class LLAVADPOTrainer(LLaVATrainer):
    def __init__(self, ref_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ref_model = ref_model
        if torch.distributed.get_rank() == 0:
            print('self.args:', self.args)
        if self.ref_model is not None and 'zero3' in self.args.deepspeed:
            eval_ds_config = get_eval_ds_config(offload=False)
            self.ref_model, *_ = deepspeed.initialize(model=self.ref_model, config=eval_ds_config)
            self.ref_model.eval()
            print('ref_model deepspeed init done!')

    def chip_loss(self, policy_chosen_logp: torch.FloatTensor,
                    policy_rejected_logp: torch.FloatTensor,
                    policy_win_diffusionImage_logp: torch.FloatTensor,
                    reference_chosen_logp: torch.FloatTensor,
                    reference_rejected_logp: torch.FloatTensor,
                    uncond_ref_win_logp: torch.FloatTensor,
                    uncond_ref_rej_logp: torch.FloatTensor,
                    chosen_position_kl: torch.FloatTensor,
                    rejected_position_kl: torch.FloatTensor,
                    beta: float=0.1, gama:float=0.3, adaptive_beta: Optional[float]=None
                    ) -> Tuple[
        torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        IRIS Grounded Preference Learning Loss (Eq. 12-14).
        
        Combines three objectives:
        - Lctp: Conditional Textual Preference (Eq. 12) - standard DPO
        - Lcvp: Conditional Visual Preference (Eq. 13) - cross-modal loss
        - Lanchor: Anchored Regularization (Eq. 14) - anchor loss
        """
        effective_beta = adaptive_beta if adaptive_beta is not None else beta
        
        # === Lctp: Conditional Textual Preference (Eq. 12) ===
        pi_logratios = policy_chosen_logp - policy_rejected_logp
        ref_logratios = reference_chosen_logp - reference_rejected_logp
        logits = pi_logratios - ref_logratios
        
        # === Lcvp: Conditional Visual Preference (Eq. 13) ===
        if self.args.use_cross_modal_loss:
            cross_modal_gama = getattr(self.args, 'use_cross_modal_gama', gama)
            if getattr(self.args, 'use_cross_modal_loss_ref', True):
                logits += cross_modal_gama * (policy_chosen_logp - reference_chosen_logp)
            if getattr(self.args, 'use_cross_modal_loss_vis', True):
                logits -= cross_modal_gama * (policy_win_diffusionImage_logp - uncond_ref_win_logp)        
        if self.args.use_tdpo:
            logits -= self.args.tok_beta * (
                         rejected_position_kl - chosen_position_kl.detach())
            chosen_values = policy_chosen_logp - reference_chosen_logp + chosen_position_kl
            rejected_values = policy_rejected_logp - reference_rejected_logp + rejected_position_kl
        else:
            chosen_values = policy_chosen_logp - reference_chosen_logp
            rejected_values = policy_rejected_logp - reference_rejected_logp
        
        losses = -F.logsigmoid(effective_beta * logits)

        chosen_rewards = effective_beta * chosen_values.detach()
        rejected_rewards = effective_beta * rejected_values.detach()

        return losses, chosen_rewards, rejected_rewards

    def compute_loss(self, model: Module, inputs: dict, return_outputs=False):
        data_dict = inputs

        win_input_ids = data_dict.pop('win_input_ids')
        rej_input_ids = data_dict.pop('rej_input_ids')
        images = data_dict.pop('images')
        diffusion_image = data_dict.pop('diffusion_image', '')
        win_size = win_input_ids.shape[0]
        rej_size = rej_input_ids.shape[0]
        assert win_size == rej_size

        concatenated_input_ids = data_dict.pop('concatenated_input_ids')
        concatenated_labels = data_dict.pop('concatenated_labels')
        concatenated_attention_mask = data_dict.pop('concatenated_attention_mask')
        concatenated_input_ids_3 = data_dict.pop('concatenated_input_ids_3')
        concatenated_labels_3 = data_dict.pop('concatenated_labels_3')
        concatenated_attention_mask_3 = data_dict.pop('concatenated_attention_mask_3')

        ref_logps = data_dict.pop('offline_ref_logits', None)
        if ref_logps is not None:
            ref_logps = torch.as_tensor(ref_logps).cuda()
        idx = data_dict.pop('idx', None)
        output, new_labels = model(
            input_ids=concatenated_input_ids_3,
            labels=concatenated_labels_3,
            attention_mask=concatenated_attention_mask_3,
            images=torch.cat([images, images, diffusion_image], dim=0),
            return_new_labels=True,
            # Do not pass extra keys (e.g., SVCO tensors) to the model forward
        )

        if ref_logps is None:
            with torch.no_grad():
                ref_output = self.ref_model(
                    input_ids=concatenated_input_ids,
                    labels=concatenated_labels,
                    attention_mask=concatenated_attention_mask,
                    images=torch.cat([images, images], dim=0),
                    # Do not pass extra keys
                )

            with torch.no_grad():
                unconditional_ref_output = self.ref_model(
                    input_ids=concatenated_input_ids,
                    labels=concatenated_labels,
                    attention_mask=concatenated_attention_mask,
                    images=torch.cat([diffusion_image, diffusion_image], dim=0),
                    # Do not pass extra keys
                )

            all_position_kl, policy_logps, ref_logps, uncond_ref_logps, \
                per_policy_token_logps, per_reference_token_logps, \
                    per_uncond_ref_token_logps = chip_get_batch_logps(
                output.logits, ref_output.logits, unconditional_ref_output.logits,
                new_labels, average_log_prob=False)
            chosen_position_kl, rejected_position_kl = all_position_kl.split([win_size, rej_size])
            uncond_ref_win_logp, uncond_ref_rej_logp = uncond_ref_logps.split([win_size, rej_size])
        
        # three-tuple logits
        per_policy_token_logps, policy_logps, average_policy_logps = get_batch_logps(output.logits, new_labels, return_all=True)
        reference_chosen_logp, reference_rejected_logp = ref_logps.split([win_size, rej_size])
        
        # per-token split
        policy_win_per_token_logps, policy_rej_per_token_logps, \
            policy_win_diffusionImage_per_token_logps = per_policy_token_logps.split(
                [win_size, rej_size, win_size])
        ref_win_per_token_logps, ref_rej_per_token_logps = per_reference_token_logps.split(
            [win_size, rej_size])

        # input_id / labels split
        win_labels, rej_labels = concatenated_labels.split([win_size, rej_size])
        win_input_ids, rej_input_ids = concatenated_input_ids.split([win_size, rej_size])

        policy_chosen_logp, policy_rejected_logp, policy_win_diffusionImage_logp = policy_logps.split([win_size, rej_size, win_size])

        if self.args.dpo_token_weighted:
            win_token_weight, rej_token_weight = self.get_seg_weight(win_labels, rej_labels, win_input_ids, rej_input_ids)
            use_avg = bool(getattr(self.args, 'dpo_use_average', False))
            uncond_ref_win_logp = self.compute_weighted_logp(uncond_ref_win_logp, win_labels, win_token_weight, use_average=use_avg)
            uncond_ref_rej_logp = self.compute_weighted_logp(uncond_ref_rej_logp, rej_labels, rej_token_weight, use_average=use_avg)
            reference_chosen_logp = self.compute_weighted_logp(ref_win_per_token_logps, win_labels, win_token_weight, use_average=use_avg)
            reference_rejected_logp = self.compute_weighted_logp(ref_rej_per_token_logps, rej_labels, rej_token_weight, use_average=use_avg)
            policy_chosen_logp = self.compute_weighted_logp(policy_win_per_token_logps, win_labels, win_token_weight, use_average=use_avg)
            policy_rejected_logp = self.compute_weighted_logp(policy_rej_per_token_logps, rej_labels, rej_token_weight, use_average=use_avg)
            policy_win_diffusionImage_logp = self.compute_weighted_logp(policy_win_diffusionImage_per_token_logps, win_labels, win_token_weight, use_average=use_avg)

            if torch.any(torch.isnan(uncond_ref_win_logp)):
                print(f'uncond_ref_win_logp fail', flush=True)
                exit()
            if torch.any(torch.isnan(uncond_ref_rej_logp)):
                print(f'uncond_ref_rej_logp fail', flush=True)
                exit()
            if torch.any(torch.isnan(reference_chosen_logp)):
                print(f'reference_chosen_logp fail', flush=True)
                exit()
            if torch.any(torch.isnan(reference_rejected_logp)):
                print(f'reference_rejected_logp fail', flush=True)
                exit()
            if torch.any(torch.isnan(policy_chosen_logp)):
                print(f'policy_chosen_logp fail', flush=True)
                exit()
            if torch.any(torch.isnan(policy_rejected_logp)):
                print(f'policy_rejected_logp fail', flush=True)
                exit()
            if torch.any(torch.isnan(policy_win_diffusionImage_logp)):
                print(f'policy_win_diffusionImage_logp fail', flush=True)
                exit()
        
        
        # -------------------- Static Adaptive Beta (pre-computed) --------------------
        # Get adaptive beta if enabled
        adaptive_beta = None
        if getattr(self.args, 'use_adaptive_beta', False):
            # Extract adaptive beta from batch data
            if 'adaptive_beta' in data_dict:
                # adaptive_beta is a tensor with shape [2] (chosen, rejected) - they should be the same
                adaptive_beta_tensor = data_dict['adaptive_beta']
                # Take the first value (they should be identical for the same sample)
                adaptive_beta = adaptive_beta_tensor[0].item() if adaptive_beta_tensor.numel() > 0 else self.args.adaptive_beta_base
                if torch.distributed.get_rank() == 0 and self.state.global_step % 10 == 0:
                    print(f"Using adaptive beta: {adaptive_beta:.3f}")
            else:
                adaptive_beta = self.args.adaptive_beta_base
                if torch.distributed.get_rank() == 0 and self.state.global_step % 10 == 0:
                    print(f"Adaptive beta not found in batch, using base beta: {adaptive_beta:.3f}")
        
        # -------------------- Online Dynamic Beta (per-step) --------------------
        beta_final = adaptive_beta  # default to static; may be None
        if getattr(self.args, 'use_dynamic_beta', False):
            # compute gaps (all in chosen-rejected form for consistency)
            gap_text = (policy_chosen_logp - policy_rejected_logp) - (reference_chosen_logp - reference_rejected_logp)
            gap_vis1 = (policy_chosen_logp - reference_chosen_logp) - (policy_rejected_logp - reference_rejected_logp)
            gap_vis2 = (policy_win_diffusionImage_logp - uncond_ref_win_logp) - (policy_rejected_logp - uncond_ref_rej_logp)
            gap_mm = gap_text + self.args.dynamic_beta_lambda1 * gap_vis1 + self.args.dynamic_beta_lambda2 * gap_vis2
            
            # Normalize gap_mm for stable sigmoid mapping (handle batch_size=1 case)
            if gap_mm.numel() > 1:
                gap_mm_std = gap_mm.std(unbiased=False)  # Use unbiased=False for single element
                if gap_mm_std > 1e-8:
                    gap_mm_normalized = (gap_mm - gap_mm.mean()) / gap_mm_std
                else:
                    gap_mm_normalized = gap_mm - gap_mm.mean()  # Just center if std is too small
            else:
                # For batch_size=1, just use the gap value directly (no normalization needed)
                gap_mm_normalized = torch.zeros_like(gap_mm)
            
            # sigmoid mapping to [beta_min, beta_max]
            beta_min = self.args.adaptive_beta_min if hasattr(self.args, 'adaptive_beta_min') else 0.01
            beta_max = self.args.adaptive_beta_max if hasattr(self.args, 'adaptive_beta_max') else 0.2
            beta_dyn = torch.sigmoid(self.args.dynamic_beta_temp * gap_mm_normalized.detach()) * (beta_max - beta_min) + beta_min
            
            # compute alpha (schedule from static to dynamic)
            if self.args.dynamic_beta_alpha_schedule == 'lin' and self.state is not None and self.state.max_steps is not None:
                switch_steps = int(self.args.dynamic_beta_switch_ratio * self.state.max_steps)
                cur_step = self.state.global_step
                alpha = min(1.0, cur_step / max(1, switch_steps))
            else:
                alpha = self.args.dynamic_beta_alpha
            
            # Blend static and dynamic beta
            if beta_final is None:
                beta_final = beta_dyn
            else:
                if isinstance(beta_final, torch.Tensor):
                    beta_final = (1 - alpha) * beta_final + alpha * beta_dyn
                else:
                    # beta_final is a scalar
                    beta_final = (1 - alpha) * beta_final + alpha * beta_dyn
            
            # Optional: log beta statistics for monitoring
            if torch.distributed.get_rank() == 0 and self.state.global_step % 10 == 0:
                gap_mm_std_val = gap_mm.std(unbiased=False).item() if gap_mm.numel() > 1 else 0.0
                print(f"[Dynamic Beta] Step {self.state.global_step}: alpha={alpha:.3f}, "
                      f"beta_range=[{beta_final.min().item() if isinstance(beta_final, torch.Tensor) else beta_final:.3f}, "
                      f"{beta_final.max().item() if isinstance(beta_final, torch.Tensor) else beta_final:.3f}], "
                      f"gap_mm_mean={gap_mm.mean().item():.3f}, gap_mm_std={gap_mm_std_val:.3f}")
        # --------------------------------------------------------------

        losses, chosen_rewards, rejected_rewards = self.chip_loss(
            policy_chosen_logp, policy_rejected_logp, policy_win_diffusionImage_logp,
            reference_chosen_logp, reference_rejected_logp,
            uncond_ref_win_logp, uncond_ref_rej_logp,
            chosen_position_kl, rejected_position_kl,
            beta=getattr(self.args, 'dpo_beta', 0.1),
            adaptive_beta=beta_final
            )
        
        # Apply sample weights if present (for weighted loss)
        weights = data_dict.pop('weights', None)
        if weights is not None:
            if weights.device != losses.device:
                weights = weights.to(losses.device)
            losses = losses * weights
        
        chip_loss_value = losses.mean()
        if not getattr(self.args, 'use_chip_loss', True):
            chip_loss_value = torch.zeros_like(chip_loss_value)

        # Initialize optional anchor loss (dual-view, no extra forward)
        anchor_loss_value = torch.tensor(0.0, device=chip_loss_value.device, dtype=chip_loss_value.dtype)
        anchor1_loss_value = torch.tensor(0.0, device=chip_loss_value.device, dtype=chip_loss_value.dtype)
        anchor2_loss_value = torch.tensor(0.0, device=chip_loss_value.device, dtype=chip_loss_value.dtype)
        anchor2_active_flag = 0.0
        # === Lanchor: Anchored Regularization (Eq. 14) ===
        if getattr(self.args, 'use_anchor_loss', False):
            beta1 = float(getattr(self.args, 'anchor_loss_beta', 0.1))
            beta2 = float(getattr(self.args, 'anchor_loss_beta2', beta1))
            delta = float(getattr(self.args, 'anchor_loss_margin_delta', 0.0))
            lam = float(getattr(self.args, 'anchor_loss_lambda', 1.0))

            anchor1_logratios = policy_chosen_logp - reference_chosen_logp
            anchor1 = -F.logsigmoid(beta1 * anchor1_logratios - delta)
            anchor1_loss_value = anchor1.mean()

            if bool(getattr(self.args, 'use_dual_anchor_loss', True)):
                try:
                    # variables computed above in this function
                    _p2 = policy_win_diffusionImage_logp
                    _r2 = uncond_ref_win_logp
                    if (_p2 is not None) and (_r2 is not None) and (_p2.shape == _r2.shape):
                        anchor2_logratios = _p2 - _r2
                        anchor2 = -F.logsigmoid(beta2 * anchor2_logratios - delta)
                        anchor2_loss_value = anchor2.mean()
                        anchor2_active_flag = 1.0
                    else:
                        if torch.distributed.get_rank() == 0:
                            print('[WARN] Dual anchor enabled but second-view logits missing or shape-mismatch; skipping anchor2.', flush=True)
                except NameError:
                    if torch.distributed.get_rank() == 0:
                        print('[WARN] Dual anchor enabled but second-view variables not defined in this batch; skipping anchor2.', flush=True)
                except Exception as e:
                    if torch.distributed.get_rank() == 0:
                        print(f'[WARN] Dual anchor computation failed ({e}); skipping anchor2.', flush=True)

            anchor_loss_value = lam * (anchor1_loss_value + anchor2_loss_value)

        # Initialize optional losses
        svco_loss_value = torch.tensor(0.0, device=chip_loss_value.device, dtype=chip_loss_value.dtype)
        svco_active_flag = 0.0
        new_visual_loss_value = torch.tensor(0.0, device=chip_loss_value.device, dtype=chip_loss_value.dtype)
        new_visual_active_flag = 0.0

        # New visual loss (L_new) based only on has-image pairs
        if getattr(self.args, 'use_new_visual_loss', False) and self.args.new_visual_lambda > 0.0:
            missing_keys_new = []
            for k in [
                'svco_hasimg_input_ids', 'svco_hasimg_labels', 'svco_hasimg_attention_mask', 'svco_hasimg_images',
            ]:
                if k not in data_dict:
                    missing_keys_new.append(k)
            if len(missing_keys_new) == 0:
                svco_hasimg_input_ids = data_dict['svco_hasimg_input_ids']
                svco_hasimg_labels = data_dict['svco_hasimg_labels']
                svco_hasimg_attention_mask = data_dict['svco_hasimg_attention_mask']
                svco_hasimg_images = data_dict['svco_hasimg_images']

                svco_hasimg_out, svco_hasimg_new_labels = model(
                    input_ids=svco_hasimg_input_ids,
                    labels=svco_hasimg_labels,
                    attention_mask=svco_hasimg_attention_mask,
                    images=svco_hasimg_images,
                    return_new_labels=True
                )
                with torch.no_grad():
                    if self.ref_model is None:
                        ref_hasimg_out = svco_hasimg_out
                    else:
                        ref_hasimg_out = self.ref_model(
                            input_ids=svco_hasimg_input_ids,
                            attention_mask=svco_hasimg_attention_mask,
                            images=svco_hasimg_images,
                            return_dict=True
                        )

                def _avg_logps(logits, labels):
                    logps = get_batch_logps(logits, labels)[0]
                    return logps
                bs = win_size
                p_hasimg_logps = _avg_logps(svco_hasimg_out.logits, svco_hasimg_new_labels)
                r_hasimg_logps = _avg_logps(ref_hasimg_out.logits, svco_hasimg_new_labels)

                # y_r slices under has-image 4-way packing
                p_img_win_res_lose = p_hasimg_logps[2*bs: 2*bs + rej_size]
                p_img_lose_res_lose = p_hasimg_logps[2*bs + rej_size:]
                r_img_win_res_lose = r_hasimg_logps[2*bs: 2*bs + rej_size]
                r_img_lose_res_lose = r_hasimg_logps[2*bs + rej_size:]

                z_new = (p_img_lose_res_lose - p_img_win_res_lose) - (r_img_lose_res_lose - r_img_win_res_lose)
                beta_new = float(getattr(self.args, 'new_visual_beta', 0.5))
                l_new = -F.logsigmoid(beta_new * z_new)
                new_visual_loss_value = l_new.mean()
                new_visual_active_flag = 1.0
            else:
                if torch.distributed.get_rank() == 0:
                    print(f"[WARN] New visual loss enabled but missing batch keys: {missing_keys_new}", flush=True)

        # Optionally compute SVCO loss and combine
        if getattr(self.args, 'use_svco_loss', False) and self.args.svco_lambda > 0.0:
            # Guard: required SVCO batch fields must exist
            missing_keys = []
            for k in [
                'svco_hasimg_input_ids', 'svco_hasimg_labels', 'svco_hasimg_attention_mask', 'svco_hasimg_images',
                'svco_noimg_input_ids', 'svco_noimg_labels', 'svco_noimg_attention_mask', 'svco_noimg_images',
            ]:
                if k not in data_dict:
                    missing_keys.append(k)
            if len(missing_keys) == 0:
                # Extract local tensors
                svco_hasimg_input_ids = data_dict['svco_hasimg_input_ids']
                svco_hasimg_labels = data_dict['svco_hasimg_labels']
                svco_hasimg_attention_mask = data_dict['svco_hasimg_attention_mask']
                svco_hasimg_images = data_dict['svco_hasimg_images']
                svco_noimg_input_ids = data_dict['svco_noimg_input_ids']
                svco_noimg_labels = data_dict['svco_noimg_labels']
                svco_noimg_attention_mask = data_dict['svco_noimg_attention_mask']
                svco_noimg_images = data_dict['svco_noimg_images']

                # Forward policy (request aligned labels)
                svco_hasimg_out, svco_hasimg_new_labels = model(
                    input_ids=svco_hasimg_input_ids,
                    labels=svco_hasimg_labels,
                    attention_mask=svco_hasimg_attention_mask,
                    images=svco_hasimg_images,
                    return_new_labels=True
                )
                svco_noimg_out, svco_noimg_new_labels = model(
                    input_ids=svco_noimg_input_ids,
                    labels=svco_noimg_labels,
                    attention_mask=svco_noimg_attention_mask,
                    images=svco_noimg_images,
                    return_new_labels=True
                )
                # Forward reference
                with torch.no_grad():
                    if self.ref_model is None:
                        ref_hasimg_out = svco_hasimg_out
                        ref_noimg_out = svco_noimg_out
                    else:
                        ref_hasimg_out = self.ref_model(
                            input_ids=svco_hasimg_input_ids,
                            attention_mask=svco_hasimg_attention_mask,
                            images=svco_hasimg_images,
                            return_dict=True
                        )
                        ref_noimg_out = self.ref_model(
                            input_ids=svco_noimg_input_ids,
                            attention_mask=svco_noimg_attention_mask,
                            images=svco_noimg_images,
                            return_dict=True
                        )
                # Compute logps for 4 has-img slices and 2 no-img slices
                def _avg_logps(logits, labels):
                    logps = get_batch_logps(logits, labels)[0]
                    return logps
                bs = win_size  # chosen batch size
                # policy
                p_hasimg_logps = _avg_logps(svco_hasimg_out.logits, svco_hasimg_new_labels)
                p_noimg_logps  = _avg_logps(svco_noimg_out.logits,  svco_noimg_new_labels)
                p_img_win_res_win      = p_hasimg_logps[:bs]
                p_img_lose_res_win     = p_hasimg_logps[bs: 2*bs]
                p_img_win_res_lose     = p_hasimg_logps[2*bs: 2*bs + rej_size]
                p_img_lose_res_lose    = p_hasimg_logps[2*bs + rej_size:]
                p_noimg_res_win        = p_noimg_logps[:bs]
                p_noimg_res_lose       = p_noimg_logps[bs:]
                # reference
                r_hasimg_logps = _avg_logps(ref_hasimg_out.logits, svco_hasimg_new_labels)
                r_noimg_logps  = _avg_logps(ref_noimg_out.logits,  svco_noimg_new_labels)
                r_img_win_res_win      = r_hasimg_logps[:bs]
                r_img_lose_res_win     = r_hasimg_logps[bs: 2*bs]
                r_img_win_res_lose     = r_hasimg_logps[2*bs: 2*bs + rej_size]
                r_img_lose_res_lose    = r_hasimg_logps[2*bs + rej_size:]
                r_noimg_res_win        = r_noimg_logps[:bs]
                r_noimg_res_lose       = r_noimg_logps[bs:]

                # Build four logits per S-VCO
                ref_free = bool(getattr(self.args, 'svco_reference_free', False))
                # 1) (res_win|img_win) > (res_win|no_img)
                logits_img_win_vs_no_img = (p_img_win_res_win - p_noimg_res_win) - (0 if ref_free else (r_img_win_res_win - r_noimg_res_win))
                # 2) (res_win|no_img) > (res_win|img_lose)
                logits_no_img_vs_img_lose = (p_noimg_res_win - p_img_lose_res_win) - (0 if ref_free else (r_noimg_res_win - r_img_lose_res_win))
                # 3) (res_lose|img_lose) > (res_lose|no_img)
                logits_img_lose_vs_no_img = (p_img_lose_res_lose - p_noimg_res_lose) - (0 if ref_free else (r_img_lose_res_lose - r_noimg_res_lose))
                # 4) (res_lose|no_img) > (res_lose|img_win)
                logits_no_img_vs_img_win = (p_noimg_res_lose - p_img_win_res_lose) - (0 if ref_free else (r_noimg_res_lose - r_img_win_res_lose))

                # losses
                b1 = max(self.args.svco_beta_img_win_vs_no_img, 0.0)
                b2 = max(self.args.svco_beta_no_img_vs_img_lose, 0.0)
                b3 = max(self.args.svco_beta_img_lose_vs_no_img, 0.0)
                b4 = max(self.args.svco_beta_no_img_vs_img_win, 0.0)
                l1 = -F.logsigmoid(b1 * logits_img_win_vs_no_img) if b1 > 0 else torch.zeros_like(logits_img_win_vs_no_img)
                l2 = -F.logsigmoid(b2 * logits_no_img_vs_img_lose) if b2 > 0 else torch.zeros_like(logits_no_img_vs_img_lose)
                l3 = -F.logsigmoid(b3 * logits_img_lose_vs_no_img) if b3 > 0 else torch.zeros_like(logits_img_lose_vs_no_img)
                l4 = -F.logsigmoid(b4 * logits_no_img_vs_img_win) if b4 > 0 else torch.zeros_like(logits_no_img_vs_img_win)
                svco_losses = l1 + l2 + l3 + l4
                svco_loss_value = svco_losses.mean()
                svco_active_flag = 1.0

                # Optional logging for SVCO
                train_test = 'train' if model.training else 'test'
                metrics = {}
                metrics[f'svco_{train_test}/img_win_vs_no_img_loss'] = self._nested_gather(l1.mean()).mean().item()
                metrics[f'svco_{train_test}/no_img_vs_img_lose_loss'] = self._nested_gather(l2.mean()).mean().item()
                metrics[f'svco_{train_test}/img_lose_vs_no_img_loss'] = self._nested_gather(l3.mean()).mean().item()
                metrics[f'svco_{train_test}/no_img_vs_img_win_loss'] = self._nested_gather(l4.mean()).mean().item()
                self.log(metrics)
            else:
                # Missing SVCO tensors; warn once per rank0
                if torch.distributed.get_rank() == 0:
                    print(f"[WARN] SVCO loss enabled but missing batch keys: {missing_keys}", flush=True)

        # Combine losses
        vco_loss_value = torch.tensor(0.0, device=chip_loss_value.device, dtype=chip_loss_value.dtype)
        if getattr(self.args, 'use_vco_loss', False) and self.args.vco_lambda > 0.0:
            # Build minimal tensors (reuse if already computed in SVCO section)
            have_vars = 'p_img_win_res_win' in locals()
            if not have_vars:
                missing_keys_vco = []
                for k in [
                    'svco_hasimg_input_ids', 'svco_hasimg_labels', 'svco_hasimg_attention_mask', 'svco_hasimg_images',
                    'svco_noimg_input_ids', 'svco_noimg_labels', 'svco_noimg_attention_mask', 'svco_noimg_images',
                ]:
                    if k not in data_dict:
                        missing_keys_vco.append(k)
                if len(missing_keys_vco) == 0:
                    svco_hasimg_input_ids = data_dict['svco_hasimg_input_ids']
                    svco_hasimg_labels = data_dict['svco_hasimg_labels']
                    svco_hasimg_attention_mask = data_dict['svco_hasimg_attention_mask']
                    svco_hasimg_images = data_dict['svco_hasimg_images']
                    svco_noimg_input_ids = data_dict['svco_noimg_input_ids']
                    svco_noimg_labels = data_dict['svco_noimg_labels']
                    svco_noimg_attention_mask = data_dict['svco_noimg_attention_mask']
                    svco_noimg_images = data_dict['svco_noimg_images']

                    svco_hasimg_out, svco_hasimg_new_labels = model(
                        input_ids=svco_hasimg_input_ids,
                        labels=svco_hasimg_labels,
                        attention_mask=svco_hasimg_attention_mask,
                        images=svco_hasimg_images,
                        return_new_labels=True
                    )
                    with torch.no_grad():
                        if self.ref_model is None:
                            ref_hasimg_out = svco_hasimg_out
                        else:
                            ref_hasimg_out = self.ref_model(
                                input_ids=svco_hasimg_input_ids,
                                attention_mask=svco_hasimg_attention_mask,
                                images=svco_hasimg_images,
                                return_dict=True
                            )
                    svco_noimg_out, svco_noimg_new_labels = model(
                        input_ids=svco_noimg_input_ids,
                        labels=svco_noimg_labels,
                        attention_mask=svco_noimg_attention_mask,
                        images=svco_noimg_images,
                        return_new_labels=True
                    )
                    with torch.no_grad():
                        if self.ref_model is None:
                            ref_noimg_out = svco_noimg_out
                        else:
                            ref_noimg_out = self.ref_model(
                                input_ids=svco_noimg_input_ids,
                                attention_mask=svco_noimg_attention_mask,
                                images=svco_noimg_images,
                                return_dict=True
                            )

                    def _avg_logps(logits, labels):
                        logps = get_batch_logps(logits, labels)[0]
                        return logps
                    bs = win_size
                    p_hasimg_logps = _avg_logps(svco_hasimg_out.logits, svco_hasimg_new_labels)
                    p_noimg_logps  = _avg_logps(svco_noimg_out.logits,  svco_noimg_new_labels)
                    r_hasimg_logps = _avg_logps(ref_hasimg_out.logits, svco_hasimg_new_labels)
                    r_noimg_logps  = _avg_logps(ref_noimg_out.logits,  svco_noimg_new_labels)

                    p_img_win_res_win      = p_hasimg_logps[:bs]
                    p_img_lose_res_win     = p_hasimg_logps[bs: 2*bs]
                    p_noimg_res_win        = p_noimg_logps[:bs]
                    r_img_win_res_win      = r_hasimg_logps[:bs]
                    r_img_lose_res_win     = r_hasimg_logps[bs: 2*bs]
                    r_noimg_res_win        = r_noimg_logps[:bs]
                else:
                    if torch.distributed.get_rank() == 0:
                        print(f"[WARN] VCO loss enabled but missing batch keys: {missing_keys_vco}", flush=True)
            # Build two asymmetric logits
            if 'p_img_win_res_win' in locals():
                ref_free_vco = bool(getattr(self.args, 'vco_reference_free', False))
                logits_img_win_vs_no_img_v = (p_img_win_res_win - p_noimg_res_win) - (0 if ref_free_vco else (r_img_win_res_win - r_noimg_res_win))
                logits_no_img_vs_img_lose_v = (p_noimg_res_win - p_img_lose_res_win) - (0 if ref_free_vco else (r_noimg_res_win - r_img_lose_res_win))
                b1_v = max(self.args.svco_beta_img_win_vs_no_img, 0.0)
                b2_v = max(self.args.svco_beta_no_img_vs_img_lose, 0.0)
                l1_v = -F.logsigmoid(b1_v * logits_img_win_vs_no_img_v) if b1_v > 0 else torch.zeros_like(logits_img_win_vs_no_img_v)
                l2_v = -F.logsigmoid(b2_v * logits_no_img_vs_img_lose_v) if b2_v > 0 else torch.zeros_like(logits_no_img_vs_img_lose_v)
                vco_loss_value = (l1_v + l2_v).mean()
        total_loss = chip_loss_value + (self.args.svco_lambda * svco_loss_value) + (self.args.new_visual_lambda * new_visual_loss_value) + (getattr(self.args, 'vco_lambda', 0.0) * vco_loss_value) + anchor_loss_value

        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        train_test = 'train' if model.training else 'test'
        metrics = {}
        metrics[f'rewards_{train_test}/chosen'] = self._nested_gather(chosen_rewards.mean()).mean().item()
        metrics[f'rewards_{train_test}/rejected'] = self._nested_gather(rejected_rewards.mean()).mean().item()
        metrics[f'rewards_{train_test}/accuracies'] = self._nested_gather(reward_accuracies.mean()).mean().item()
        metrics[f'rewards_{train_test}/margins'] = metrics[f'rewards_{train_test}/chosen'] - metrics[f'rewards_{train_test}/rejected']
        metrics[f'logps_{train_test}/rejected'] = self._nested_gather(policy_rejected_logp.mean()).mean().item()
        metrics[f'logps_{train_test}/chosen'] = self._nested_gather(policy_chosen_logp.mean()).mean().item()
        metrics['loss'] = float(total_loss)
        metrics['chip_loss'] = float(chip_loss_value)
        metrics['anchor_loss'] = float(anchor_loss_value)
        metrics['anchor1_loss'] = float(anchor1_loss_value)
        metrics['anchor2_loss'] = float(anchor2_loss_value)
        metrics['anchor2_active'] = float(anchor2_active_flag)
        metrics['svco_loss'] = float(svco_loss_value)
        metrics['svco_active'] = float(svco_active_flag)
        metrics['vco_loss'] = float(vco_loss_value)
        metrics['new_visual_loss'] = float(new_visual_loss_value)
        metrics['new_visual_active'] = float(new_visual_active_flag)
        self.log(metrics)
        return total_loss

    def prediction_step(self, model: Module, inputs: Dict[str, torch.Tensor], prediction_loss_only: bool, ignore_keys: Optional[List[str]] = None):
        """Override evaluation step to use the custom compute_loss with DPODataset batches.

        This avoids passing unsupported keys like 'concatenated_input_ids' directly to model.forward.
        """
        # Shallow copy as compute_loss mutates the dict via pop()
        safe_inputs = {k: v for k, v in inputs.items()}
        model.eval()
        with torch.no_grad():
            loss = self.compute_loss(model, safe_inputs)
        if prediction_loss_only:
            return (loss.detach(), None, None)
        return (loss.detach(), None, None)

    def get_seg_weight(self, 
                        win_labels, rej_labels,
                        win_input_ids, rej_input_ids
                        ):
        win_token_weight = torch.ones_like(win_labels[:, 1:], dtype=torch.bfloat16)
        rej_token_weight = torch.ones_like(rej_labels[:, 1:], dtype=torch.bfloat16)
        for idx, (w, r) in enumerate(zip(win_input_ids, rej_input_ids)):
            valid_w = w[1:]
            valid_r = r[1:]
            min_match_size = 3
            r_mod, w_mod = get_diff_ids(valid_r.tolist(), valid_w.tolist(), min_match_size=min_match_size)
            win_token_weight[idx][w_mod] = self.args.dpo_token_weight
            rej_token_weight[idx][r_mod] = self.args.dpo_token_weight

        return win_token_weight, rej_token_weight
    
    @staticmethod    
    def compute_weighted_logp(per_token_logp, labels, token_weight, use_average=False):
        loss_mask = (labels[:, 1:].clone() != -100)
        weighted_mask = token_weight * loss_mask
        if len(per_token_logp.shape)!=1:
            per_token_logp = per_token_logp[:, -weighted_mask.shape[1]:]
        # Replace -inf/+inf with 0 before multiplication to avoid 0 * inf -> nan
        per_token_logp = torch.where(torch.isfinite(per_token_logp), per_token_logp, torch.zeros_like(per_token_logp))
        logp_sum = (per_token_logp * weighted_mask).sum(-1)
        denom = weighted_mask.sum(-1).clamp_min(1)
        average_logp = logp_sum / denom
        return average_logp if use_average else logp_sum
