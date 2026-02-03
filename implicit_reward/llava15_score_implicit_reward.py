#!/usr/bin/env python
"""
Implicit Reward Scoring & RVG-based Preference Pair Construction (Eq. 8-11)

Key components:
- r_image: Implicit reward with image context (Eq. 8)
- r_text: Implicit reward with text-only context (Eq. 9)  
- RVG Score: Rectified Visual Guidance scoring (Eq. 10)
- Preference Pair Selection: Select y_w, y_l based on RVG (Eq. 11)
"""
import os
import json
import argparse
from collections import defaultdict
from typing import List, Dict, Any

import torch
import torch.distributed as dist
from PIL import Image

from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, get_model_name_from_path
from muffin.train.train_utils import preprocess_v1, DEFAULT_IMAGE_TOKEN
from muffin.eval.muffin_inference_logp import get_batch_logps


def read_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict):
            for k in ['data', 'samples', 'rows']:
                if k in data and isinstance(data[k], list):
                    return data[k]
            return [data]
        return data
    except Exception:
        rows = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows


def build_inputs_for_llava(model, tokenizer, image_tensor: torch.Tensor, question: str, answer: str, image_sizes=None):
    """Construct inputs_embeds and labels using preprocess_v1 + LLaVA multimodal prepare helper."""
    sample = [
        {"from": "human", "value": f"{DEFAULT_IMAGE_TOKEN}\n{question}"},
        {"from": "gpt", "value": answer},
    ]
    pack = preprocess_v1([sample], tokenizer, has_image=True)
    input_ids = pack["input_ids"].to(model.device)
    labels = pack["labels"].to(model.device)

    (
        _,
        _,
        _,
        _,
        inputs_embeds,
        labels
    ) = model.prepare_inputs_labels_for_multimodal(
        input_ids=input_ids,
        position_ids=None,
        attention_mask=None,
        past_key_values=None,
        labels=labels,
        images=image_tensor.to(dtype=torch.bfloat16, device=model.device),
        image_sizes=image_sizes,
    )
    return inputs_embeds, labels


@torch.inference_mode()
def avg_logp_llava(model, tokenizer, image_tensor: torch.Tensor, question: str, answer: str, image_sizes=None) -> float:
    inputs_embeds, labels = build_inputs_for_llava(model, tokenizer, image_tensor, question, answer, image_sizes=image_sizes)
    out = model.forward(inputs_embeds=inputs_embeds, labels=None)
    # get average log probability over non-masked tokens (assistant output)
    _, avg_lp = get_batch_logps(out.logits, labels, return_all=False)
    return avg_lp.item()


@torch.inference_mode()
def avg_logp_llava_text_only(model, tokenizer, question: str, answer: str) -> float:
    """Compute avg log-probability for text-only (no image) condition p(y | Q)."""
    sample = [
        {"from": "human", "value": f"{question}"},
        {"from": "gpt", "value": answer},
    ]
    pack = preprocess_v1([sample], tokenizer, has_image=False)
    input_ids = pack["input_ids"].to(model.device)
    labels = pack["labels"].to(model.device)

    out = model.forward(input_ids=input_ids, labels=None)
    _, avg_lp = get_batch_logps(out.logits, labels, return_all=False)
    return avg_lp.item()


@torch.inference_mode()
def avg_logp_llava_batch(model, tokenizer, image_tensor: torch.Tensor, question: str, answers: List[str], image_sizes=None) -> List[float]:
    """Compute avg log-probability for multiple answers with image (batch processing)."""
    from llava.mm_utils import tokenizer_image_token
    from torch.nn.utils.rnn import pad_sequence
    
    # Tokenize question part to get its length
    prompt_template = f"{DEFAULT_IMAGE_TOKEN}\n{question}"
    question_tokens = tokenizer_image_token(prompt_template, tokenizer, return_tensors='pt').squeeze(0)
    question_len = len(question_tokens)
    
    # Tokenize each answer separately to handle different lengths
    input_ids_list = []
    question_lengths = []
    for answer in answers:
        full_prompt = f"{prompt_template}{answer}"
        input_ids = tokenizer_image_token(full_prompt, tokenizer, return_tensors='pt').squeeze(0)
        input_ids_list.append(input_ids)
        question_lengths.append(question_len)
    
    # Pad sequences to the same length
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_id).to(model.device)
    
    # Create labels: mask out question part and padding tokens, only keep answer part
    labels = input_ids.clone()
    labels[labels == pad_token_id] = -100  # Mask padding
    for i, q_len in enumerate(question_lengths):
        labels[i, :q_len] = -100  # Mask question part
    
    # Expand image_tensor to match batch size
    # image_tensor should be [1, C, H, W] or [C, H, W]
    if image_tensor.dim() == 4:
        # Already has batch dimension [1, C, H, W], expand to [batch_size, C, H, W]
        if image_tensor.size(0) == 1:
            batch_image_tensor = image_tensor.repeat(len(answers), 1, 1, 1)
        else:
            batch_image_tensor = image_tensor
    elif image_tensor.dim() == 3:
        # [C, H, W], add batch dimension and expand
        batch_image_tensor = image_tensor.unsqueeze(0).repeat(len(answers), 1, 1, 1)
    else:
        batch_image_tensor = image_tensor
    
    batch_image_sizes = image_sizes * len(answers) if image_sizes else None

    (
        _,
        _,
        _,
        _,
        inputs_embeds,
        labels
    ) = model.prepare_inputs_labels_for_multimodal(
        input_ids=input_ids,
        position_ids=None,
        attention_mask=None,
        past_key_values=None,
        labels=labels,
        images=batch_image_tensor.to(dtype=torch.bfloat16, device=model.device),
        image_sizes=batch_image_sizes,
    )
    out = model.forward(inputs_embeds=inputs_embeds, labels=None)
    _, _, avg_lps = get_batch_logps(out.logits, labels, return_all=True)
    return [lp.item() for lp in avg_lps]


@torch.inference_mode()
def avg_logp_llava_text_only_batch(model, tokenizer, question: str, answers: List[str]) -> List[float]:
    """Compute avg log-probability for multiple answers text-only (batch processing)."""
    # Process each sample separately to ensure correct label masking
    # This is simpler and ensures consistency with single-sample processing
    results = []
    for answer in answers:
        sample = [
            {"from": "human", "value": f"{question}"},
            {"from": "gpt", "value": answer},
        ]
        pack = preprocess_v1([sample], tokenizer, has_image=False)
        input_ids = pack["input_ids"].to(model.device)
        labels = pack["labels"].to(model.device)
        
        out = model.forward(input_ids=input_ids, labels=None)
        _, avg_lp = get_batch_logps(out.logits, labels, return_all=False)
        results.append(avg_lp.item())
    return results


def load_llava(ckpt_path: str, local_rank: int = 0):
    model_path = os.path.expanduser(ckpt_path)
    model_name = get_model_name_from_path(model_path)
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    tokenizer, model, image_processor, _context_len = load_pretrained_model(
        model_path, None, model_name, device_map={"": device}
    )
    model = model.to(dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32), device=device)
    return tokenizer, model, image_processor


def _score_with_model(ckpt: str, groups: Dict[str, List[Dict[str, Any]]], score_key: str, local_rank: int = 0, batch_size: int = 1):
    tok, model, img_proc = load_llava(ckpt, local_rank=local_rank)
    for key, cand_list in groups.items():
        if not cand_list:
            continue
        image_path = cand_list[0]['image_path']
        question = cand_list[0]['question']
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Failed to open image: {image_path}: {e}")
        
        processed = process_images([image], img_proc, model.config)
        image_tensor = processed if not isinstance(processed, (list, tuple)) else processed[0]
        image_sizes = [(image.width, image.height)]
        
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        
        valid_candidates = [c for c in cand_list if isinstance(c.get('answer', ''), str) and c.get('answer', '').strip()]
        
        for i in range(0, len(valid_candidates), batch_size):
            batch = valid_candidates[i:i + batch_size]
            if not batch:
                continue
            
            answers = [c.get('answer', '') for c in batch]
            avg_imgs = avg_logp_llava_batch(model, tok, image_tensor, question, answers, image_sizes=image_sizes)
            avg_txts = avg_logp_llava_text_only_batch(model, tok, question, answers)
            
            for j, c in enumerate(batch):
                avg_img = avg_imgs[j] if isinstance(avg_imgs, list) else avg_imgs
                avg_txt = avg_txts[j] if isinstance(avg_txts, list) else avg_txts
                c[score_key] = avg_img
                c[f"{score_key}_image"] = avg_img
                c[f"{score_key}_text"] = avg_txt
    del model
    torch.cuda.empty_cache()


def score_candidates(policy_ckpt: str, ref_ckpt: str, input_path: str, output_pairs_path: str, beta: float = 0.1, gamma: float = 0.0, gamma_mode: str = 'linear', save_all_candidates: bool = False, batch_size: int = 1):
    """Score candidates with implicit rewards and construct preference pairs (Eq. 8-11)."""
    world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    local_rank = int(os.getenv('LOCAL_RANK', '0'))

    data = read_json_or_jsonl(input_path)
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError(f"No data loaded from {input_path}")

    groups = defaultdict(list)
    for item in data:
        question = item.get('raw_question') or item.get('question') or ''
        metainfos = item.get('metainfos', {})
        image_path = metainfos.get('image_path') or item.get('image_path')
        ds_qid = metainfos.get('ds_question_id') or item.get('ds_question_id') or (image_path or '')
        if image_path is None:
            if 'metainfos' in item and isinstance(item['metainfos'], dict):
                image_path = item['metainfos'].get('image_path')
        if image_path is None:
            raise ValueError("Missing image_path in generated data; please ensure inputs contain image_path")
        key = f"{ds_qid}@{question}"
        groups[key].append({
            'question_id': item.get('question_id'),
            'question': question,
            'answer': item.get('answer', ''),
            'image_path': image_path,
            'ds_question_id': ds_qid,
            'metainfos': metainfos,
        })

    print(f"Loaded {len(data)} samples, grouped into {len(groups)} questions.")

    # Shard groups across ranks if distributed
    if world_size > 1:
        keys_sorted = sorted(groups.keys())
        assigned_keys = [k for i, k in enumerate(keys_sorted) if i % world_size == rank]
        shard_groups = {k: groups[k] for k in assigned_keys}
    else:
        shard_groups = groups

    # Score sequentially to reduce memory footprint
    _score_with_model(policy_ckpt, shard_groups, '__p_avg', local_rank=local_rank, batch_size=batch_size)
    _score_with_model(ref_ckpt, shard_groups, '__r_avg', local_rank=local_rank, batch_size=batch_size)

    out_pairs = []
    for key, cand_list in shard_groups.items():
        # Only keep candidates scored by both models
        scored = []
        
        for c in cand_list:
            if '__p_avg' in c and '__r_avg' in c:
                p_img = c.get('__p_avg_image', c['__p_avg'])
                r_img = c.get('__r_avg_image', c['__r_avg'])
                p_txt = c.get('__p_avg_text', None)
                r_txt = c.get('__r_avg_text', None)

                # === r_image: Implicit Reward with Image (Eq. 8) ===
                # r_image(v,x,y) = beta * log[pi(y|v,x) / pi_ref(y|v,x)]
                r_image = beta * (p_img - r_img)
                
                # === r_text: Implicit Reward Text-only (Eq. 9) ===
                # r_text(x,y) = beta * log[pi(y|x) / pi_ref(y|x)]
                r_text = beta * (p_txt - r_txt) if (p_txt is not None and r_txt is not None) else 0.0

                # === RVG Score (Eq. 10) ===
                # S(v,x,y) = r_image - gamma * max(0, r_text - r_image)
                if gamma_mode == 'relu':
                    reward = r_image - gamma * (r_text - r_image) if r_text > r_image else r_image
                elif gamma_mode == 'linear':
                    reward = r_image + gamma * (r_image - r_text)
                else:
                    raise ValueError(f"Unknown gamma_mode: {gamma_mode}")
                
                scored.append((reward, r_image, r_text, p_img, r_img, p_txt, r_txt, c))
        if len(scored) < 2:
            continue
        # === Preference Pair Selection (Eq. 11) ===
        # y_w = argmax S(v,x,y), y_l = argmin S(v,x,y)
        scored.sort(key=lambda x: x[0])
        rejected_tuple = scored[0]  # y_l: lowest RVG score
        chosen_tuple   = scored[-1]  # y_w: highest RVG score

        rejected_reward, rejected_r_image, rejected_r_text, rej_p_img, rej_r_img, rej_p_txt, rej_r_txt, rejected = rejected_tuple
        chosen_reward, chosen_r_image, chosen_r_text, ch_p_img, ch_r_img, ch_p_txt, ch_r_txt, chosen = chosen_tuple

        image_path = chosen['image_path']
        ds_qid = chosen['ds_question_id']
        image_id = os.path.basename(image_path)
        question = chosen['question']

        org_infos = {
            'scores': {
                'chosen': {
                    'reward': float(chosen_reward),
                    'r_image': float(chosen_r_image),
                    'r_text': float(chosen_r_text),
                    'p_avg_image': float(ch_p_img),
                    'r_avg_image': float(ch_r_img),
                    'p_avg_text': (float(ch_p_txt) if ch_p_txt is not None else None),
                    'r_avg_text': (float(ch_r_txt) if ch_r_txt is not None else None),
                },
                'rejected': {
                    'reward': float(rejected_reward),
                    'r_image': float(rejected_r_image),
                    'r_text': float(rejected_r_text),
                    'p_avg_image': float(rej_p_img),
                    'r_avg_image': float(rej_r_img),
                    'p_avg_text': (float(rej_p_txt) if rej_p_txt is not None else None),
                    'r_avg_text': (float(rej_r_txt) if rej_r_txt is not None else None),
                },
            },
            'policy_ckpt': policy_ckpt,
            'ref_ckpt': ref_ckpt,
            'beta': beta,
            'gamma': gamma,
            'gamma_mode': gamma_mode,
        }

        pair_data = {
            'image_id': image_id,
            'image_path': image_path,
            'ds_question_id': ds_qid,
            'question': question,
            'chosen': chosen['answer'],
            'rejected': rejected['answer'],
            'org_infos': org_infos,
        }

        # Add all candidates information if requested
        if save_all_candidates:
            all_candidates = []
            for reward, r_image, r_text, p_img, r_img, p_txt, r_txt, candidate in scored:
                candidate_info = {
                    'answer': candidate['answer'],
                    'reward': float(reward),
                    'r_image': float(r_image),
                    'r_text': float(r_text),
                    'p_avg_image': float(p_img),
                    'r_avg_image': float(r_img),
                    'p_avg_text': (float(p_txt) if p_txt is not None else None),
                    'r_avg_text': (float(r_txt) if r_txt is not None else None),
                    'question_id': candidate.get('question_id'),
                    'metainfos': candidate.get('metainfos', {}),
                }
                all_candidates.append(candidate_info)
            pair_data['all_candidates'] = all_candidates

        out_pairs.append(pair_data)

    # Write outputs: partial per rank, then merge on rank 0
    os.makedirs(os.path.dirname(output_pairs_path), exist_ok=True)
    if world_size > 1:
        part_path = f"{output_pairs_path}.part{rank}"
        with open(part_path, 'w', encoding='utf-8') as f:
            for row in out_pairs:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        if rank == 0:
            merged = 0
            with open(output_pairs_path, 'w', encoding='utf-8') as fout:
                for r in range(world_size):
                    p = f"{output_pairs_path}.part{r}"
                    if not os.path.exists(p):
                        continue
                    with open(p, 'r', encoding='utf-8') as fin:
                        for line in fin:
                            fout.write(line)
                            merged += 1
                    try:
                        os.remove(p)
                    except Exception:
                        pass
            print(f"Wrote {merged} pairs to {output_pairs_path}")
    else:
        with open(output_pairs_path, 'w', encoding='utf-8') as f:
            for row in out_pairs:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
        print(f"Wrote {len(out_pairs)} pairs to {output_pairs_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy_ckpt', type=str, required=True)
    parser.add_argument('--ref_ckpt', type=str, required=True)
    parser.add_argument('--input_file', type=str, required=True, help='Diverse generation output (json or jsonl)')
    parser.add_argument('--output_pairs_file', type=str, required=True)
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.0, help='weight for (r_image - r_text) term')
    parser.add_argument('--gamma_mode', type=str, default='linear', choices=['linear', 'relu'],
                        help="Reward computation mode: 'linear' (original) or 'relu' (ReLU-gated, only penalize when r_text > r_image)")
    parser.add_argument('--save_all_candidates', action='store_true',
                        help="Save all candidates with their implicit reward scores in addition to chosen/rejected pairs")
    parser.add_argument('--batch_size', type=int, default=1,
                        help="Batch size for processing candidates (per GPU)")

    args = parser.parse_args()

    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    world_size_env = int(os.getenv('WORLD_SIZE', '1'))
    use_dist = world_size_env > 1 and torch.cuda.is_available()

    if use_dist and dist.is_available() and not dist.is_initialized():
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
    else:
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

    score_candidates(
        policy_ckpt=args.policy_ckpt,
        ref_ckpt=args.ref_ckpt,
        input_path=args.input_file,
        output_pairs_path=args.output_pairs_file,
        beta=args.beta,
        gamma=args.gamma,
        gamma_mode=args.gamma_mode,
        save_all_candidates=args.save_all_candidates,
        batch_size=args.batch_size,
    )

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
