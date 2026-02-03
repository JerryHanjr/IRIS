# IRIS: Implicit Reward-Guided Internal Sifting

This repository contains the core implementation of all key components presented in the paper. This codebase provides the complete source code for implicit reward computation, RVG-based preference pair construction, and the grounded preference learning framework.

**Note:** Pre-trained model checkpoints will be released upon paper acceptance.

## Directory Structure

```
IRIS_core/
├── training/
│   ├── train.py              # DPO training with Lctp, Lcvp, Lanchor (Eq. 12-14)
│   ├── llava_trainer.py      # LLAVADPOTrainer with chip_loss()
│   ├── train_mem.py          # Training entry point
│   └── zero3.json            # DeepSpeed ZeRO-3 configuration
├── implicit_reward/
│   └── llava15_score_implicit_reward.py  # Implicit reward scoring (Eq. 8-11)
└── data_generation/
    └── llava15_gen_data.py   # On-policy self-generation (Eq. 7)
```

## Core Components

| Component | Paper Reference | Implementation |
|-----------|-----------------|----------------|
| Implicit Reward (r_image) | Eq. 8 | `llava15_score_implicit_reward.py` |
| Implicit Reward (r_text) | Eq. 9 | `llava15_score_implicit_reward.py` |
| RVG Score S(v,x,y) | Eq. 10 | `score_candidates()` |
| Preference Pair Selection | Eq. 11 | `score_candidates()` |
| Lctp (DPO Loss) | Eq. 12 | `chip_loss()` in `llava_trainer.py` |
| Lcvp (Cross-Modal Loss) | Eq. 13 | `--use_cross_modal_loss` in `train.py` |
| Lanchor (Anchor Loss) | Eq. 14 | `--use_anchor_loss` in `train.py` |
| Diffusion Noise | Section 4.4 | `add_diffusion_noise()` in `train.py` |

## Key Hyperparameters

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| `--dpo_beta` | β | 0.1 | DPO temperature |
| `--use_cross_modal_gama` | γ | 1.0 | RVG correction strength |
| `--diffusion_step` | - | 500 | Noise steps for perturbed image |
| `--gamma` (scoring) | γ | 0.7 | RVG penalty coefficient |

## Usage

### 1. On-policy Self-Generation (Eq. 7)
```bash
python data_generation/llava15_gen_data.py \
    --checkpoint <model_path> \
    --ds_name <input_questions> \
    --repeat 5 \
    --temperature 0.7
```

### 2. Implicit Reward Scoring (Eq. 8-11)
```bash
python implicit_reward/llava15_score_implicit_reward.py \
    --policy_ckpt <current_model> \
    --ref_ckpt <reference_model> \
    --input_file <generated_responses> \
    --gamma 0.7 \
    --gamma_mode relu
```

### 3. Grounded Preference Learning (Eq. 12-14)
```bash
deepspeed training/train_mem.py \
    --deepspeed training/zero3.json \
    --task DPO \
    --use_cross_modal_loss True \
    --use_anchor_loss True \
    --dpo_beta 0.1 \
    --use_cross_modal_gama 1.0
```
