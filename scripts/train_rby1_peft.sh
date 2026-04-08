#!/bin/bash
#SBATCH --job-name=PuttingCupintotheDish_demo50
#SBATCH --partition=gigabyte_a6000
#SBATCH --gres=gpu:1
#SBATCH --time=36:00:00
#SBATCH --output=/lustre/meat124/X-VLA/logs/%x_%j.out
#SBATCH --error=/lustre/meat124/X-VLA/logs/%x_%j.err

# ---------------------------------------------------------------
# X-VLA LoRA Fine-tuning for RBY1 — PuttingCupintotheDish
#
# Usage:
#   sbatch scripts/train_rby1_peft.sh                        # default
#   sbatch scripts/train_rby1_peft.sh --iters 100000         # longer
#   sbatch scripts/train_rby1_peft.sh --batch_size 8         # smaller BS
#   sbatch --partition=suma_a100 scripts/train_rby1_peft.sh  # A100
# ---------------------------------------------------------------

set -euo pipefail

# --- configurable paths ---
XVLA_DIR=/lustre/meat124/X-VLA
DATASET_META=/lustre/meat124/rby1_demo/LeRobotDataset_v2/PuttingCupintotheDishV2/meta/info.json
BASE_MODEL=2toINF/X-VLA-Pt
OUTPUT_DIR=${XVLA_DIR}/checkpoints/${SLURM_JOB_NAME}

cd "$XVLA_DIR"

# create dirs
mkdir -p "$XVLA_DIR/logs"
mkdir -p "$OUTPUT_DIR"

echo "=============================="
echo "Job ID       : $SLURM_JOB_ID"
echo "Job Name     : $SLURM_JOB_NAME"
echo "Partition    : $SLURM_JOB_PARTITION"
echo "Node         : $(hostname)"
echo "GPU          : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Output Dir   : $OUTPUT_DIR"
echo "Dataset      : $DATASET_META"
echo "Base Model   : $BASE_MODEL"
echo "Start        : $(date)"
echo "=============================="

# --- activate conda ---
source ~/anaconda3/etc/profile.d/conda.sh
conda activate XVLA

# --- train ---
# All extra arguments passed to sbatch are forwarded to peft_train.py.
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
accelerate launch --mixed_precision bf16 peft_train.py \
    --models "$BASE_MODEL" \
    --train_metas_path "$DATASET_META" \
    --output_dir "$OUTPUT_DIR" \
    --action_mode auto \
    --real_action_dim 16 \
    --max_action_dim 20 \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --learning_coef 1.0 \
    --weight_decay 0.0 \
    --iters 50000 \
    --freeze_steps 1000 \
    --warmup_steps 2000 \
    --save_interval 10000 \
    --log_interval 20 \
    --use_wandb \
    --wandb_project "X-VLA" \
    --wandb_run_name "${SLURM_JOB_NAME}" \
    --seed 42 \
    --num_episodes 50 \
    "$@"

echo "=============================="
echo "End          : $(date)"
echo "=============================="
