#!/bin/bash
#SBATCH --job-name=huginn_ccot
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1          # adjust to your cluster's GPU name
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu            # adjust to your cluster's partition name

# ── Create log dir if it doesn't exist ────────────────────────────────────
mkdir -p logs

# ── Activate conda env ────────────────────────────────────────────────────
source ~/.bashrc
conda activate huginn            # or: module load anaconda; conda activate huginn

# ── Point HuggingFace cache to scratch (avoids filling home quota) ────────
export HF_HOME=/scratch/$USER/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME

# ── Move to repo root ─────────────────────────────────────────────────────
cd $SLURM_SUBMIT_DIR

# ── Run ──────────────────────────────────────────────────────────────────
RUN_ID=$(date +%Y%m%d_%H%M%S)

python train_ccot.py \
    --experiments iter_only ccot_only both \
    --output_dir  ./runs/ccot_${RUN_ID} \
    --model_name  tomg-group-umd/huginn-0125 \
    --hf_cache    /scratch/$USER/hf_cache \
    --epochs      5 \
    --lr          3e-4 \
    --batch_size  2 \
    --grad_accum  16 \
    --num_steps   32 \
    --backprop_depth 8 \
    --n_eval      200

echo "Job finished: $RUN_ID"
