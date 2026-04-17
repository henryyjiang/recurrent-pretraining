#!/bin/bash
#SBATCH -J huginn_ccot
#SBATCH -A gts-vfung3
#SBATCH -o logs/ccot_%j.out
#SBATCH -qinferno
#SBATCH -N1 --gres=gpu:A100:1
#SBATCH --mem=80G
#SBATCH -t24:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hjiang378@gatech.edu

# ── Create log dir if it doesn't exist ────────────────────────────────────
mkdir -p logs

# ── Activate conda env ────────────────────────────────────────────────────
source ~/.bashrc
conda activate huginn

# ── Point HuggingFace cache to scratch (avoids filling home quota) ────────
export HF_HOME=$HOME/scratch/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME
mkdir -p $HF_HOME

# ── Move to repo root ─────────────────────────────────────────────────────
cd $SLURM_SUBMIT_DIR

# ── Run ──────────────────────────────────────────────────────────────────
RUN_ID=$(date +%Y%m%d_%H%M%S)

PYTHON=$HOME/conda_envs/huginn/bin/python
$PYTHON train_ccot.py \
    --experiments ccot_only \
    --output_dir  ./runs/ccot_${RUN_ID} \
    --model_name  tomg-group-umd/huginn-0125 \
    --hf_cache    $HOME/scratch/hf_cache \
    --dataset     winogrande_hellaswag \
    --epochs      3 \
    --lr          1e-4 \
    --weight_decay 0.1 \
    --batch_size  1 \
    --grad_accum  32 \
    --num_steps   32 \
    --backprop_depth 4 \
    --grad_checkpoint \
    --no_train_loop \
    --skip_eval \
    --skip_qualitative

echo "Job finished: $RUN_ID"
