#!/bin/bash
#SBATCH -J huginn_noise_ablation
#SBATCH -A gts-vfung3
#SBATCH -o logs/noise_ablation_%j.out
#SBATCH -qinferno
#SBATCH -N1 --gres=gpu:A100:1
#SBATCH --mem=80G
#SBATCH -t24:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hjiang378@gatech.edu

# ── Gaussian-noise CCoT ablation ──────────────────────────────────────────────
# Mirrors the original full-rank frozen run on winogrande_hellaswag, but
# replaces CCoT latent chains with norm-matched Gaussian noise.
#
# Submit:
#   sbatch submit_noise_ablation.slurm
#
# Optional overrides via --export:
#   sbatch --export=NOISE_SCALE=0.5 submit_noise_ablation.slurm
# ──────────────────────────────────────────────────────────────────────────────

NOISE_SCALE=${NOISE_SCALE:-1.0}

mkdir -p logs

source ~/.bashrc
conda activate huginn

export HF_HOME=$HOME/scratch/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME
mkdir -p $HF_HOME

cd $SLURM_SUBMIT_DIR

RUN_ID=$(date +%Y%m%d_%H%M%S)

PYTHON=$HOME/conda_envs/huginn/bin/python
$PYTHON train_noise_ablation.py \
    --output_dir         ./runs/noise_ablation_${RUN_ID} \
    --model_name         tomg-group-umd/huginn-0125 \
    --hf_cache           $HOME/scratch/hf_cache \
    --max_train_samples  32000 \
    --epochs             3 \
    --lr                 1e-4 \
    --weight_decay       0.1 \
    --batch_size         1 \
    --grad_accum         32 \
    --num_steps          32 \
    --num_passes         4 \
    --backprop_depth     4 \
    --noise_scale        $NOISE_SCALE \
    --grad_checkpoint

echo "Job finished: noise_ablation noise_scale=${NOISE_SCALE} run=${RUN_ID}"
