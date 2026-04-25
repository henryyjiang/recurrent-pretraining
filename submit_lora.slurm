#!/bin/bash
#SBATCH -J huginn_lora
#SBATCH -A gts-vfung3
#SBATCH -o logs/lora_%j.out
#SBATCH -qinferno
#SBATCH -N1 --gres=gpu:A100:1
#SBATCH --mem=80G
#SBATCH -t24:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hjiang378@gatech.edu

# ── LoRA fine-tuning of Huginn core_block ─────────────────────────────────────
# Applies rank-r LoRA adapters to attention (Wqkv, proj) and MLP (fc, proj)
# layers in the recurrent core_block.  No CCoT latent chaining — standard
# single-pass LM training.  Comparable parameter budget to the CCoT bottleneck
# run; adjust LORA_RANK to match.
#
# Submit:
#   sbatch submit_lora.slurm
#
# Optional overrides:
#   sbatch --export=LORA_RANK=16 submit_lora.slurm
# ──────────────────────────────────────────────────────────────────────────────

LORA_RANK=${LORA_RANK:-32}

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
$PYTHON train_lora.py \
    --output_dir         ./runs/lora_r${LORA_RANK}_${RUN_ID} \
    --model_name         tomg-group-umd/huginn-0125 \
    --hf_cache           $HOME/scratch/hf_cache \
    --max_train_samples  32000 \
    --epochs             3 \
    --lr                 1e-4 \
    --weight_decay       0.1 \
    --batch_size         1 \
    --grad_accum         32 \
    --num_steps          32 \
    --backprop_depth     4 \
    --lora_rank          $LORA_RANK \
    --lora_targets       "Wqkv,proj,fc" \
    --grad_checkpoint

echo "Job finished: lora rank=${LORA_RANK} run=${RUN_ID}"
