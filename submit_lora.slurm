#!/bin/bash
#SBATCH -J huginn_lora_ccot
#SBATCH -A gts-vfung3
#SBATCH -o logs/lora_%j.out
#SBATCH -qinferno
#SBATCH -N1 --gres=gpu:A100:1
#SBATCH --mem=80G
#SBATCH -t24:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hjiang378@gatech.edu

# ── LoRA + CCoT: LoRA adapters on core_block + CCoT latent chaining ───────────
# Frozen base weights; LoRA deltas (rank=LORA_RANK) + ccot_proj are trained.
# Run with ccot_only or both; single slurm job each.
#
# Submit:
#   sbatch --export=EXPERIMENT=ccot_only,LORA_RANK=32 submit_lora.slurm
#   sbatch --export=EXPERIMENT=both,LORA_RANK=32      submit_lora.slurm
# ──────────────────────────────────────────────────────────────────────────────

EXPERIMENT=${EXPERIMENT:-ccot_only}
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
$PYTHON train_ccot.py \
    --experiments        $EXPERIMENT \
    --output_dir         ./runs/lora_r${LORA_RANK}_${EXPERIMENT}_${RUN_ID} \
    --model_name         tomg-group-umd/huginn-0125 \
    --hf_cache           $HOME/scratch/hf_cache \
    --dataset            winogrande_hellaswag \
    --max_train_samples  32000 \
    --epochs             3 \
    --lr                 1e-4 \
    --weight_decay       0.1 \
    --batch_size         1 \
    --grad_accum         32 \
    --num_steps          32 \
    --num_passes         4 \
    --backprop_depth     4 \
    --proj_bottleneck_dim 0 \
    --lora_rank          $LORA_RANK \
    --lora_targets       "Wqkv,proj,fc" \
    --no_train_loop \
    --grad_checkpoint \
    --skip_eval \
    --skip_qualitative

echo "Job finished: lora+ccot exp=${EXPERIMENT} rank=${LORA_RANK} run=${RUN_ID}"
