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

# ── Usage ─────────────────────────────────────────────────────────────────────
# Submit 4 variants (frozen/unfrozen × ccot_only/iter_only):
#
#   sbatch --export=EXPERIMENT=ccot_only,FROZEN=1 submit_ccot.slurm   # frozen ccot
#   sbatch --export=EXPERIMENT=ccot_only,FROZEN=0 submit_ccot.slurm   # unfrozen ccot
#   sbatch --export=EXPERIMENT=iter_only,FROZEN=1 submit_ccot.slurm   # frozen iter
#   sbatch --export=EXPERIMENT=iter_only,FROZEN=0 submit_ccot.slurm   # unfrozen iter
#
# EXPERIMENT defaults to ccot_only, FROZEN defaults to 1 if not set.
# ──────────────────────────────────────────────────────────────────────────────

EXPERIMENT=${EXPERIMENT:-ccot_only}
FROZEN=${FROZEN:-1}

mkdir -p logs

source ~/.bashrc
conda activate huginn

export HF_HOME=$HOME/scratch/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME
mkdir -p $HF_HOME

cd $SLURM_SUBMIT_DIR

RUN_ID=$(date +%Y%m%d_%H%M%S)

FROZEN_FLAG=""
if [ "$FROZEN" = "1" ]; then
    FROZEN_FLAG="--no_train_loop"
fi

PYTHON=$HOME/conda_envs/huginn/bin/python
$PYTHON train_ccot.py \
    --experiments        $EXPERIMENT \
    --output_dir         ./runs/${EXPERIMENT}_frozen${FROZEN}_${RUN_ID} \
    --model_name         tomg-group-umd/huginn-0125 \
    --hf_cache           $HOME/scratch/hf_cache \
    --dataset            pretrain_sample \
    --max_train_samples  32000 \
    --epochs             3 \
    --lr                 1e-4 \
    --weight_decay       0.1 \
    --batch_size         1 \
    --grad_accum         32 \
    --num_steps          32 \
    --backprop_depth     4 \
    --proj_bottleneck_dim 64 \
    --noise              0.15 \
    --noise_type         geom \
    --grad_checkpoint \
    $FROZEN_FLAG \
    --skip_eval \
    --skip_qualitative

echo "Job finished: ${EXPERIMENT} frozen=${FROZEN} run=${RUN_ID}"
