#!/bin/bash
#SBATCH -J huginn_eval
#SBATCH -A gts-vfung3
#SBATCH -o logs/huginn_eval_%j.out
#SBATCH -qinferno
#SBATCH -N1 --gres=gpu:A100:1
#SBATCH --mem=80G
#SBATCH -t24:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hjiang378@gatech.edu

mkdir -p logs

export HF_HOME=$HOME/scratch/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME
mkdir -p $HF_HOME

cd $SLURM_SUBMIT_DIR

# ── Set this to the output_dir used when running train_ccot.py ─────────────
RUN_DIR=./runs/ccot_PLACEHOLDER

PYTHON=$HOME/conda_envs/huginn/bin/python
$PYTHON eval_ccot.py \
    --run_dir    $RUN_DIR \
    --model_name tomg-group-umd/huginn-0125 \
    --hf_cache   $HOME/scratch/hf_cache \
    --tasks      arc_challenge,hellaswag,winogrande,gsm8k \
    --num_steps  32 \
    --num_passes 4 \
    --output_dir $RUN_DIR/eval \
    --batch_size  4 \
    --experiments ccot_only

echo "Eval finished."
