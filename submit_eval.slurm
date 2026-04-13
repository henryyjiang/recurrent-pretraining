#!/bin/bash
#SBATCH -J huginn_eval
#SBATCH -A gts-vfung3
#SBATCH -o logs/huginn_eval_%j.out
#SBATCH -qinferno
#SBATCH -N1 --gres=gpu:A100:1
#SBATCH --mem=80G
#SBATCH -t08:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hjiang378@gatech.edu

mkdir -p logs

source ~/.bashrc
conda activate huginn

export HF_HOME=$HOME/scratch/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME
mkdir -p $HF_HOME

cd $SLURM_SUBMIT_DIR

python eval_ccot.py \
    --run_dir    ./runs/ccot_20260413_012258 \
    --model_name tomg-group-umd/huginn-0125 \
    --hf_cache   $HOME/scratch/hf_cache \
    --tasks      arc_challenge,hellaswag,winogrande,mmlu,gsm8k \
    --num_steps  8,16,32 \
    --output_dir ./runs/ccot_20260413_012258/eval \
    --experiments Baseline iter_only ccot_only both

echo "Eval finished."
