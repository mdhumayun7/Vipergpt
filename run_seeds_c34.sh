#!/bin/bash
#SBATCH --job-name=viper_seedc34
#SBATCH --partition=gpu
#SBATCH --output=outputs/slurm/%x_%j.out
#SBATCH --error=outputs/slurm/%x_%j.err
#SBATCH --gres=shard:25
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
module load anaconda3-2024.2
module load cuda-12.8
cd /home/mazaveri/hpc-prog/humayun/vipergpt
export HF_HOME=/home/mazaveri/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
source "$(conda info --base)/etc/profile.d/conda.sh"
MODEL=Qwen/Qwen2.5-Coder-7B-Instruct

# ---- generation (vipergpt env)
conda activate vipergpt || true
export PATH="/home/mazaveri/.conda/envs/vipergpt/bin:$PATH"
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo "[FATAL] no GPU"; exit 1; }
for S in 1 2 3; do
  for P in api_depth_cond.prompt api_depth.prompt; do
    echo ""; echo ">>> GEN seed=$S | $P | refcoco"
    python -m vipergpt_repro.eval.codegen_analysis --version refcoco --split testA \
        --max-samples 500 --model "$MODEL" --prompt "prompts/$P" \
        --batch-size 8 --temperature 0.7 --seed $S
  done
done
conda deactivate || true

# ---- execution (glip_env)
conda activate glip_env || true
export PATH="/home/mazaveri/.conda/envs/glip_env/bin:$PATH"
export PYTHONPATH=$PWD/src
for S in 1 2 3; do
  for P in depth_cond depth; do
    D=$(ls -1dt outputs/runs/*m1_refcoco_testA_7B_${P}_t0.7s${S} 2>/dev/null | head -1)
    [ -z "$D" ] && { echo "[SKIP] $P seed $S"; continue; }
    echo ""; echo ">>> EXEC seed=$S | $P  ($D)"
    python -m vipergpt_repro.eval.execute_refcoco --programs "$D/programs.jsonl" \
        --version refcoco --max-samples 500 --timeout 300 --tag seed_${P}_s${S}
  done
done
