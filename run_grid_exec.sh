#!/bin/bash
#================================================================
#  GRID EXECUTION — score every generated cell that has not been scored.
#
#      sbatch run_grid_exec.sh
#
#  Discovers generated run directories automatically, so it can be resubmitted
#  after each generation job without editing anything. A cell whose summary
#  already exists is skipped.
#
#  Runs in glip_env; generation runs in vipergpt. The two never coexist in one
#  interpreter, which is why the pipeline is split at programs.jsonl.
#================================================================
#SBATCH --job-name=grid_exec
#SBATCH --partition=gpu
#SBATCH --output=outputs/slurm/%x_%j.out
#SBATCH --error=outputs/slurm/%x_%j.err
#SBATCH --gres=shard:25
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

set -uo pipefail

module load anaconda3-2024.2
module load cuda-12.8
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate glip_env || true
export PATH="/home/mazaveri/.conda/envs/glip_env/bin:$PATH"
export PYTHONNOUSERSITE=1

cd /home/mazaveri/hpc-prog/humayun/vipergpt || exit 1
export HF_HOME=/home/mazaveri/hf_cache
export PYTHONPATH=$PWD/src
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
mkdir -p outputs/slurm results/grid_summaries results/grid_records

echo "============================================================"
echo "  GRID EXECUTION"
echo "  node: $SLURMD_NODENAME   started: $(date)"
echo "============================================================"

python -c "import torch,sys; ok=torch.cuda.is_available(); print('CUDA:',ok); sys.exit(0 if ok else 1)" \
  || { echo "[FATAL] no GPU visible"; exit 1; }

DONE=0; SKIP=0; FAIL=0

# RefCOCO first: it carries the larger spatial subset and the headline numbers.
for DS in refcoco refcoco+; do
  for D in $(ls -1d outputs/runs/*m1_${DS}_testA_*_greedy 2>/dev/null | sort); do
    BASE=$(basename "$D")
    # ..._m1_<dataset>_testA_<model>_<condition>_greedy
    CELL=${BASE#*__*__m1_${DS}_testA_}
    CELL=${CELL%_greedy}
    TAG="grid_${DS}_${CELL}"

    if compgen -G "results/grid_summaries/*${TAG}.json" > /dev/null; then
      echo ">>> SKIP  $TAG"
      SKIP=$((SKIP+1))
      continue
    fi
    if [[ ! -f "$D/programs.jsonl" ]]; then
      echo ">>> SKIP  $TAG  (no programs.jsonl)"
      SKIP=$((SKIP+1))
      continue
    fi

    echo ""
    echo ">>> EXEC  $TAG   $(date '+%H:%M:%S')"
    # 300 s: verify_property over ~40 candidates is roughly 40 CLIP forward passes,
    # which exceeded the original 60 s limit on attribute-heavy RefCOCO+ queries.
    python -m vipergpt_repro.eval.execute_refcoco \
        --programs "$D/programs.jsonl" --version "$DS" \
        --max-samples 500 --timeout 300 --tag "$TAG"
    RC=$?
    if [[ $RC -eq 0 ]]; then DONE=$((DONE+1)); else FAIL=$((FAIL+1)); fi

    NEW=$(ls -1dt outputs/runs/*"${TAG}" 2>/dev/null | head -1)
    if [[ -n "$NEW" ]]; then
      cp "$NEW/summary.json"   "results/grid_summaries/$(basename "$NEW").json" 2>/dev/null
      cp "$NEW/records.jsonl"  "results/grid_records/$(basename "$NEW").jsonl"  2>/dev/null
    fi
  done
done

echo ""
echo "============================================================"
echo "  executed $DONE   skipped $SKIP   failed $FAIL"
echo "  summaries archived: $(ls -1 results/grid_summaries/*.json 2>/dev/null | wc -l)"
echo "  finished: $(date)"
echo "============================================================"
exit $FAIL
