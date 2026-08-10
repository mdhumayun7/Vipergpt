#!/usr/bin/env bash
# [LOGIN NODE ONLY] Download datasets.
#
# SCOPE: RefCOCO / RefCOCO+ only (Table 1, visual grounding).
# GQA / OK-VQA / NExT-QA are out of scope — see README "Scope" and
# docs/deviations.md. NExT-QA alone would need 100 GB+ of video.
#
# Annotations are ~90 MB and are all Milestone 1 needs.
# COCO images (~13 GB) are only needed once the perception modules run,
# so they are behind the --images flag.
#
# Usage:
#   bash scripts/download_data.sh            # annotations only (~90 MB)
#   bash scripts/download_data.sh --images   # + COCO train2014 (~13 GB)
set -euo pipefail

DATA_PATH="${DATA_PATH:-$HOME/hpc-prog/humayun/vipergpt_store/data}"
IMAGES=0
[[ "${1:-}" == "--images" ]] && IMAGES=1

mkdir -p "$DATA_PATH/refcoco"
cd "$DATA_PATH/refcoco"
echo "Data -> $DATA_PATH"

# --- 1. RefCOCO + RefCOCO+ referring-expression annotations --------------------
# DEVIATION D3: the canonical host bvisionweb1.cs.unc.edu is permanently offline
# (DNS no longer resolves, verified 2026-08). We fetch the Internet Archive
# snapshots, which several published repos now use as the de-facto source. The
# archived zips are byte-identical to the originals. Recorded in docs/deviations.md.
declare -A SNAP=(
  [refcoco]="20220413011718"
  [refcoco+]="20220413011656"
)

for V in refcoco refcoco+; do
  if [[ -f "$V/instances.json" ]]; then
    echo ">>> $V already present, skipping"
    continue
  fi
  echo ">>> $V annotations (Internet Archive mirror)"
  URL="https://web.archive.org/web/${SNAP[$V]}/https://bvisionweb1.cs.unc.edu/licheng/referit/data/${V}.zip"
  if ! wget -c --tries=3 --timeout=60 -O "${V}.zip" "$URL"; then
    echo "ABORT: could not fetch $V from the archive mirror." >&2
    echo "  Check the login node can reach web.archive.org:" >&2
    echo "    getent hosts web.archive.org && curl -sI https://web.archive.org | head -1" >&2
    rm -f "${V}.zip"
    exit 1
  fi
  # Guard: a proxy error page would also 'download' successfully.
  if ! unzip -tq "${V}.zip" >/dev/null 2>&1; then
    echo "ABORT: ${V}.zip is not a valid zip — likely an HTML error page." >&2
    file "${V}.zip" >&2
    rm -f "${V}.zip"
    exit 1
  fi
  unzip -q -o "${V}.zip"
  rm -f "${V}.zip"
done

# --- 2. COCO train2014 images (Milestone 2 only) ------------------------------
if [[ $IMAGES -eq 1 ]]; then
  AVAIL_GB=$(df -BG --output=avail "$DATA_PATH" | tail -1 | tr -dc '0-9')
  if (( AVAIL_GB < 30 )); then
    echo "ABORT: ${AVAIL_GB}G free, COCO train2014 needs ~30G to download+extract." >&2
    exit 1
  fi
  mkdir -p "$DATA_PATH/coco"
  cd "$DATA_PATH/coco"
  if [[ ! -d train2014 ]]; then
    echo ">>> COCO train2014 images (~13 GB, slow)"
    wget -c http://images.cocodataset.org/zips/train2014.zip
    unzip -q train2014.zip && rm -f train2014.zip
  fi
fi

# --- 3. validate ---------------------------------------------------------------
echo
echo "=== Validation ==="
for V in refcoco refcoco+; do
  D="$DATA_PATH/refcoco/$V"
  if [[ -f "$D/instances.json" ]]; then
    N=$(python - "$D" <<'PY'
import json, sys, glob, os
d = sys.argv[1]
with open(os.path.join(d, "instances.json")) as f:
    inst = json.load(f)
refs = glob.glob(os.path.join(d, "refs(*).p"))
print(f"images={len(inst['images'])} anns={len(inst['annotations'])} refs_files={len(refs)}")
PY
)
    echo "OK  $V -> $N"
  else
    echo "MISSING  $V/instances.json"
  fi
done
[[ $IMAGES -eq 1 ]] && echo "COCO images: $(ls "$DATA_PATH/coco/train2014" 2>/dev/null | wc -l) files"
du -sh "$DATA_PATH"