#!/usr/bin/env bash
# Let execute_refcoco pick up max_detections / find_nms_iou from the environment, so
# the D12 ablation can sweep them without editing code between runs.
#
# Defaults stay OFF (0), so every previously reported number is unaffected.
set -euo pipefail
cd "$(dirname "$0")"

python - <<'PYEOF'
from pathlib import Path
p = Path("src/vipergpt_repro/eval/execute_refcoco.py")
s = p.read_text()

old = '''        "crop_larger_margin": False,
        "ratio_box_area_to_image_area": 0.0,
        "device": "cuda:0",
    })'''
new = '''        "crop_larger_margin": False,
        "ratio_box_area_to_image_area": 0.0,
        "device": "cuda:0",
        # D12: candidate-set controls, read from the environment so an ablation can
        # sweep them without touching code. Both default to 0 = disabled, which is
        # the configuration every previously reported number was produced under.
        "max_detections": int(os.environ.get("MAX_DET", "0")),
        "find_nms_iou": float(os.environ.get("FIND_NMS", "0")),
    })
    if cfg["max_detections"] or cfg["find_nms_iou"]:
        logger.info("D12 candidate cap active: max_detections=%s find_nms_iou=%s",
                    cfg["max_detections"], cfg["find_nms_iou"])'''
assert old in s, "cfg anchor not found"
s = s.replace(old, new)

# record the setting in the summary so a run cannot be misread later
old2 = '''        "elapsed_s": round(time.time() - t0, 1),
        "git_sha": sha,
    }'''
new2 = '''        "elapsed_s": round(time.time() - t0, 1),
        "git_sha": sha,
        "max_detections": cfg["max_detections"],
        "find_nms_iou": cfg["find_nms_iou"],
    }'''
assert old2 in s, "summary anchor not found"
s = s.replace(old2, new2)
p.write_text(s)
print("patched: MAX_DET / FIND_NMS honoured and recorded")
PYEOF

python -c "import ast; ast.parse(open('src/vipergpt_repro/eval/execute_refcoco.py').read())" && echo "SYNTAX OK"
grep -n "MAX_DET\|max_detections" src/vipergpt_repro/eval/execute_refcoco.py | head
