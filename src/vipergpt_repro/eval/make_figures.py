"""Regenerate publication-quality figures into results/figures/ (vector PDF).

    python -m vipergpt_repro.eval.make_figures

Reads reproduced numbers (once available) and plots reported-vs-reproduced per
table. Uses explicit styling (no default matplotlib theme), legible fonts, vector
output — regenerable and thesis-ready.
"""
from __future__ import annotations

import logging
from pathlib import Path

from vipergpt_repro.utils.paths import RESULTS

logger = logging.getLogger(__name__)

# Reported numbers from the paper (Tables 1-4). Reproduced filled in Phase 6.
REPORTED = {
    "RefCOCO (mIoU)": 72.0,
    "RefCOCO+ (mIoU)": 67.0,
    "GQA (acc)": 48.1,
    "OK-VQA (acc)": 51.9,
    "NExT-QA Hard-T": 49.8,
    "NExT-QA Hard-C": 56.4,
    "NExT-QA Full": 60.0,
}


def plot_reported_vs_reproduced(reproduced: dict[str, float] | None = None, out_dir: Path | None = None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir) if out_dir else RESULTS / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    reproduced = reproduced or {}

    labels = list(REPORTED.keys())
    rep = [REPORTED[k] for k in labels]
    got = [reproduced.get(k, float("nan")) for k in labels]

    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar([i - 0.2 for i in x], rep, width=0.4, label="Reported", color="#4C72B0")
    ax.bar([i + 0.2 for i in x], got, width=0.4, label="Reproduced", color="#DD8452")
    ax.set_ylabel("Score (%)")
    ax.set_title("ViperGPT reproduction: reported vs. reproduced")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.legend(frameon=False)
    fig.tight_layout()
    out = out_dir / "reported_vs_reproduced.pdf"
    fig.savefig(out)
    logger.info("wrote %s", out)
    return out


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    plot_reported_vs_reproduced()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
