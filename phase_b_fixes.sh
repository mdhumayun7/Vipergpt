#!/usr/bin/env bash
#================================================================
#  PHASE B — documentation corrections, all in one pass.
#
#      bash phase_b_fixes.sh
#
#  Idempotent: safe to run twice. Reports what it changed and what it left
#  alone. Nothing is deleted without saying so first.
#================================================================
set -uo pipefail
cd "$(dirname "$0")"

echo "============================================================"
echo "  PHASE B — documentation corrections"
echo "============================================================"

# ---------------------------------------------------------------- B0
# A stray file appeared in the tree, almost certainly from a sed invocation
# whose backup suffix was mistyped. Report before touching it.
echo
echo "--- B0  stray files"
if [[ -f docs/research_log.mdo ]]; then
  if diff -q docs/research_log.mdo docs/research_log.md >/dev/null 2>&1; then
    echo "    docs/research_log.mdo is identical to research_log.md -> removing"
    rm -f docs/research_log.mdo
  else
    LINES_MDO=$(wc -l < docs/research_log.mdo)
    LINES_MD=$(wc -l < docs/research_log.md)
    echo "    docs/research_log.mdo DIFFERS from research_log.md"
    echo "      .mdo : $LINES_MDO lines"
    echo "      .md  : $LINES_MD lines"
    echo "    Not deleting. Inspect with: diff docs/research_log.md docs/research_log.mdo"
  fi
else
  echo "    none"
fi

# ---------------------------------------------------------------- B1
echo
echo "--- B1  reproduction table"
if [[ -f make_repro_table.py ]]; then
  python make_repro_table.py 2>&1 | tail -2
else
  echo "    make_repro_table.py not found — skipped"
fi

# There appear to be two copies of the table. Keep results/ as canonical and
# make docs/ a pointer, so they cannot drift apart again.
if [[ -f docs/reproduction_table.md && -f results/reproduction_table.md ]]; then
  if ! diff -q docs/reproduction_table.md results/reproduction_table.md >/dev/null 2>&1; then
    cp results/reproduction_table.md docs/reproduction_table.md
    echo "    docs/reproduction_table.md refreshed from results/ (they had diverged)"
  else
    echo "    docs/ and results/ copies already identical"
  fi
fi

# ---------------------------------------------------------------- B2
echo
echo "--- B2  contribution spec status"
if grep -q "design, not yet implemented" docs/contribution_spec.md 2>/dev/null; then
  sed -i 's|\*\*Status:\*\* design, not yet implemented\.|**Status:** implemented and evaluated; see docs/research_log.md, 2026-08-15 to 08-17.|' \
      docs/contribution_spec.md
  echo "    status line updated"
else
  echo "    already updated"
fi

# ---------------------------------------------------------------- B4
echo
echo "--- B4  perception ceiling"
# 78.9% came from a 20-sample check. The held-out sweep (n=200, samples 200-399,
# disjoint from every reported cell) gives 81.5%. Forward-facing documents must
# use the held-out figure.
#
# The research log is NOT rewritten. It is a dated record of what was known at the
# time, and editing it retroactively would misrepresent the sequence of the work.
# A correction note is appended instead.
for f in results/reproduction_table.md docs/reproduction_table.md docs/progress_report_m1_m2.tex; do
  [[ -f "$f" ]] || continue
  if grep -q "78\.9" "$f"; then
    sed -i 's|78\.9\\%|81.5\\%|g; s|78\.9%|81.5%|g; s|78\.9|81.5|g' "$f"
    sed -i 's|(n=20, threshold 0\.2)|(n=200 held out, threshold 0.2)|g; s|(n=20, thr 0\.2)|(n=200 held out, thr 0.2)|g' "$f"
    echo "    updated: $f"
  fi
done

if ! grep -q "Correction to the perception ceiling" docs/research_log.md 2>/dev/null; then
cat >> docs/research_log.md << 'ENDNOTE'

## Correction to the perception ceiling (recorded, not rewritten)

Entries above quote a raw GLIP top-box accuracy of 78.9% from a 20-sample check. That
figure is superseded by the held-out sweep of 2026-08-12 (job 29318), which measured
81.5% on RefCOCO/testA samples 200-399 — a slice disjoint from every sample on which
results are reported.

The earlier entries are deliberately left as written. A research log records what was
believed at the time; editing it retroactively would misrepresent the order in which
the work happened. All forward-facing documents (the reproduction table, the report,
and Figure 10) use 81.5% with the held-out sample size stated.
ENDNOTE
  echo "    correction note appended to research_log.md (history left intact)"
else
  echo "    correction note already present"
fi

# ---------------------------------------------------------------- B3
echo
echo "--- B3  legacy summary table in deviations.md"
python - <<'PYEOF'
from pathlib import Path
p = Path("docs/deviations.md")
if not p.exists():
    print("    deviations.md not found"); raise SystemExit
s = p.read_text()

if "SUPERSEDED — see the detailed entries" in s:
    print("    already marked"); raise SystemExit

# The legacy table at the top carries placeholder rows such as
# "finalized in plan", which contradict the detailed D1/D2 entries below it.
markers = ["finalized in Phase 3 plan", "finalized in plan", "_tbd_", "TBD"]
if not any(m in s for m in markers):
    print("    no placeholder rows found — nothing to do"); raise SystemExit

lines = s.split("\n")
# Find the first detailed entry; everything before it is the legacy header block.
first = next((i for i, l in enumerate(lines) if l.startswith("## D1 ")), None)
if first is None:
    first = next((i for i, l in enumerate(lines) if l.startswith("## D")), None)
if first is None:
    print("    could not locate the detailed entries — left untouched"); raise SystemExit

head, tail = lines[:first], lines[first:]
note = [
    "",
    "> **Note.** The summary table above is the original planning-stage register and",
    "> retains placeholder wording for some rows. It is **SUPERSEDED — see the detailed",
    "> entries below**, which record what was actually done, with reasons and measured",
    "> effects. Where the two disagree, the detailed entries are authoritative.",
    "",
]
p.write_text("\n".join(head + note + tail))
print(f"    superseded-note inserted before line {first+1}")
PYEOF

# ---------------------------------------------------------------- B5
echo
echo "--- B5  new deviations D13, D14"
if grep -q "^## D13" docs/deviations.md 2>/dev/null; then
  echo "    D13/D14 already present"
else
cat >> docs/deviations.md << 'ENDDEV'

## D13 — crop_larger_margin disabled
**Paper / official configuration:** `crop_larger_margin: true`, expanding each detection
by 10%.
**Used instead:** `False` in every evaluation configuration.
**Reason:** not a deliberate choice. The evaluation scripts build their configuration
inline rather than reading `configs/default.yaml`, and the inline default is `False`.
Identified during the audit of the dissertation, after all results had been produced.
**Effect on results:** unmeasured. A 10% margin raises IoU on tight detections and
lowers it on loose ones, so the direction is not predictable without running it.
Recorded as an open item rather than as a controlled decision.

## D14 — Evaluation configuration is not read from configs/default.yaml
**Repository principle:** "ALL hyperparameters live in `configs/` — never hardcode them
in `src/`."
**Actual behaviour:** `eval/execute_refcoco.py` and the analysis scripts construct their
configuration inline. `configs/default.yaml` therefore describes no run that was
actually performed. It lists `xvlm`, `blip2` and `llm_qa` as loaded (none were),
`crop_larger_margin: true` (False was used), `max_new_tokens: 512` (320 was used) and
`batch_size: 20` (8 was used).
**Reason:** the two-environment split. The execution stage runs in `glip_env`, which
does not have `omegaconf`, so the runner cannot load the project's configuration
objects.
**Effect on results:** none. The inline values are what ran and are recorded in every
run summary. The file is nonetheless misleading to a reader and should be either
corrected to match, or removed in favour of the per-run summaries.
ENDDEV
  echo "    D13 and D14 appended"
fi

# ---------------------------------------------------------------- figures
echo
echo "--- figures"
BEFORE=$(ls -1 results/figures/*.pdf 2>/dev/null | wc -l)
for s in make_arch_figures.py make_comparison_table.py; do
  [[ -f "$s" ]] && python "$s" >/dev/null 2>&1 && echo "    ran $s"
done
AFTER=$(ls -1 results/figures/*.pdf 2>/dev/null | wc -l)
echo "    figures: $BEFORE -> $AFTER"
if (( AFTER < 16 )); then
  echo "    NOTE: fewer than 16. Missing generators are likely make_arch_figures.py"
  echo "          and/or make_comparison_table.py — upload them and re-run."
fi

# ---------------------------------------------------------------- verify
echo
echo "============================================================"
echo "  VERIFICATION"
echo "============================================================"
printf "  deviations D-entries : %s\n" "$(grep -c '^## D' docs/deviations.md 2>/dev/null)"
printf "  reproduction table   : %s lines\n" "$(wc -l < results/reproduction_table.md 2>/dev/null)"
printf "  research log         : %s lines\n" "$(wc -l < docs/research_log.md 2>/dev/null)"
printf "  figures              : %s\n" "$AFTER"
echo
echo "  remaining 78.9 references (research log only is expected):"
grep -rln "78\.9" docs/ results/ 2>/dev/null | sed 's/^/    /' || echo "    none"

echo
echo "  Review the above, then:"
echo "    git add -A"
echo "    git commit -m \"docs: phase B corrections — D13, D14, ceiling, stale files\""
echo "    git push"
