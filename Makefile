# ViperGPT reproduction — developer entrypoints.
# Every target is safe to run from a clean clone (except full/figures, which need data+weights).

.PHONY: help setup lint test smoke full figures clean

PKG := vipergpt_repro
PY  := python

help:
	@echo "Targets:"
	@echo "  setup    - create conda env and freeze requirements.lock.txt"
	@echo "  lint     - ruff check on src/ and tests/"
	@echo "  test     - run fast tests (excludes slow/GPU tests)"
	@echo "  smoke    - run the toy end-to-end pipeline on bundled tiny inputs"
	@echo "  full     - reproduce a table (pass EXP=repro_table1, etc.)"
	@echo "  figures  - regenerate all figures in results/figures/"

setup:
	conda env create -f environment.yml || conda env update -f environment.yml
	@echo "Activate with: conda activate vipergpt ; then run: make lock"

lock:
	pip freeze > requirements.lock.txt
	@echo "Wrote requirements.lock.txt"

lint:
	ruff check src tests

test:
	pytest -m "not slow"

smoke:
	$(PY) -m $(PKG).pipeline.run_smoke

full:
	@test -n "$(EXP)" || (echo "Set EXP, e.g. make full EXP=repro_table1_refcoco" && exit 1)
	$(PY) -m $(PKG).pipeline.run_batch experiment=$(EXP)

figures:
	$(PY) -m $(PKG).eval.make_figures

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache
