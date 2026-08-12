"""Milestone 1 - program-generation analysis. Needs only the code generator."""
from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import pickle
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

logger = logging.getLogger("m1")

RELATION_TERMS = ["left","right","behind","in front","front","above","below",
    "under","underneath","on top","top","beneath","next to","beside","near",
    "nearest","closest","farthest","closer","between","around","middle",
    "center","corner","side","back","far","facing","over","across"]
_REL = re.compile(r"\b(" + "|".join(re.escape(t) for t in RELATION_TERMS) + r")\b", re.I)

API_CALLS = {"find","exists","verify_property","best_text_match","best_image_match",
    "simple_query","compute_depth","distance","llm_query","crop","overlaps_with"}
OPAQUE = {"simple_query","llm_query"}

def is_spatial(q): return bool(_REL.search(q))

def load_queries(root, version, split, limit):
    vdir = Path(root) / "refcoco" / version
    files = sorted(vdir.glob("refs(*).p"), key=lambda f: ("unc" not in f.name, f.name))
    if not files: raise FileNotFoundError(f"No refs(*).p under {vdir}")
    with open(files[0], "rb") as f: refs = pickle.load(f)
    out = []
    for r in refs:
        if r.get("split") != split: continue
        for s in r["sentences"]:
            out.append({"ref_id": r["ref_id"], "query": s["sent"].strip()})
            if limit and len(out) >= limit: return out
    return out

def parse_program(code):
    res = {"parses": False, "calls": [], "n_api": 0, "collapsed": False}
    try: tree = ast.parse(code)
    except SyntaxError: return res
    res["parses"] = True
    calls = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            f = n.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
            if name: calls.append(name)
    api = [c for c in calls if c in API_CALLS]
    res["calls"] = calls
    res["n_api"] = len(api)
    res["collapsed"] = bool(api) and all(c in OPAQUE for c in api)
    return res

def extract(text):
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    if m: text = m.group(1)
    i = text.find("def execute_command")
    if i != -1: text = text[i:]
    return text.strip()

def summarise(recs):
    def rate(sub, k):
        return round(100.0*sum(1 for r in sub if r[k])/len(sub), 1) if sub else 0.0
    out = {"n_total": len(recs)}
    for label, sub in [("all", recs),
                       ("spatial", [r for r in recs if r["is_spatial"]]),
                       ("non_spatial", [r for r in recs if not r["is_spatial"]])]:
        ok = [r for r in sub if r["parses"]]
        out[label] = {"n": len(sub), "parse": rate(sub, "parses"),
            "collapse": rate(ok, "collapsed"),
            "mean_api": round(sum(r["n_api"] for r in ok)/len(ok), 2) if ok else 0.0}
    c = Counter()
    for r in recs: c.update(x for x in r["calls"] if x in API_CALLS)
    out["freq"] = dict(c.most_common())
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--version", default="refcoco")
    p.add_argument("--split", default="testA")
    p.add_argument("--max-samples", type=int, default=500)
    p.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    p.add_argument("--prompt", default="prompts/api.prompt")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=320)
    p.add_argument("--data-root", default=os.environ.get("DATA_PATH",
        str(Path.home()/"hpc-prog/humayun/vipergpt_store/data")))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.0,
                   help="0.0 = greedy (paper setting). >0 enables sampling for variance estimates.")
    a = p.parse_args()

    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
        format="%(asctime)s | %(levelname)-7s | %(message)s")

    try:
        sha = subprocess.check_output(["git","rev-parse","--short","HEAD"], text=True).strip()
        if subprocess.check_output(["git","status","--porcelain"], text=True).strip():
            sha += "-dirty"
    except Exception:
        sha = "nogit"
    import random
    import numpy as _np
    random.seed(a.seed)
    _np.random.seed(a.seed)
    try:
        import torch as _t
        _t.manual_seed(a.seed)
        _t.cuda.manual_seed_all(a.seed)
    except Exception:
        pass
    _dec = "greedy" if a.temperature == 0 else f"t{a.temperature}s{a.seed}"
    # model + prompt in the run name: the sweep runs several models under the same
    # decoding, and without this they would all collide on "..._greedy".
    _m = a.model.split("/")[-1].replace("Qwen2.5-Coder-", "").replace("-Instruct", "")
    _p = Path(a.prompt).stem.replace("api_", "").replace("api", "base")
    _dec = f"{_m}_{_p}_{_dec}"
    run_dir = Path("outputs/runs")/f"{time.strftime('%Y%m%dT%H%M%S')}__{sha}__m1_{a.version}_{a.split}_{_dec}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Run dir: %s", run_dir)

    samples = load_queries(a.data_root, a.version, a.split, a.max_samples)
    if not samples:
        logger.error("No queries. Try --split val / testB.")
        return 1
    ns = sum(is_spatial(s["query"]) for s in samples)
    logger.info("Loaded %d queries. Spatial: %d (%.1f%%)", len(samples), ns, 100.0*ns/len(samples))
    for s in samples[:8]:
        logger.info("  %-45r spatial=%s", s["query"], is_spatial(s["query"]))
    if a.dry_run:
        logger.info("DRY RUN done.")
        return 0

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tpl = Path(a.prompt).read_text()
    if "INSERT_QUERY_HERE" not in tpl:
        logger.error("Prompt has no INSERT_QUERY_HERE placeholder")
        return 1

    logger.info("Loading %s ...", a.model)
    tok = AutoTokenizer.from_pretrained(a.model, padding_side="left")
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="auto")
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    logger.info("Loaded on %s", model.device)

    recs = []
    qs = [s["query"] for s in samples]
    fh = open(run_dir/"programs.jsonl", "w")
    for i in range(0, len(qs), a.batch_size):
        chunk = qs[i:i+a.batch_size]
        prompts = [tok.apply_chat_template(
            [{"role":"user","content": tpl.replace("INSERT_QUERY_HERE", q)}],
            tokenize=False, add_generation_prompt=True) for q in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            gen_kw = dict(max_new_tokens=a.max_new_tokens, pad_token_id=tok.pad_token_id)
            if a.temperature > 0:
                # Sampling path: used only for variance estimates. The paper runs
                # Codex at temperature 0, so greedy remains the headline setting.
                gen_kw.update(do_sample=True, temperature=a.temperature, top_p=0.95)
            else:
                gen_kw.update(do_sample=False)
            gen = model.generate(**enc, **gen_kw)
        for j, q in enumerate(chunk):
            txt = tok.decode(gen[j][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            prog = extract(txt)
            r = {"query": q, "program": prog, "is_spatial": is_spatial(q), **parse_program(prog)}
            recs.append(r)
            fh.write(json.dumps(r) + "\n")
        fh.flush()
        logger.info("  %d/%d", min(i+a.batch_size, len(qs)), len(qs))
    fh.close()

    s = summarise(recs)
    meta = {"model": a.model, "version": a.version, "split": a.split, "sha": sha}
    (run_dir/"analysis.json").write_text(json.dumps({"meta": meta, "summary": s}, indent=2))

    lines = ["# Milestone 1 - program generation analysis", "",
        f"- Generator: {a.model} (D1: replaces Codex)",
        f"- Dataset: {a.version}/{a.split}, {s['n_total']} queries",
        f"- Run: {run_dir}  git {sha}", "",
        "| Subset | N | Parse % | Collapse % | Mean API calls |", "|---|---:|---:|---:|---:|"]
    for k in ("all","spatial","non_spatial"):
        d = s[k]
        lines.append(f"| {k} | {d['n']} | {d['parse']} | {d['collapse']} | {d['mean_api']} |")
    lines += ["", "Collapse = every API call is opaque (simple_query/llm_query).",
        "", "| Call | Count |", "|---|---:|"]
    for k, v in s["freq"].items(): lines.append(f"| {k} | {v} |")
    (run_dir/"analysis.md").write_text("\n".join(lines) + "\n")

    print("\n" + "="*60)
    for k in ("all","spatial","non_spatial"):
        d = s[k]
        print(f"  {k:<12} n={d['n']:<5} parse={d['parse']:>5}%  collapse={d['collapse']:>5}%  api={d['mean_api']}")
    print("="*60)
    print(f"  Written: {run_dir}/analysis.md")
    return 0

if __name__ == "__main__":
    sys.exit(main())
