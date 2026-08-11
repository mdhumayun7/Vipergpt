import ast
import glob
import json

POS = {"left","right","upper","lower","horizontal_center","vertical_center",
       "compute_depth","distance","overlaps_with","width","height","area"}
def returns_patch(code):
    try: t = ast.parse(code)
    except SyntaxError: return None
    kinds = []
    for n in ast.walk(t):
        if isinstance(n, ast.Return) and n.value is not None:
            v = n.value
            if isinstance(v, ast.Constant) and isinstance(v.value, str): kinds.append("str")
            elif isinstance(v, ast.Call):
                f = v.func
                nm = f.attr if isinstance(f, ast.Attribute) else getattr(f,"id","")
                kinds.append("str" if nm in {"simple_query","llm_query","best_text_match"} else "other")
            elif isinstance(v, ast.Name): kinds.append("patch?")
            else: kinds.append("other")
    return kinds
def uses_spatial(code):
    return any(p in code for p in POS)
for path in sorted(glob.glob("outputs/runs/*m1_*/programs.jsonl")):
    recs = [json.loads(l) for l in open(path)]
    if not recs: continue
    sp = [r for r in recs if r["is_spatial"]]
    viol = sum(1 for r in recs if (k:=returns_patch(r["program"])) and "str" in k)
    drop = sum(1 for r in sp if not uses_spatial(r["program"]))
    print(f"\n{path.split('/')[-2]}")
    print(f"  n={len(recs)}  spatial={len(sp)}")
    print(f"  returns a STRING (task-invalid) : {100*viol/len(recs):.1f}%")
    print(f"  spatial query, NO positional op : {100*drop/len(sp):.1f}%" if sp else "")
