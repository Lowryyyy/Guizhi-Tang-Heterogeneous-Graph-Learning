"""
Real AutoDock Vina docking with a decoy comparison for the revision.

The original submission reported 26 compound-target pairs "all below -5.0
kcal/mol" but did not provide grid boxes, decoys or raw outputs.  Here docking
is repeated from scratch for the subset of predicted targets that have a
structurally defined ligand-binding site (a co-crystallised drug-like ligand,
the ATP site, or the catalytic zinc site).  Targets without such a site are
explicitly excluded rather than docked blindly.

Predicted compound-target pairs (from the revised HetGNN predictions) are
compared with a decoy set of compound-target pairs that are not predicted by the
model.  This provides the discriminative-power control requested by the
reviewers.

Outputs
-------
data/docking/docking_runs.json   one record per Vina run (best affinity, box)
results/docking_results.json     processed predicted/decoy comparison + AUC
results/docking_grid_boxes.json  grid boxes used for Table S1
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCK = os.path.join(ROOT, "data", "docking")
PREP = os.path.join(DOCK, "prep")
RESULTS = os.path.join(ROOT, "results")
VINA = os.path.join(ROOT, "tools", "vina_1.2.7_win.exe")
OBABEL = r"C:\Python314\Scripts\obabel.exe"

# target -> (PDB, box center, box size, justification)
TARGETS = {
    "AKT1":  ("4EKL", (28.03, 5.22, 10.89), 25, "co-crystallised inhibitor 0RF (allosteric pocket)"),
    "BCL2":  ("4LVT", (7.63, -2.99, -8.72), 25, "co-crystallised inhibitor 1XJ (BH3 groove)"),
    "CASP3": ("1NME", (42.09, 96.34, 24.13), 25, "co-crystallised inhibitor 159 (active site)"),
    "CDK2":  ("1HCK", (100.54, 97.89, 81.72), 25, "ATP-binding site (ATP ligand centroid)"),
    "SRC":   ("1Y57", (18.36, 35.44, 39.77), 25, "co-crystallised inhibitor MPZ (kinase hinge)"),
    "PTGS2": ("5KIR", (23.21, 1.32, 34.26), 25, "co-crystallised inhibitor RCX (COX-2 active site)"),
    "TNF":   ("2AZ5", (-19.41, 74.65, 33.85), 25, "co-crystallised small-molecule inhibitor 307"),
    "HRAS":  ("5P21", (11.79, 33.43, 19.66), 25, "GppNHp nucleotide-binding pocket"),
    "MMP2":  ("1QIB", (64.27, 29.45, 19.56), 25, "catalytic zinc site (centroid of catalytic Zn)"),
}

DECOYS = ["Amantadine", "Amitriptyline", "Ciprofloxacin", "Reserpine"]


def fetch_pubchem_sdf(name, cid=None):
    dest = os.path.join(DOCK, re.sub(r"[^A-Za-z0-9]+", "_", name) + ".sdf")
    if os.path.exists(dest) and os.path.getsize(dest) > 500:
        return dest
    if cid is None:
        cid = None
        for nm in (name, name.replace("-", " "), name.replace(" ", "-")):
            url = ("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
                   + urllib.parse.quote(nm) + "/cids/TXT")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            try:
                cid = urllib.request.urlopen(req, timeout=60).read().decode().strip().splitlines()[0]
                break
            except Exception:
                continue
        if cid is None:
            raise RuntimeError(f"PubChem name lookup failed for {name}")
    url = (f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/record/"
           "SDF?record_type=3d")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    data = urllib.request.urlopen(req, timeout=90).read()
    open(dest, "wb").write(data)
    return dest


def prep_ligand(name):
    sdf = fetch_pubchem_sdf(name)
    out = os.path.join(PREP, re.sub(r"[^A-Za-z0-9]+", "_", name) + ".pdbqt")
    if os.path.exists(out) and os.path.getsize(out) > 500:
        return out
    cmd = [OBABEL, sdf, "-O", out, "--partialcharge", "gasteiger", "-h"]
    subprocess.run(cmd, check=True, capture_output=True)
    return out


def prep_receptor(target):
    pdb, center, size, why = TARGETS[target]
    src = os.path.join(DOCK, pdb + ".pdb")
    rec_pdb = os.path.join(PREP, f"{target}_{pdb}_receptor.pdb")
    rec_pdbqt = os.path.join(PREP, f"{target}_{pdb}_receptor.pdbqt")
    if not os.path.exists(rec_pdbqt) or os.path.getsize(rec_pdbqt) < 1000:
        keep_het = set()
        if target == "MMP2":
            keep_het = {"ZN"}
        with open(src, encoding="utf-8") as fi, open(rec_pdb, "w", encoding="utf-8") as fo:
            for line in fi:
                if line.startswith(("ATOM", "TER", "END")):
                    fo.write(line)
                elif line.startswith("HETATM") and line[17:20].strip() in keep_het:
                    fo.write(line)
        subprocess.run([OBABEL, rec_pdb, "-O", rec_pdbqt, "-xr", "-p", "7.4",
                        "--partialcharge", "gasteiger"], check=True, capture_output=True)
    if center is None:
        # catalytic zinc centroid for MMP2
        xs, ys, zs = [], [], []
        for line in open(src, encoding="utf-8"):
            if line.startswith("HETATM") and line[17:20].strip() == "ZN":
                xs.append(float(line[30:38])); ys.append(float(line[38:46])); zs.append(float(line[46:54]))
        center = (sum(xs) / len(xs), sum(ys) / len(ys), sum(zs) / len(zs))
    return rec_pdbqt, center, size, why


def run_vina(rec_pdbqt, lig_pdbqt, center, size, out_name, exhaustiveness=8, seed=42):
    out_pdbqt = os.path.join(PREP, out_name + "_out.pdbqt")
    log = os.path.join(PREP, out_name + ".log")
    cmd = [VINA, "--receptor", rec_pdbqt, "--ligand", lig_pdbqt,
           "--center_x", f"{center[0]:.3f}", "--center_y", f"{center[1]:.3f}",
           "--center_z", f"{center[2]:.3f}", "--size_x", str(size), "--size_y", str(size),
           "--size_z", str(size), "--exhaustiveness", str(exhaustiveness), "--seed", str(seed),
           "--num_modes", "9", "--out", out_pdbqt]
    affinities = []
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, timeout=900, text=True)
        stdout = proc.stdout
    except Exception as exc:
        return {"error": str(exc)[:200]}
    with open(log, "w", encoding="utf-8") as fh:
        fh.write(stdout)
    for line in stdout.splitlines():
        m = re.match(r"^\s+(\d+)\s+(-?\d+\.?\d*)", line)
        if m:
            affinities.append(float(m.group(2)))
    return {"best_affinity": affinities[0] if affinities else None,
            "n_modes": len(affinities), "all_affinities": affinities}


def main():
    os.makedirs(PREP, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)
    pairs_path = os.path.join(DOCK, "docking_pairs.json")
    if not os.path.exists(pairs_path):
        raise SystemExit("Docking pairs file not found; run make_docking_inputs.py first.")
    pairs = json.load(open(pairs_path, encoding="utf-8"))
    predicted = pairs["predicted"]
    decoy_pairs = pairs["decoy"]
    runs = []
    for target in sorted({p["target"] for p in predicted} | {p["target"] for p in decoy_pairs}):
        if target not in TARGETS:
            print("skip target without structure:", target)
            continue
        rec, center, size, why = prep_receptor(target)
        for p in [x for x in predicted if x["target"] == target]:
            try:
                lig = prep_ligand(p["compound"])
            except Exception as exc:
                print("ligand prep failed", p["compound"], exc); continue
            res = run_vina(rec, lig, center, size, f"pred_{p['compound']}_{target}".replace(" ", "_"))
            runs.append({**p, "pdb": TARGETS[target][0], "center": center, "size": size,
                         "pocket": why, "set": "predicted", **res})
            print("pred", p["compound"], target, res.get("best_affinity"))
        for p in [x for x in decoy_pairs if x["target"] == target]:
            try:
                lig = prep_ligand(p["compound"])
            except Exception as exc:
                print("ligand prep failed", p["compound"], exc); continue
            res = run_vina(rec, lig, center, size, f"decoy_{p['compound']}_{target}".replace(" ", "_"))
            runs.append({**p, "pdb": TARGETS[target][0], "center": center, "size": size,
                         "pocket": why, "set": "decoy", **res})
            print("decoy", p["compound"], target, res.get("best_affinity"))
    with open(os.path.join(DOCK, "docking_runs.json"), "w", encoding="utf-8") as fh:
        json.dump(runs, fh, indent=2)
    analyse(runs)


def analyse(runs):
    from scipy.stats import mannwhitneyu
    pred = [r["best_affinity"] for r in runs if r["set"] == "predicted" and r.get("best_affinity") is not None]
    dec = [r["best_affinity"] for r in runs if r["set"] == "decoy" and r.get("best_affinity") is not None]
    out = {"n_predicted": len(pred), "n_decoy": len(dec),
           "predicted_mean": float(sum(pred) / len(pred)) if pred else None,
           "decoy_mean": float(sum(dec) / len(dec)) if dec else None,
           "predicted_median": float(sorted(pred)[len(pred) // 2]) if pred else None,
           "decoy_median": float(sorted(dec)[len(dec) // 2]) if dec else None,
           "predicted_min": min(pred) if pred else None, "predicted_max": max(pred) if pred else None,
           "decoy_min": min(dec) if dec else None, "decoy_max": max(dec) if dec else None,
           "hit_rate_predicted_6": float(sum(a <= -6.0 for a in pred) / len(pred)) if pred else None,
           "hit_rate_decoy_6": float(sum(a <= -6.0 for a in dec) / len(dec)) if dec else None,
           "hit_rate_predicted_5": float(sum(a <= -5.0 for a in pred) / len(pred)) if pred else None,
           "hit_rate_decoy_5": float(sum(a <= -5.0 for a in dec) / len(dec)) if dec else None}
    if pred and dec:
        try:
            u = mannwhitneyu(pred, dec, alternative="less")
            out["mannwhitney_p"] = float(u.pvalue)
        except Exception:
            out["mannwhitney_p"] = None
    # AUC of using -affinity to separate predicted from decoy pairs
    try:
        labels = [1] * len(pred) + [0] * len(dec)
        scores = [-a for a in pred + dec]
        from sklearn.metrics import roc_auc_score
        out["auc_predicted_vs_decoy"] = float(roc_auc_score(labels, scores))
    except Exception:
        out["auc_predicted_vs_decoy"] = None
    out["runs"] = runs
    with open(os.path.join(RESULTS, "docking_results.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    boxes = {}
    for target, (pdb, center, size, why) in TARGETS.items():
        boxes[target] = {"pdb": pdb, "center": center, "size_A": [size, size, size], "pocket": why}
    with open(os.path.join(RESULTS, "docking_grid_boxes.json"), "w", encoding="utf-8") as fh:
        json.dump(boxes, fh, indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "runs"}, indent=2))


if __name__ == "__main__":
    main()




