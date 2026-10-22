"""Ligand-property confound analysis for the docking comparison.

A strict reviewer may object that predicted and decoy compounds differ in size or
lipophilicity, so that a Vina-score difference could reflect ligand properties
rather than target complementarity.  This script computes RDKit descriptors for
all docked compounds, ligand efficiencies (LE = -affinity / heavy-atom count),
and residual affinities after regressing affinity on molecular weight and
logP.  It then repeats the predicted-versus-decoy comparison on these adjusted
quantities.
"""
from __future__ import annotations

import csv
import json
import os
import re
from collections import defaultdict

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
from scipy.stats import mannwhitneyu

RDLogger.DisableLog("rdApp.*")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCK = os.path.join(ROOT, "data", "docking")
RES = os.path.join(ROOT, "results")


def find_sdf(name):
    target = re.sub(r"[^a-z0-9]", "", name.lower())
    for fn in os.listdir(DOCK):
        if fn.lower().endswith(".sdf") and re.sub(r"[^a-z0-9]", "", fn[:-4].lower()) == target:
            return os.path.join(DOCK, fn)
    return None


def descriptors(name):
    path = find_sdf(name)
    if not path:
        return None
    mol = Chem.MolFromMolFile(path)
    if mol is None:
        return None
    Chem.SanitizeMol(mol)
    mw = Descriptors.MolWt(mol)
    hac = mol.GetNumHeavyAtoms()
    logp = Crippen.MolLogP(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    return {"mw": float(mw), "heavy_atoms": int(hac), "logp": float(logp),
            "tpsa": float(tpsa), "rotatable": int(rot), "hbd": int(hbd), "hba": int(hba)}


def main():
    runs = json.load(open(os.path.join(DOCK, "docking_runs.json"), encoding="utf-8"))
    valid = []
    for r in runs:
        a = r.get("best_affinity")
        if a is None or a >= 0:
            continue
        valid.append(r)
    # de-duplicate predicted pairs
    seen, rows = set(), []
    for r in valid:
        if r["set"] == "predicted":
            key = (r["compound"], r["target"])
            if key in seen:
                continue
            seen.add(key)
        rows.append(r)
    props = {}
    for r in rows:
        if r["compound"] not in props:
            props[r["compound"]] = descriptors(r["compound"])
    table = []
    for r in rows:
        p = props.get(r["compound"]) or {}
        row = {"set": r["set"], "compound": r["compound"], "target": r["target"],
               "affinity": float(r["best_affinity"]), **p}
        if p.get("heavy_atoms"):
            row["ligand_efficiency"] = -row["affinity"] / p["heavy_atoms"]
        table.append(row)
    # properties by compound
    compounds = []
    for name, p in props.items():
        sets = sorted({r["set"] for r in table if r["compound"] == name})
        compounds.append({"compound": name, "set": sets[0] if sets else "", **p})
    pred = [r for r in table if r["set"] == "predicted"]
    dec = [r for r in table if r["set"] == "decoy"]
    out = {"compounds": compounds,
           "predicted_compound_properties": [c for c in compounds if c["set"] == "predicted"],
           "decoy_compound_properties": [c for c in compounds if c["set"] == "decoy"],
           "runs": table}
    # raw comparison
    pa = [r["affinity"] for r in pred]
    da = [r["affinity"] for r in dec]
    out["raw"] = {"n_pred": len(pa), "n_decoy": len(da),
                  "pred_mean": float(np.mean(pa)), "decoy_mean": float(np.mean(da)),
                  "mannwhitney_p": float(mannwhitneyu(pa, da, alternative="less").pvalue)}
    # ligand efficiency comparison
    ple = [r["ligand_efficiency"] for r in pred if "ligand_efficiency" in r]
    dle = [r["ligand_efficiency"] for r in dec if "ligand_efficiency" in r]
    out["ligand_efficiency"] = {"n_pred": len(ple), "n_decoy": len(dle),
                                "pred_mean": float(np.mean(ple)), "decoy_mean": float(np.mean(dle)),
                                "mannwhitney_p": float(mannwhitneyu(ple, dle, alternative="greater").pvalue)}
    # residual affinity after linear adjustment for MW and logP (fit on all runs)
    X, y = [], []
    for r in table:
        if all(k in r for k in ("mw", "logp")):
            X.append([1.0, r["mw"], r["logp"]])
            y.append(r["affinity"])
    X, y = np.array(X), np.array(y)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    pred_res = [resid[i] for i, r in enumerate([q for q in table if all(k in q for k in ("mw", "logp"))]) if r["set"] == "predicted"]
    dec_res = [resid[i] for i, r in enumerate([q for q in table if all(k in q for k in ("mw", "logp"))]) if r["set"] == "decoy"]
    out["mw_logp_adjusted"] = {"beta": [float(b) for b in beta],
                               "pred_mean_residual": float(np.mean(pred_res)),
                               "decoy_mean_residual": float(np.mean(dec_res)),
                               "mannwhitney_p": float(mannwhitneyu(pred_res, dec_res, alternative="less").pvalue)}
    # correlations
    out["correlations"] = {}
    for var in ["mw", "heavy_atoms", "logp", "tpsa"]:
        xs = [r[var] for r in table if var in r]
        ys = [r["affinity"] for r in table if var in r]
        out["correlations"][var] = float(np.corrcoef(xs, ys)[0, 1])
    with open(os.path.join(RES, "ligand_efficiency.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    # CSV for supplementary
    with open(os.path.join(RES, "ligand_properties.csv"), "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["set", "compound", "target", "affinity",
                                                "mw", "heavy_atoms", "logp", "tpsa",
                                                "rotatable", "hbd", "hba", "ligand_efficiency"])
        writer.writeheader()
        writer.writerows(table)
    print(json.dumps({k: v for k, v in out.items() if k != "runs"}, indent=2))


if __name__ == "__main__":
    main()
