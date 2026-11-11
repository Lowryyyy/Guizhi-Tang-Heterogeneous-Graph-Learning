"""Literature-based external benchmark for the Guizhi Tang predictions.

Builds a benchmark set of compound-disease associations that have published
experimental (in vitro/in vivo) support in the open literature, then measures
where the five-seed HetGNN ranks those associations among all 77 x 31 = 2,387
compound-disease pairs, together with AUC and permutation statistics against
randomly sampled pairs.
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy.stats import mannwhitneyu

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")

# (compound, disease, reference key, evidence type, note)
EVIDENCE = [
    ("Kaempferol", "Gastric ulcer", "seabuckthorn2025",
     "in vitro (constituent-level)", "Kaempferol-containing flavonoid fraction inhibits Helicobacter pylori; no dedicated kaempferol gastric-ulcer trial was found, so this association is marked as indirect"),
    ("Kaempferol", "COVID-19", "redox2026",
     "in vivo (lung injury)", "Kaempferol protects against mechanical and infectious lung injury and endothelial dysfunction"),
    ("Kaempferol", "COVID-19", "flavonol2026",
     "in silico (MPro docking)", "Flavonols including kaempferol dock the SARS-CoV-2 main protease with molecular dynamics"),
    ("Formononetin", "Rheumatoid arthritis", "formononetin2026",
     "in vitro (RA synoviocytes)", "Formononetin inhibits TNF-alpha-induced HFLS-RA cells"),
    ("Kaempferol", "Liver cirrhosis", "kaempferol_fibrosis2025",
     "transcriptomics + pharmacology", "Kaempferol anti-hepatic-fibrosis mechanism via transcriptomics-guided analysis"),
    ("Isoliquiritigenin", "Hypertension", "isoliq2024",
     "in vivo (endothelial function)", "Isoliquiritigenin protects endothelial function in diabetic mice (vascular protection)"),
    ("6-Gingerol", "Breast cancer", "leehs2008",
     "in vitro (cancer cells)", "6-Gingerol inhibits MDA-MB-231 breast cancer cell metastasis"),
    ("6-Gingerol", "Lung cancer", "yao2018",
     "in vivo (mouse model)", "6-Gingerol prevents urethane-induced lung carcinogenesis"),
    ("Kaempferol", "Pancreatic cancer", "wangf2021",
     "in vitro (cancer cells)", "Kaempferol induces ROS-dependent apoptosis in pancreatic cancer cells"),
    ("Cinnamaldehyde", "Lung cancer", "tian2017",
     "in vitro (cancer cells)", "Cinnamaldehyde induces apoptosis in non-small cell lung cancer"),
    ("Paeoniflorin", "Rheumatoid arthritis", "zhang2020pae",
     "review of experimental studies", "Anti-inflammatory and immunoregulatory effects of paeoniflorin"),
    ("Kaempferol", "Alzheimer disease", "kaempferol_ad2026",
     "in vitro/in vivo (microglia)", "Kaempferol attenuates microglial activation in Alzheimer's disease via HK2/NF-kappaB"),
    ("Quercetin", "Sepsis", "liy2016",
     "review of experimental studies", "Quercetin anti-inflammatory and immunomodulatory effects relevant to sepsis"),
    ("Licochalcone A", "Coronary artery disease", "licochA2019",
     "in vivo (mouse model)", "Licochalcone A attenuates angiotensin II-induced abdominal aortic aneurysm"),
    ("Calycosin", "Rheumatoid arthritis", "calycosin2026",
     "in vivo (mouse model)", "Calycosin co-delivered nanomicelles ameliorate rheumatoid arthritis via macrophage polarisation"),
    ("6-Gingerol", "Hypertension", "zingiber2020",
     "ex vivo (SHR aorta)", "Ginger rhizome extract relaxes aorta of spontaneously hypertensive rats"),
]


def main():
    data = json.load(open(os.path.join(ROOT, "data", "processed", "graph_data.json"), encoding="utf-8"))
    compounds = data["nodes"]["compound"]
    diseases = data["nodes"]["disease"]
    scores = np.load(os.path.join(RES, "final_score_matrix.npy"))
    assert scores.shape == (len(compounds), len(diseases)), scores.shape
    labels = set(tuple(x) for x in data["labels"])
    unlabelled_mask = np.ones(scores.shape, dtype=bool)
    for i, j in labels:
        unlabelled_mask[i, j] = False
    n_unlabelled = int(unlabelled_mask.sum())

    rows = []
    for compound, disease, ref, etype, note in EVIDENCE:
        ci = compounds.index(compound)
        di = diseases.index(disease)
        s = float(scores[ci, di])
        is_label = (ci, di) in labels
        if is_label:
            rows.append({"compound": compound, "disease": disease, "ref": ref,
                         "evidence": etype, "note": note, "score": round(s, 4),
                         "rank": None, "percentile": None, "status": "training label (excluded)"})
            continue
        rank = int(1 + ((scores > s) & unlabelled_mask).sum())
        rows.append({"compound": compound, "disease": disease, "ref": ref,
                     "evidence": etype, "note": note, "score": round(s, 4),
                     "rank": rank, "percentile": round(rank / n_unlabelled, 4),
                     "status": "predicted"})

    all_pairs = int(scores.size)
    free_rows = [r for r in rows if r["rank"] is not None]
    ev_scores = np.array([r["score"] for r in free_rows])
    top20 = json.load(open(os.path.join(RES, "top_predictions.json"), encoding="utf-8"))
    top20_pairs = {(t["compound"], t["disease"]) for t in top20}
    in_top20 = sum(1 for r in free_rows if (r["compound"], r["disease"]) in top20_pairs)
    n_ev = len(free_rows)

    # AUC vs 2,000 random unlabelled, non-evidence pairs + permutation test
    rng = np.random.default_rng(42)
    ev_set = {(r["compound"], r["disease"]) for r in free_rows}
    all_idx = [(i, j) for i in range(len(compounds)) for j in range(len(diseases))
               if unlabelled_mask[i, j] and (compounds[i], diseases[j]) not in ev_set]
    idx_arr = np.array(all_idx)
    neg = rng.choice(len(idx_arr), size=2000, replace=False)
    neg_scores = scores[idx_arr[neg, 0], idx_arr[neg, 1]]
    auc = mannwhitneyu(ev_scores, neg_scores, alternative="greater").statistic / (n_ev * len(neg_scores))
    n_perm = 2000
    perm_aucs = np.empty(n_perm)
    for k in range(n_perm):
        s = rng.choice(len(idx_arr), size=n_ev, replace=False)
        s_scores = scores[idx_arr[s, 0], idx_arr[s, 1]]
        perm_aucs[k] = mannwhitneyu(s_scores, neg_scores, alternative="greater").statistic / (n_ev * len(neg_scores))
    p_auc = float((perm_aucs >= auc).mean())
    mean_perm_auc = float(perm_aucs.mean())

    out = {
        "n_pairs_total": all_pairs,
        "n_unlabelled": n_unlabelled,
        "n_evidence": len(rows),
        "n_evidence_predicted": n_ev,
        "evidence_top20": int(in_top20),
        "evidence_top5pct": int(sum(1 for r in free_rows if r["rank"] <= 0.05 * n_unlabelled)),
        "evidence_top1pct": int(sum(1 for r in free_rows if r["rank"] <= 0.01 * n_unlabelled)),
        "median_rank": int(np.median([r["rank"] for r in free_rows])),
        "median_percentile": round(float(np.median([r["rank"] for r in free_rows])) / n_unlabelled, 4),
        "auc_vs_random": round(auc, 4),
        "permutation_p": round(p_auc, 4),
        "permutation_mean_auc": round(mean_perm_auc, 4),
        "rows": rows,
    }
    with open(os.path.join(RES, "literature_benchmark.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    with open(os.path.join(RES, "literature_benchmark_table.csv"), "w", encoding="utf-8", newline="") as fh:
        import csv
        w = csv.DictWriter(fh, fieldnames=["compound", "disease", "score", "rank", "percentile",
                                           "evidence", "note", "ref", "status"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=2))
    for r in rows:
        if r["rank"] is None:
            print(f"{r['compound']:<16} {r['disease']:<22} {r['status']}")
        else:
            print(f"{r['compound']:<16} {r['disease']:<22} rank={r['rank']:>4} ({r['percentile']*100:5.2f}%) {r['evidence']}")


if __name__ == "__main__":
    main()
