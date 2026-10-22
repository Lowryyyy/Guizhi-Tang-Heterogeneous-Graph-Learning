"""Bootstrap 95% confidence intervals for held-out AUC-ROC and AUPRC.

For each model (MLP, GCN, R-GCN, HetGNN) and each of five random seeds, the
fixed 70/15/15 split is retrained and predictions are computed on an identical,
fixed test set of positives and 5:1 sampled negatives.  Nonparametric bootstrap
(2,000 resamples) gives 95% confidence intervals, and paired bootstrap gives
confidence intervals for the AUC/AUPRC difference between the HetGNN and each
baseline.  Results are written to results/bootstrap_ci.json.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train as T

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
SEEDS = [42, 7, 2024, 13, 99]
N_BOOT = 2000

torch.set_num_threads(4)


def bootstrap_metric(y, scores, metric, n_boot=N_BOOT, seed=123):
    rng = np.random.default_rng(seed)
    from sklearn.metrics import average_precision_score, roc_auc_score
    fn = roc_auc_score if metric == "auc" else average_precision_score
    vals = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(set(y[idx])) < 2:
            continue
        try:
            vals.append(fn(y[idx], scores[idx]))
        except ValueError:
            continue
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), vals


def main():
    graph = T.load_graph()
    labels = [tuple(x) for x in graph["labels"]]
    n_comp = len(graph["nodes"]["compound"])
    n_dis = len(graph["nodes"]["disease"])
    train_labels, val_labels, test_labels = T.split_labels(labels, seed=42)
    known = set(labels)
    test_neg = T.sample_negatives(list(known), n_comp, n_dis,
                                  int(len(test_labels) * T.NEG_RATIO),
                                  np.random.default_rng(20261020))
    test_x = np.array(test_labels + test_neg)
    test_y = np.array([1] * len(test_labels) + [0] * len(test_neg))

    out = {"notes": "Bootstrap 95% CIs on a fixed held-out set (23 positives, 115 negatives).",
           "n_pos": int((test_y == 1).sum()), "n_neg": int((test_y == 0).sum()),
           "n_boot": N_BOOT, "models": {}}
    probs = {}
    for name in ["mlp", "gcn", "rgcn", "hetgnn"]:
        ps = []
        for seed in SEEDS:
            model, res, _, _, _ = T.train_one(name, graph, train_labels, val_labels,
                                             test_labels, seed=seed)
            data = T.build_hetero_data(graph, train_labels, T.DEVICE)
            model.eval()
            with torch.no_grad():
                p = torch.sigmoid(model(data,
                                        torch.tensor(test_x[:, 0]),
                                        torch.tensor(test_x[:, 1]))).numpy()
            ps.append(p)
            print(name, seed, "auc", round(res["auc"], 3), flush=True)
        probs[name] = np.array(ps)

    from sklearn.metrics import average_precision_score, roc_auc_score
    for name, ps in probs.items():
        mean_prob = ps.mean(axis=0)
        auc = roc_auc_score(test_y, mean_prob)
        auprc = average_precision_score(test_y, mean_prob)
        auc_ci = bootstrap_metric(test_y, mean_prob, "auc")
        auprc_ci = bootstrap_metric(test_y, mean_prob, "auprc", seed=456)
        out["models"][name] = {
            "auc": float(auc), "auc_ci95": [auc_ci[0], auc_ci[1]],
            "auprc": float(auprc), "auprc_ci95": [auprc_ci[0], auprc_ci[1]],
        }
        print(name, "AUC", round(auc, 3), auc_ci[:2], "AUPRC", round(auprc, 3), auprc_ci[:2])

    # paired bootstrap differences vs HetGNN (using ensemble probabilities)
    ref = probs["hetgnn"].mean(axis=0)
    rng = np.random.default_rng(789)
    n = len(test_y)
    out["paired_differences"] = {}
    for name in ["mlp", "gcn", "rgcn"]:
        other = probs[name].mean(axis=0)
        dauc, dauprc = [], []
        for _ in range(N_BOOT):
            idx = rng.integers(0, n, n)
            if len(set(test_y[idx])) < 2:
                continue
            try:
                dauc.append(roc_auc_score(test_y[idx], ref[idx]) - roc_auc_score(test_y[idx], other[idx]))
                dauprc.append(average_precision_score(test_y[idx], ref[idx]) -
                              average_precision_score(test_y[idx], other[idx]))
            except ValueError:
                continue
        out["paired_differences"][name] = {
            "auc_diff_mean": float(np.mean(dauc)),
            "auc_diff_ci95": [float(np.percentile(dauc, 2.5)), float(np.percentile(dauc, 97.5))],
            "auprc_diff_mean": float(np.mean(dauprc)),
            "auprc_diff_ci95": [float(np.percentile(dauprc, 2.5)), float(np.percentile(dauprc, 97.5))],
        }
        print("diff", name, out["paired_differences"][name])

    np.save(os.path.join(RESULTS, "bootstrap_test_probs.npy"),
            np.stack([probs[k] for k in ["mlp", "gcn", "rgcn", "hetgnn"]]))
    with open(os.path.join(RESULTS, "bootstrap_ci.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("written", os.path.join(RESULTS, "bootstrap_ci.json"))


if __name__ == "__main__":
    main()
