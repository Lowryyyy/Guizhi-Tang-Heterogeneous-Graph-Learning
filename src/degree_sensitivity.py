"""Sensitivity of the HetGNN to degree-weighted (hard) negative sampling.

The reviewer asked how degree normalisation was handled and whether study bias
was controlled.  Our main analysis uses uniform negative sampling at 5:1 and adds
a degree-only baseline.  Here we add a robustness check in which training
negatives are sampled with probability proportional to the product of compound
and disease target degrees, i.e. the model sees mostly high-degree, hard
negatives.  If the HetGNN performance is stable under this stress test, the
conclusions do not depend on easy negatives.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train as T
from model import build_hetero_data, build_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
SEEDS = [42, 7, 2024, 13, 99]
torch.set_num_threads(4)


def degree_maps(graph):
    n_comp = len(graph["nodes"]["compound"])
    n_dis = len(graph["nodes"]["disease"])
    comp_deg = np.zeros(n_comp)
    dis_deg = np.zeros(n_dis)
    for key, rows in graph["edges"].items():
        if key == "compound-acts_on-target":
            for c, t in rows:
                comp_deg[c] += 1
        if key == "target-associated-with-disease":
            for t, d in rows:
                dis_deg[d] += 1
    return comp_deg, dis_deg


def weighted_negatives(known, n_comp, n_dis, n_neg, comp_deg, dis_deg, rng):
    pos_set = set(known)
    pool = [(c, d) for c in range(n_comp) for d in range(n_dis) if (c, d) not in pos_set]
    w = np.array([(1.0 + comp_deg[c]) * (1.0 + dis_deg[d]) for c, d in pool], dtype=float)
    w /= w.sum()
    idx = rng.choice(len(pool), size=min(n_neg, len(pool)), replace=False, p=w)
    return [pool[i] for i in idx]


def run(seed, graph, train_labels, val_labels, test_labels, test_x, test_y,
        comp_deg, dis_deg, weighted=True):
    T.set_seed(seed)
    data = build_hetero_data(graph, train_labels, T.DEVICE)
    in_dims = {nt: len(graph["features"][nt][0]) for nt in graph["features"]}
    model = build_model("hetgnn", in_dims, edge_types=data.edge_types, hidden=T.HIDDEN,
                        layers=T.LAYERS, dropout=T.DROPOUT, heads=4).to(T.DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=T.LR, weight_decay=T.WEIGHT_DECAY)
    n_comp = len(graph["nodes"]["compound"])
    n_dis = len(graph["nodes"]["disease"])
    known = set(map(tuple, graph["labels"]))
    neg_val = T.sample_negatives(list(known), n_comp, n_dis,
                                 int(len(val_labels) * T.NEG_RATIO),
                                 np.random.default_rng(seed + 500))
    val_x = np.array(val_labels + neg_val)
    val_y = np.array([1] * len(val_labels) + [0] * len(neg_val))
    best_state, best_auc, patience = None, -1, T.PATIENCE
    for epoch in range(T.MAX_EPOCHS):
        model.train()
        rng = np.random.default_rng(seed * 1000 + epoch)
        if weighted:
            train_neg = weighted_negatives(list(known), n_comp, n_dis,
                                           int(len(train_labels) * T.NEG_RATIO),
                                           comp_deg, dis_deg, rng)
        else:
            train_neg = T.sample_negatives(list(known), n_comp, n_dis,
                                           int(len(train_labels) * T.NEG_RATIO), rng)
        pairs = np.array(train_labels + train_neg)
        y = torch.tensor([1.0] * len(train_labels) + [0.0] * len(train_neg))
        opt.zero_grad()
        logits = model(data, torch.tensor(pairs[:, 0]), torch.tensor(pairs[:, 1]))
        loss = F.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(data, torch.tensor(val_x[:, 0]),
                                     torch.tensor(val_x[:, 1]))).numpy()
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(val_y, pv)
        except ValueError:
            auc = 0.5
        if auc > best_auc + 1e-5:
            best_auc, patience = auc, T.PATIENCE
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience -= 1
            if patience <= 0:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(data, torch.tensor(test_x[:, 0]),
                               torch.tensor(test_x[:, 1]))).numpy()
    return T.metrics(test_y, p, threshold=0.5), p


def main():
    graph = T.load_graph()
    labels = [tuple(x) for x in graph["labels"]]
    train_labels, val_labels, test_labels = T.split_labels(labels, seed=42)
    n_comp = len(graph["nodes"]["compound"])
    n_dis = len(graph["nodes"]["disease"])
    known = set(labels)
    test_neg = T.sample_negatives(list(known), n_comp, n_dis,
                                  int(len(test_labels) * T.NEG_RATIO),
                                  np.random.default_rng(20261020))
    test_x = np.array(test_labels + test_neg)
    test_y = np.array([1] * len(test_labels) + [0] * len(test_neg))
    comp_deg, dis_deg = degree_maps(graph)
    out = {"n_seeds": len(SEEDS), "n_test_pos": int((test_y == 1).sum()),
           "n_test_neg": int((test_y == 0).sum()),
           "runs": {"uniform": [], "degree_weighted": []}}
    for weighted, key in [(False, "uniform"), (True, "degree_weighted")]:
        for seed in SEEDS:
            res, _ = run(seed, graph, train_labels, val_labels, test_labels, test_x, test_y,
                         comp_deg, dis_deg, weighted=weighted)
            out["runs"][key].append(res)
            print(key, seed, "auc", round(res["auc"], 3), "auprc", round(res["auprc"], 3), flush=True)
    for key in out["runs"]:
        out[key] = {
            "auc_mean": float(np.mean([r["auc"] for r in out["runs"][key]])),
            "auc_std": float(np.std([r["auc"] for r in out["runs"][key]])),
            "auprc_mean": float(np.mean([r["auprc"] for r in out["runs"][key]])),
            "auprc_std": float(np.std([r["auprc"] for r in out["runs"][key]])),
        }
    with open(os.path.join(RESULTS, "degree_sensitivity.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "runs"}, indent=2))


if __name__ == "__main__":
    main()
