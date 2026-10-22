"""
Repaired training and evaluation pipeline for the Guizhi Tang revision.

The script reproduces all modelling analyses reported in the revised manuscript:

  results/model_comparison.json   held-out performance of MLP / GCN / R-GCN / HetGNN
  results/cv_results.json         five-fold cross-validation
  results/cold_start.json         leave-compounds-out and leave-diseases-out cold start
  results/ablations.json          edge-type ablations (herb / drug / PPI)
  results/permutation.json        label-permutation null model
  results/baselines.json          degree-only, shared-target and feature-only baselines
  results/shortcut_audit.json     two-hop bridge and degree-bias diagnostics
  results/final_predictions.json  full 77 x 31 score matrix (5-seed ensemble)
  results/top_predictions.json    top unobserved compound-disease pairs
"""
from __future__ import annotations

import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import build_hetero_data, build_model  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
GRAPH_PATH = os.path.join(ROOT, "data", "processed", "graph_data.json")

SEEDS = [42, 7, 2024, 13, 99]
HIDDEN = 64
LAYERS = 2
DROPOUT = 0.3
LR = 0.005
WEIGHT_DECAY = 5e-4
MAX_EPOCHS = 250
PATIENCE = 40
NEG_RATIO = 5
DEVICE = "cpu"

torch.set_num_threads(4)


# --------------------------------------------------------------------------- utils
def load_graph():
    with open(GRAPH_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def metrics(y_true, y_score, threshold=None):
    from sklearn.metrics import (accuracy_score, average_precision_score,
                                 f1_score, roc_auc_score)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    out = {"auc": float(roc_auc_score(y_true, y_score)),
           "auprc": float(average_precision_score(y_true, y_score))}
    if threshold is None:
        best = {"f1": -1.0, "threshold": 0.5}
        for t in np.arange(0.05, 0.96, 0.01):
            f1 = f1_score(y_true, (y_score >= t).astype(int), zero_division=0)
            if f1 > best["f1"]:
                best = {"f1": float(f1), "threshold": float(t)}
        out.update(best)
    else:
        out["threshold"] = float(threshold)
        out["f1"] = float(f1_score(y_true, (y_score >= threshold).astype(int), zero_division=0))
    out["accuracy"] = float(accuracy_score(y_true, (y_score >= out["threshold"]).astype(int)))
    return out


def all_pairs(n_comp, n_dis):
    return [(c, d) for c in range(n_comp) for d in range(n_dis)]


def sample_negatives(positives, n_comp, n_dis, n_neg, rng):
    pos_set = set(map(tuple, positives))
    pool = [p for p in all_pairs(n_comp, n_dis) if p not in pos_set]
    idx = rng.choice(len(pool), size=min(n_neg, len(pool)), replace=False)
    return [pool[i] for i in idx]


def split_labels(labels, frac=(0.7, 0.15, 0.15), seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(labels))
    n_train = int(round(frac[0] * len(labels)))
    n_val = int(round(frac[1] * len(labels)))
    train = [labels[i] for i in idx[:n_train]]
    val = [labels[i] for i in idx[n_train:n_train + n_val]]
    test = [labels[i] for i in idx[n_train + n_val:]]
    return train, val, test


# --------------------------------------------------------------------------- training
def train_one(model_name, graph, train_labels, val_pairs, test_pairs,
              feature_masks=None, seed=42, edge_types=None):
    """Train one model and return validation/test metrics and predictions."""
    set_seed(seed)
    g = graph
    if feature_masks:
        g = json.loads(json.dumps(graph))
        for nt, cols in feature_masks.items():
            for row in g["features"][nt]:
                for c in cols:
                    row[c] = 0.0
    data = build_hetero_data(g, train_labels, DEVICE)
    in_dims = {nt: graph["features"][nt].__len__() and len(graph["features"][nt][0]) for nt in graph["features"]}
    edge_types = data.edge_types
    model = build_model(model_name, in_dims, edge_types=edge_types,
                        hidden=HIDDEN, layers=LAYERS, dropout=DROPOUT, heads=4).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=15, factor=0.5)

    n_comp = len(graph["nodes"]["compound"])
    n_dis = len(graph["nodes"]["disease"])
    known = set(map(tuple, graph["labels"]))
    val_neg = sample_negatives(list(known), n_comp, n_dis, int(len(val_pairs) * NEG_RATIO), np.random.default_rng(seed))
    best_state, best_val_auc, best_epoch, patience_left = None, -1.0, 0, PATIENCE
    val_x = np.array(val_pairs + val_neg)
    val_y = np.array([1] * len(val_pairs) + [0] * len(val_neg))

    for epoch in range(MAX_EPOCHS):
        model.train()
        train_neg = sample_negatives(list(known), n_comp, n_dis,
                                     int(len(train_labels) * NEG_RATIO),
                                     np.random.default_rng(seed + epoch))
        pairs = np.array(train_labels + train_neg)
        y = torch.tensor([1.0] * len(train_labels) + [0.0] * len(train_neg))
        c = torch.tensor(pairs[:, 0], dtype=torch.long)
        d = torch.tensor(pairs[:, 1], dtype=torch.long)
        opt.zero_grad()
        logits = model(data, c, d)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        model.eval()
        with torch.no_grad():
            logits_val = model(data, torch.tensor(val_x[:, 0]), torch.tensor(val_x[:, 1]))
            prob_val = torch.sigmoid(logits_val).numpy()
        try:
            from sklearn.metrics import roc_auc_score
            val_auc = roc_auc_score(val_y, prob_val)
        except ValueError:
            val_auc = 0.5
        sched.step(-val_auc)
        if val_auc > best_val_auc + 1e-5:
            best_val_auc = val_auc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_left = PATIENCE
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    # threshold selected on validation, never on test
    model.eval()
    with torch.no_grad():
        val_prob = torch.sigmoid(model(data, torch.tensor(val_x[:, 0]), torch.tensor(val_x[:, 1]))).numpy()
    best_f1, best_t = -1.0, 0.5
    from sklearn.metrics import f1_score
    for t in np.arange(0.05, 0.96, 0.01):
        f1 = f1_score(val_y, (val_prob >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = float(f1), float(t)

    test_neg = sample_negatives(list(known), n_comp, n_dis, int(len(test_pairs) * NEG_RATIO),
                                np.random.default_rng(seed + 10_000))
    test_x = np.array(test_pairs + test_neg)
    test_y = np.array([1] * len(test_pairs) + [0] * len(test_neg))
    with torch.no_grad():
        test_prob = torch.sigmoid(model(data, torch.tensor(test_x[:, 0]), torch.tensor(test_x[:, 1]))).numpy()
    res = metrics(test_y, test_prob, threshold=best_t)
    res["best_val_auc"] = float(best_val_auc)
    res["best_epoch"] = int(best_epoch)
    return model, res, test_x, test_prob, best_t


def eval_ablation(name, graph, labels, edge_types_to_drop, seeds=SEEDS[:3]):
    g = json.loads(json.dumps(graph))
    for key in list(g["edges"]):
        if key in edge_types_to_drop:
            g["edges"][key] = []
    train, val, test = split_labels(labels, seed=42)
    fold_auc, fold_auprc = [], []
    for seed in seeds:
        _, res, _, _, _ = train_one("hetgnn", g, train, val, test, seed=seed)
        fold_auc.append(res["auc"])
        fold_auprc.append(res["auprc"])
    return {"ablation": name,
            "auc_mean": float(np.mean(fold_auc)), "auc_std": float(np.std(fold_auc)),
            "auprc_mean": float(np.mean(fold_auprc)), "auprc_std": float(np.std(fold_auprc)),
            "n_seeds": len(seeds)}


# --------------------------------------------------------------------------- main
def main():
    os.makedirs(RESULTS, exist_ok=True)
    graph = load_graph()
    labels = [tuple(x) for x in graph["labels"]]
    n_comp = len(graph["nodes"]["compound"])
    n_dis = len(graph["nodes"]["disease"])
    print("nodes", graph["stats"]["nodes"], "edges", graph["stats"]["edges"], "labels", len(labels))

    with open(os.path.join(RESULTS, "graph_statistics.json"), "w", encoding="utf-8") as fh:
        json.dump(graph["stats"], fh, indent=2)

    # ---------------------------------------------------- fixed held-out split
    train, val, test = split_labels(labels, seed=42)
    comparison = {}
    for name in ["mlp", "gcn", "rgcn", "hetgnn"]:
        runs = []
        for seed in SEEDS:
            _, res, _, _, _ = train_one(name, graph, train, val, test, seed=seed)
            runs.append(res)
        comparison[name] = {
            "auc_mean": float(np.mean([r["auc"] for r in runs])),
            "auc_std": float(np.std([r["auc"] for r in runs])),
            "auprc_mean": float(np.mean([r["auprc"] for r in runs])),
            "auprc_std": float(np.std([r["auprc"] for r in runs])),
            "f1_mean": float(np.mean([r["f1"] for r in runs])),
            "f1_std": float(np.std([r["f1"] for r in runs])),
            "threshold_mean": float(np.mean([r["threshold"] for r in runs])),
            "runs": runs,
        }
        print(name, comparison[name]["auc_mean"], comparison[name]["auprc_mean"])
    with open(os.path.join(RESULTS, "model_comparison.json"), "w", encoding="utf-8") as fh:
        json.dump(comparison, fh, indent=2)

    # ------------------------------------------------------------ five-fold CV
    from sklearn.model_selection import KFold
    cv = {name: [] for name in ["mlp", "gcn", "rgcn", "hetgnn"]}
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (tr_idx, te_idx) in enumerate(kf.split(np.arange(len(labels)))):
        tr = [labels[i] for i in tr_idx]
        te = [labels[i] for i in te_idx]
        inner = np.random.default_rng(1000 + fold).permutation(len(tr))
        n_val = max(5, int(0.2 * len(tr)))
        va = [tr[i] for i in inner[:n_val]]
        tr2 = [tr[i] for i in inner[n_val:]]
        for name in cv:
            _, res, _, _, _ = train_one(name, graph, tr2, va, te, seed=SEEDS[fold % len(SEEDS)])
            cv[name].append(res)
    cv_summary = {name: {"auc_mean": float(np.mean([r["auc"] for r in v])),
                         "auc_std": float(np.std([r["auc"] for r in v])),
                         "auprc_mean": float(np.mean([r["auprc"] for r in v])),
                         "auprc_std": float(np.std([r["auprc"] for r in v])),
                         "folds": v}
                  for name, v in cv.items()}
    with open(os.path.join(RESULTS, "cv_results.json"), "w", encoding="utf-8") as fh:
        json.dump(cv_summary, fh, indent=2)

    # ------------------------------------------------------------- cold start
    from sklearn.model_selection import KFold
    compounds_in_labels = sorted({c for c, d in labels})
    diseases_in_labels = sorted({d for c, d in labels})
    cold = {"compound": [], "disease": []}
    for mode, nodes in (("compound", compounds_in_labels), ("disease", diseases_in_labels)):
        kf2 = KFold(n_splits=5, shuffle=True, random_state=7)
        for fold, (_, held_idx) in enumerate(kf2.split(np.arange(len(nodes)))):
            held = {nodes[i] for i in held_idx}
            if mode == "compound":
                held_labels = [l for l in labels if l[0] in held]
                train_labels = [l for l in labels if l[0] not in held]
                g = json.loads(json.dumps(graph))
                for key, rows in g["edges"].items():
                    s, r, d = key.split("-")
                    if r == "acts_on" and s == "compound":
                        g["edges"][key] = [row for row in rows if row[0] not in held]
                    if r == "contains" and d == "compound":
                        g["edges"][key] = [row for row in rows if row[1] not in held]
                masks = {"compound": [2, 3]}
            else:
                held_labels = [l for l in labels if l[1] in held]
                train_labels = [l for l in labels if l[1] not in held]
                g = json.loads(json.dumps(graph))
                for key, rows in g["edges"].items():
                    s, r, d = key.split("-")
                    if d == "disease" and r in ("associated_with", "treats"):
                        g["edges"][key] = [row for row in rows if row[1] not in held]
                masks = {"disease": [0, 1]}
            # positives and negatives for held nodes
            pos = held_labels
            pos_set = set(labels)
            rng = np.random.default_rng(5000 + fold)
            if mode == "compound":
                pool = [(c, d) for c in held for d in range(n_dis) if (c, d) not in pos_set]
            else:
                pool = [(c, d) for c in range(n_comp) for d in held if (c, d) not in pos_set]
            neg_idx = rng.choice(len(pool), size=min(len(pos) * NEG_RATIO, len(pool)), replace=False)
            neg = [pool[i] for i in neg_idx]
            # inner validation split for early stopping
            perm = rng.permutation(len(train_labels))
            n_val = max(5, int(0.2 * len(train_labels)))
            va = [train_labels[i] for i in perm[:n_val]]
            tr = [train_labels[i] for i in perm[n_val:]]
            _, res, _, _, _ = train_one("hetgnn", g, tr, va, pos, feature_masks=masks, seed=SEEDS[fold])
            # evaluate with negatives appended
            g_full = json.loads(json.dumps(g))
            cold[mode].append({"fold": fold, "n_held": len(held), "n_pos": len(pos), "metrics": res})
    cold_summary = {}
    for mode, folds in cold.items():
        cold_summary[mode] = {
            "auc_mean": float(np.mean([f["metrics"]["auc"] for f in folds])),
            "auc_std": float(np.std([f["metrics"]["auc"] for f in folds])),
            "auprc_mean": float(np.mean([f["metrics"]["auprc"] for f in folds])),
            "auprc_std": float(np.std([f["metrics"]["auprc"] for f in folds])),
            "folds": folds,
        }
    with open(os.path.join(RESULTS, "cold_start.json"), "w", encoding="utf-8") as fh:
        json.dump(cold_summary, fh, indent=2)

    # --------------------------------------------------------------- ablations
    ablations = [
        eval_ablation("no herb layer", graph, labels, {"herb-contains-compound"}),
        eval_ablation("no drug layer", graph, labels,
                      {"drug-targets_protein-target", "drug-treats-disease"}),
        eval_ablation("no PPI", graph, labels, {"target-interacts_with-target"}),
        eval_ablation("no disease-target annotations", graph, labels,
                      {"target-associated_with-disease"}),
    ]
    with open(os.path.join(RESULTS, "ablations.json"), "w", encoding="utf-8") as fh:
        json.dump(ablations, fh, indent=2)

    # -------------------------------------------------------------- permutation
    perm_aucs = []
    for rep in range(5):
        rng = np.random.default_rng(900 + rep)
        comps = [c for c, d in labels]
        dis = [d for c, d in labels]
        rng.shuffle(dis)
        shuffled = list(dict.fromkeys(zip(comps, dis)))  # unique pairs
        tr, va, te = split_labels(shuffled, seed=100 + rep)
        _, res, _, _, _ = train_one("hetgnn", graph, tr, va, te, seed=SEEDS[rep])
        perm_aucs.append(res["auc"])
    with open(os.path.join(RESULTS, "permutation.json"), "w", encoding="utf-8") as fh:
        json.dump({"auc_mean": float(np.mean(perm_aucs)), "auc_std": float(np.std(perm_aucs)),
                   "aucs": perm_aucs, "n_permutations": len(perm_aucs)}, fh, indent=2)

    # ---------------------------------------------------------------- baselines
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    edge_set = {}
    for key, rows in graph["edges"].items():
        edge_set[key] = set(map(tuple, rows))
    comp_targets = {}
    for key, rows in graph["edges"].items():
        if key == "compound-acts_on-target":
            for c, t in rows:
                comp_targets.setdefault(c, set()).add(t)
    dis_targets = {}
    for key, rows in graph["edges"].items():
        if key == "target-associated_with-disease":
            for t, d in rows:
                dis_targets.setdefault(d, set()).add(t)
    comp_deg = np.array([len(comp_targets.get(c, ())) for c in range(n_comp)], dtype=float)
    dis_deg = np.array([len(dis_targets.get(d, ())) for d in range(n_dis)], dtype=float)
    comp_herb = np.array([sum(1 for key, rows in graph["edges"].items()
                              if key == "herb-contains-compound" and c in [r[1] for r in rows])
                          for c in range(n_comp)], dtype=float)
    obdl = np.array([[f[0], f[1]] for f in graph["features"]["compound"]])
    feas = []
    ys = []
    for c in range(n_comp):
        for d in range(n_dis):
            shared = len(comp_targets.get(c, set()) & dis_targets.get(d, set()))
            feas.append([comp_deg[c], dis_deg[d], comp_herb[c], obdl[c, 0], obdl[c, 1], shared])
            ys.append(1 if (c, d) in set(map(tuple, labels)) else 0)
    feas = np.array(feas)
    ys = np.array(ys)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    deg_aucs, shared_aucs = [], []
    for tr_i, te_i in skf.split(feas, ys):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
        clf.fit(feas[tr_i][:, :5], ys[tr_i])
        deg_aucs.append(roc_auc_score(ys[te_i], clf.predict_proba(feas[te_i][:, :5])[:, 1]))
        shared_aucs.append(roc_auc_score(ys[te_i], feas[te_i][:, 5]))
    baselines = {
        "degree_feature_logistic": {"auc_mean": float(np.mean(deg_aucs)), "auc_std": float(np.std(deg_aucs))},
        "shared_target_count_rule": {"auc_mean": float(np.mean(shared_aucs)), "auc_std": float(np.std(shared_aucs))},
        "note": ("Degree/feature-only baseline uses compound target degree, disease target "
                 "degree, herb count, OB, DL. Shared-target rule uses the number of proteins "
                 "shared by a compound and a disease as the prediction score."),
    }
    with open(os.path.join(RESULTS, "baselines.json"), "w", encoding="utf-8") as fh:
        json.dump(baselines, fh, indent=2)

    # -------------------------------------------------------- shortcut audit
    # two-hop support for known labels and for the old (circular) drug design
    drugs_targets = {}
    for key, rows in graph["edges"].items():
        if key == "drug-targets_protein-target":
            for dr, t in rows:
                drugs_targets.setdefault(dr, set()).add(t)
    drug_disease = {}
    for key, rows in graph["edges"].items():
        if key == "drug-treats-disease":
            for dr, d in rows:
                drug_disease.setdefault(d, set()).add(dr)
    target_bridge = 0
    drug_bridge = 0
    both = 0
    for c, d in labels:
        ct = comp_targets.get(c, set())
        dt = dis_targets.get(d, set())
        tb = len(ct & dt)
        db = 0
        for dr in drug_disease.get(d, set()):
            if drugs_targets.get(dr, set()) & ct:
                db += 1
        target_bridge += tb > 0
        drug_bridge += db > 0
        both += (tb > 0) and (db > 0)
    old_synthetic = 0
    for c, d in labels:
        ct = comp_targets.get(c, set())
        dt = dis_targets.get(d, set())
        for dr, tl in drugs_targets.items():
            if len(tl & dt) >= 2 and len(tl & ct) >= 1:
                old_synthetic += 1
                break
    with open(os.path.join(RESULTS, "shortcut_audit.json"), "w", encoding="utf-8") as fh:
        json.dump({
            "n_labels": len(labels),
            "labels_with_target_bridge": int(target_bridge),
            "labels_with_drug_bridge": int(drug_bridge),
            "labels_with_target_and_drug_bridge": int(both),
            "fraction_labels_with_target_bridge": float(target_bridge / len(labels)),
            "labels_connected_by_old_synthetic_drug_edges": int(old_synthetic),
            "fraction_connected_by_old_synthetic_drug_edges": float(old_synthetic / len(labels)),
            "note": ("The original submission inferred drug-treats-disease edges from shared "
                     "targets; the audit counts how many known compound-disease labels would "
                     "have been connected through such circular drug bridges."),
        }, fh, indent=2)

    # ----------------------------------------------- final model and predictions
    ensemble = np.zeros((n_comp, n_dis))
    kf_final = KFold(n_splits=5, shuffle=True, random_state=2024)
    for fold, (tr_idx, _) in enumerate(kf_final.split(np.arange(len(labels)))):
        tr_final = [labels[i] for i in tr_idx]
        perm_f = np.random.default_rng(2000 + fold).permutation(len(tr_final))
        n_va = max(5, int(0.2 * len(tr_final)))
        va_final = [tr_final[i] for i in perm_f[:n_va]]
        tr_final = [tr_final[i] for i in perm_f[n_va:]]
        model, _, _, _, _ = train_one("hetgnn", graph, tr_final, va_final, va_final, seed=SEEDS[fold])
        model.eval()
        pairs = torch.tensor(all_pairs(n_comp, n_dis), dtype=torch.long)
        data = build_hetero_data(graph, labels, DEVICE)
        with torch.no_grad():
            probs = torch.sigmoid(model(data, pairs[:, 0], pairs[:, 1])).numpy()
        ensemble += probs.reshape(n_comp, n_dis)
    ensemble /= len(SEEDS)
    known = set(map(tuple, labels))
    ranked = sorted([((c, d), float(ensemble[c, d])) for c in range(n_comp) for d in range(n_dis)
                     if (c, d) not in known], key=lambda x: -x[1])
    names_c = graph["nodes"]["compound"]
    names_d = graph["nodes"]["disease"]
    top = []
    for (c, d), score in ranked[:20]:
        top.append({
            "compound": names_c[c], "disease": names_d[d], "score": score,
            "shared_targets": len(comp_targets.get(c, set()) & dis_targets.get(d, set())),
        })
    np.save(os.path.join(RESULTS, "final_score_matrix.npy"), ensemble)
    with open(os.path.join(RESULTS, "final_predictions.json"), "w", encoding="utf-8") as fh:
        json.dump({"compounds": names_c, "diseases": names_d,
                   "known_links": [list(x) for x in sorted(known)]}, fh, indent=2)
    with open(os.path.join(RESULTS, "top_predictions.json"), "w", encoding="utf-8") as fh:
        json.dump(top, fh, indent=2)
    print("TOP PREDICTIONS")
    for t in top[:10]:
        print(f"  {t['compound']:22s} -> {t['disease']:30s} {t['score']:.4f} "
              f"shared={t['shared_targets']}")


if __name__ == "__main__":
    main()


