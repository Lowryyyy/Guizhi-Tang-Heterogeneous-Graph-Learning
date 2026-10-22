"""Generate the eight main figures for the revised manuscript.

Scientific Reports submission requirements respected here:
  * widths within 180 mm (figsize <= 7.0 in);
  * no figure titles or legends inside the image (only axis labels, panel letters
    and legends);
  * 600 dpi PNG plus editable SVG;
  * one figure per file, single page.
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
PROC = os.path.join(ROOT, "data", "processed")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

COLORS = {"herb": "#c0392b", "compound": "#2e86c1", "target": "#27ae60",
          "disease": "#e67e22", "drug": "#8e44ad"}
LABELS = {"herb": "Herb", "compound": "Compound", "target": "Target",
          "disease": "Disease", "drug": "Approved drug"}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                     "axes.linewidth": 0.8, "savefig.dpi": 600})


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".png"), bbox_inches="tight", dpi=600)
    fig.savefig(os.path.join(FIG, name + ".svg"), bbox_inches="tight")
    plt.close(fig)


def load(name, default=None):
    path = os.path.join(RES, name)
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_graph():
    with open(os.path.join(PROC, "graph_data.json"), encoding="utf-8") as fh:
        return json.load(fh)


def panel(ax, letter, x=0.01, y=1.02):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="bottom", ha="left")


def fig1_screening():
    graph = load_graph()
    names = graph["nodes"]["compound"]
    feats = graph["features"]["compound"]
    ob = np.array([f[0] * 100 for f in feats])
    dl = np.array([f[1] for f in feats])
    colors = np.array([f[3] for f in feats])
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    scatter = ax.scatter(ob, dl, c=colors, cmap="viridis", s=22, edgecolor="black", linewidth=0.2)
    ax.axvline(30, color="grey", ls="--", lw=0.8)
    ax.axhline(0.18, color="grey", ls="--", lw=0.8)
    annotate = ["Cinnamaldehyde", "Paeonol", "Isoliquiritigenin", "Kaempferol",
                "6-Gingerol", "Quercetin", "Glycyrrhizin", "Paeoniflorin"]
    for nm in annotate:
        if nm in names:
            i = names.index(nm)
            ax.annotate(nm, (ob[i], dl[i]), fontsize=6, xytext=(3, 2),
                        textcoords="offset points")
    below = dl < 0.18
    ax.scatter(ob[below], dl[below], s=22, facecolors="none", edgecolors="red", linewidth=0.8)
    ax.set_xlabel("Predicted oral bioavailability (OB, %)")
    ax.set_ylabel("Drug-likeness (DL)")
    cb = fig.colorbar(scatter, ax=ax, pad=0.02)
    cb.set_label("Herb membership count (normalised)")
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
                      markeredgecolor="red", markersize=4, label="DL < 0.18 (retained)")]
    ax.legend(handles=handles, loc="lower right", fontsize=6.5, frameon=False)
    save(fig, "Fig1_compound_screening")


def fig2_graph():
    import networkx as nx
    graph = load_graph()
    G = nx.Graph()
    for nt, names in graph["nodes"].items():
        for i, nm in enumerate(names):
            G.add_node(f"{nt}:{i}", ntype=nt, label=nm)
    for key, edges in graph["edges"].items():
        s, r, d = key.split("-")
        if r == "may_treat":
            continue
        for a, b in edges:
            G.add_edge(f"{s}:{a}", f"{d}:{b}", rel=r)
    pos = nx.spring_layout(G, seed=42, k=0.45, iterations=120)
    fig, ax = plt.subplots(figsize=(7.0, 5.4))
    for nt in graph["nodes"]:
        nodes = [n for n, d in G.nodes(data=True) if d["ntype"] == nt]
        sizes = {"herb": 140, "compound": 45, "target": 60, "disease": 75, "drug": 38}[nt]
        nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=COLORS[nt], node_size=sizes,
                               edgecolors="white", linewidths=0.4, ax=ax, label=LABELS[nt])
    nx.draw_networkx_edges(G, pos, alpha=0.16, width=0.3, ax=ax)
    ax.legend(scatterpoints=1, frameon=False, fontsize=6.5, loc="lower left", ncol=2)
    ax.axis("off")
    save(fig, "Fig2_knowledge_graph")


def fig3_heatmap():
    graph = load_graph()
    scores = np.load(os.path.join(RES, "final_score_matrix.npy"))
    comp = graph["nodes"]["compound"]
    dis = graph["nodes"]["disease"]
    row_order = np.argsort(-scores.max(axis=1))[:20]
    col_order = np.argsort(-scores.max(axis=0))[:12]
    mat = scores[np.ix_(row_order, col_order)]
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    im = ax.imshow(mat, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(col_order)))
    ax.set_xticklabels([dis[i] for i in col_order], rotation=45, ha="right", fontsize=6.5)
    ax.set_yticks(range(len(row_order)))
    ax.set_yticklabels([comp[i] for i in row_order], fontsize=6.5)
    ax.set_xlabel("Disease")
    ax.set_ylabel("Compound")
    fig.colorbar(im, ax=ax, label="Predicted probability", pad=0.02)
    save(fig, "Fig5_prediction_heatmap")


def fig4_performance():
    mc = load("model_comparison.json", {})
    cold = load("cold_start.json", {})
    perm = load("permutation.json", {})
    names = {"mlp": "MLP", "gcn": "GCN", "rgcn": "R-GCN", "hetgnn": "HetGNN"}
    order = ["mlp", "gcn", "rgcn", "hetgnn"]
    colors = ["#95a5a6", "#3498db", "#e67e22", "#c0392b"]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.6))
    ax = axes[0, 0]
    vals = [mc[k]["auc_mean"] for k in order]
    errs = [mc[k]["auc_std"] for k in order]
    ax.bar(range(4), vals, yerr=errs, color=colors, capsize=2.5)
    ax.set_xticks(range(4)); ax.set_xticklabels([names[k] for k in order], fontsize=7)
    ax.set_ylabel("Test AUC-ROC"); ax.set_ylim(0.5, 1.02)
    panel(ax, "(a)")
    ax = axes[0, 1]
    vals = [mc[k]["auprc_mean"] for k in order]
    errs = [mc[k]["auprc_std"] for k in order]
    ax.bar(range(4), vals, yerr=errs, color=colors, capsize=2.5)
    ax.set_xticks(range(4)); ax.set_xticklabels([names[k] for k in order], fontsize=7)
    ax.set_ylabel("Test AUPRC"); ax.set_ylim(0.3, 1.0)
    panel(ax, "(b)")
    ax = axes[1, 0]
    modes = ["compound", "disease"]
    vals = [cold[m]["auc_mean"] for m in modes]
    errs = [cold[m]["auc_std"] for m in modes]
    ax.bar(range(2), vals, yerr=errs, color=["#16a085", "#8e44ad"], capsize=4)
    ax.axhline(mc["hetgnn"]["auc_mean"], color="grey", ls="--", lw=0.8)
    ax.text(0.02, mc["hetgnn"]["auc_mean"] + 0.02, "transductive", fontsize=6.5, color="grey")
    ax.set_xticks(range(2)); ax.set_xticklabels(["Unseen compounds", "Unseen diseases"], fontsize=7)
    ax.set_ylabel("Cold-start AUC-ROC"); ax.set_ylim(0.3, 1.0)
    panel(ax, "(c)")
    ax = axes[1, 1]
    ax.hist(perm["aucs"], bins=8, color="#bdc3c7", edgecolor="black", linewidth=0.4)
    ax.axvline(mc["hetgnn"]["auc_mean"], color="#c0392b", lw=1.2)
    ax.set_xlabel("AUC-ROC under label permutation"); ax.set_ylabel("Count")
    panel(ax, "(d)")
    fig.tight_layout()
    save(fig, "Fig3_model_controls")


def fig5_kegg():
    kegg = load("enrichment_kegg.json", {})
    sets = [k for k in kegg if not k.startswith("_")]
    if not sets:
        return
    top = load("top_predictions.json", [])
    prefer = f"{top[0]['compound']}__{top[0]['disease']}" if top else None
    key = prefer if prefer in sets else max(sets, key=lambda k: len(kegg[k]["query_genes"]))
    rows = sorted(kegg[key]["rows"], key=lambda r: r["p_value"])[:15]
    fig, ax = plt.subplots(figsize=(6.3, 4.8))
    y = np.arange(len(rows))[::-1]
    pvals = [-np.log10(max(r["p_value"], 1e-300)) for r in rows]
    norm = plt.Normalize(0, max(1.0, max(pvals)))
    ax.scatter([r["fold_enrichment"] for r in rows], y,
               s=[max(15, r["overlap"] * 22) for r in rows],
               c=pvals, cmap="magma_r", norm=norm, edgecolor="black", linewidth=0.3)
    for yi, r in zip(y, rows):
        ax.text(r["fold_enrichment"] + 0.4, yi,
                f"{r['overlap']}/{r['pathway_size']} (q={r['fdr']:.1e})", va="center", fontsize=5.5)
    ax.set_yticks(y)
    ax.set_yticklabels([r["pathway_name"].replace(" - Homo sapiens (human)", "")[:45] for r in rows],
                       fontsize=6)
    ax.set_xlabel("Fold enrichment (observed / expected)")
    fig.colorbar(plt.cm.ScalarMappable(cmap="magma_r", norm=norm), ax=ax,
                 label="-log10 P", pad=0.02)
    save(fig, "Fig6_kegg_enrichment")


def fig6_go():
    go = load("enrichment_go.json", {})
    sets = [k for k in go if isinstance(go[k], dict)]
    if not sets:
        return
    top = load("top_predictions.json", [])
    prefer = f"{top[0]['compound']}__{top[0]['disease']}" if top else None
    key = prefer if prefer in sets else max(
        sets, key=lambda k: len(go[k].get("GO_Biological_Process_2023", {}).get("rows", [])))
    rows = go[key].get("GO_Biological_Process_2023", {}).get("rows", [])[:12]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(6.1, 4.3))
    vals = [-np.log10(max(r["p_value"], 1e-300)) for r in rows][::-1]
    labels = [r["term"][:40] for r in rows][::-1]
    ax.barh(range(len(vals)), vals, color="#2980b9")
    ax.set_yticks(range(len(vals))); ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel("-log10 P (Enrichr)")
    save(fig, "Fig7_go_enrichment")


def fig7_docking():
    if not os.path.exists(os.path.join(RES, "docking_results.json")):
        return
    dock = load("docking_results.json", {})
    runs = dock["runs"]
    pred = [r["best_affinity"] for r in runs
            if r["set"] == "predicted" and r["best_affinity"] < 0]
    dec = [r["best_affinity"] for r in runs
           if r["set"] == "decoy" and r["best_affinity"] < 0]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4), gridspec_kw={"width_ratios": [1.0, 1.1]})
    ax = axes[0]
    ax.hist(dec, bins=12, alpha=0.65, label=f"Decoy pairs (n={len(dec)})", color="#95a5a6",
            edgecolor="black", linewidth=0.3, density=True)
    ax.hist(pred, bins=8, alpha=0.7, label=f"Predicted pairs (n={len(pred)})", color="#c0392b",
            edgecolor="black", linewidth=0.3, density=True)
    ax.axvline(-5.0, color="grey", ls="--", lw=0.8)
    ax.set_xlabel("Vina binding affinity (kcal/mol)"); ax.set_ylabel("Density")
    ax.legend(fontsize=6, frameon=False)
    panel(ax, "(a)")
    ax = axes[1]
    pairs = sorted([r for r in runs if r["set"] == "predicted" and r["best_affinity"] < 0],
                   key=lambda r: r["best_affinity"])
    y = np.arange(len(pairs))
    ax.barh(y, [r["best_affinity"] for r in pairs], color="#c0392b", height=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['compound']}-{r['target']}" for r in pairs], fontsize=5.5)
    ax.set_xlabel("Best Vina affinity (kcal/mol)")
    panel(ax, "(b)")
    fig.tight_layout()
    save(fig, "Fig8_docking")


def fig8_bias():
    audit = load("shortcut_audit.json", {})
    base = load("baselines.json", {})
    mc = load("model_comparison.json", {})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))
    ax = axes[0]
    vals = [base["degree_feature_logistic"]["auc_mean"],
            base["shared_target_count_rule"]["auc_mean"],
            mc["mlp"]["auc_mean"], mc["hetgnn"]["auc_mean"]]
    errs = [base["degree_feature_logistic"]["auc_std"],
            base["shared_target_count_rule"]["auc_std"],
            mc["mlp"]["auc_std"], mc["hetgnn"]["auc_std"]]
    labels = ["Degree/feature\nlogistic", "Shared-target\nrule", "MLP", "HetGNN"]
    colors = ["#7f8c8d", "#f39c12", "#3498db", "#c0392b"]
    ax.bar(range(4), vals, yerr=errs, color=colors, capsize=2.5)
    ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=6)
    ax.set_ylabel("AUC-ROC"); ax.set_ylim(0.3, 1.02)
    panel(ax, "(a)")
    ax = axes[1]
    total = audit.get("n_labels", 154)
    n_bridge = audit.get("labels_with_target_bridge", 0)
    ax.barh([0], [total - n_bridge], color="#bdc3c7", label="No two-hop target bridge")
    ax.barh([0], [n_bridge], left=[total - n_bridge], color="#c0392b",
            label="Two-hop target bridge")
    ax.set_yticks([])
    ax.set_xlabel("Number of known compound-disease labels")
    ax.legend(fontsize=6, frameon=False, loc="lower right")
    panel(ax, "(b)")
    fig.tight_layout()
    save(fig, "Fig4_bias_audit")


def main():
    fig1_screening()
    fig2_graph()
    if os.path.exists(os.path.join(RES, "final_score_matrix.npy")):
        fig3_heatmap()
    if os.path.exists(os.path.join(RES, "model_comparison.json")):
        fig4_performance()
    if os.path.exists(os.path.join(RES, "enrichment_kegg.json")):
        fig5_kegg()
    if os.path.exists(os.path.join(RES, "enrichment_go.json")):
        fig6_go()
    if os.path.exists(os.path.join(RES, "docking_results.json")):
        fig7_docking()
    if os.path.exists(os.path.join(RES, "shortcut_audit.json")):
        fig8_bias()
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
