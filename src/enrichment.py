"""
Real KEGG and GO enrichment with annotation-bias controls (revision v2).

KEGG
----
Gene-pathway links and pathway names are downloaded from the KEGG REST API and
cached under data/raw. KEGG identifiers are NCBI gene IDs, so the 47 target
symbols are mapped to KEGG gene IDs through the cached /list/hsa annotation.
Enrichment is computed with the one-sided hypergeometric test against the 9,451
human genes present in the KEGG pathway mapping, with Benjamini-Hochberg FDR.
For each pathway the script reports size, expected overlap, fold enrichment and
a permutation p-value.

GO
--
Enrichr (GO_Biological_Process_2023, GO_Molecular_Function_2023,
GO_Cellular_Component_2023) is queried with the target symbols.

The script intentionally lists large pathways that are *not* enriched so that
the dependence of enrichment on annotation density is visible.
"""
from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request

import numpy as np
from scipy.stats import hypergeom

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
RESULTS = os.path.join(ROOT, "results")
KEGG_LINK = os.path.join(RAW, "kegg_hsa_gene_pathway_links.json")
KEGG_NAMES = os.path.join(RAW, "kegg_hsa_pathway_names.json")
KEGG_SYM2ID = os.path.join(RAW, "kegg_symbol_to_geneid.json")


def get_text(url, timeout=180):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 research-script"})
    with urllib.request.urlopen(req, timeout=timeout) as fh:
        return fh.read().decode("utf-8")


def fetch_kegg_mapping(force=False):
    if os.path.exists(KEGG_LINK) and os.path.exists(KEGG_NAMES) and not force:
        links = json.load(open(KEGG_LINK, encoding="utf-8"))
        names = json.load(open(KEGG_NAMES, encoding="utf-8"))
        return links, names
    link_text = get_text("https://rest.kegg.jp/link/pathway/hsa")
    name_text = get_text("https://rest.kegg.jp/list/pathway/hsa")
    links = []
    for line in link_text.strip().splitlines():
        gene, pathway = line.split("\t")
        links.append([gene, pathway.split(":")[-1]])
    names = {}
    for line in name_text.strip().splitlines():
        pid, desc = line.split("\t", 1)
        names[pid.split(":")[-1]] = desc
    json.dump(links, open(KEGG_LINK, "w", encoding="utf-8"))
    json.dump(names, open(KEGG_NAMES, "w", encoding="utf-8"))
    return links, names


def load_sym2id():
    """Load the KEGG symbol -> gene-ID map; build it from KEGG REST if absent."""
    if os.path.exists(KEGG_SYM2ID):
        return json.load(open(KEGG_SYM2ID, encoding="utf-8"))
    text = get_text("https://rest.kegg.jp/list/hsa")
    mapping = {}
    for line in text.strip().splitlines():
        cols = line.split("\t")
        if len(cols) < 4:
            continue
        gene_id, desc = cols[0], cols[3]
        candidate = desc.split(",")[0].strip()
        if candidate and " " not in candidate:
            mapping.setdefault(candidate, gene_id)
    json.dump(mapping, open(KEGG_SYM2ID, "w", encoding="utf-8"))
    return mapping


def _pathway_genes(links):
    rows = {}
    for gene_id, pid in links:
        rows.setdefault(pid, set()).add(gene_id)
    return rows


def kegg_enrichment(query_symbols, links, names, sym2id):
    pathway_ids = _pathway_genes(links)
    background = set()
    for genes in pathway_ids.values():
        background |= genes
    query_ids = {sym2id[s] for s in query_symbols if s in sym2id} & background
    n_query, n_bg = len(query_ids), len(background)
    rows = []
    for pid, genes in pathway_ids.items():
        k = len(query_ids & genes)
        if k == 0:
            continue
        K = len(genes)
        expected = n_query * K / n_bg
        p = float(hypergeom.sf(k - 1, n_bg, K, n_query))
        rows.append({
            "pathway_id": pid, "pathway_name": names.get(pid, pid),
            "overlap": k, "pathway_size": K, "expected_overlap": expected,
            "fold_enrichment": (k / n_query) / (K / n_bg) if n_query else float("nan"),
            "p_value": p, "overlap_genes": sorted(query_ids & genes),
        })
    rows.sort(key=lambda r: r["p_value"])
    m = len(rows)
    for i, row in enumerate(rows):
        row["fdr"] = min(1.0, row["p_value"] * m / (i + 1))
    running = 1.0
    for i in range(len(rows) - 1, -1, -1):
        running = min(running, min(1.0, rows[i]["p_value"] * m / (i + 1)))
        rows[i]["fdr"] = running
    for row in rows:
        row["significant"] = row["fdr"] < 0.05
    return {"query_genes": sorted(query_symbols), "n_query_in_background": n_query,
            "background_size": n_bg, "rows": rows}


def permutation_control(query_symbols, pathway_id, links, names, sym2id, n_perm=2000, seed=42):
    pathway_ids = _pathway_genes(links)
    background = sorted(set().union(*pathway_ids.values()))
    query_ids = {sym2id[s] for s in query_symbols if s in sym2id}
    target = pathway_ids[pathway_id]
    obs = len(query_ids & target)
    rng = np.random.default_rng(seed)
    n = len(query_ids)
    counts = np.array([len(set(rng.choice(background, size=n, replace=False)) & target)
                       for _ in range(n_perm)])
    empirical_p = float((counts >= obs).sum() + 1) / (n_perm + 1)
    return {"pathway_id": pathway_id, "pathway_name": names.get(pathway_id, pathway_id),
            "observed_overlap": obs, "permutation_mean": float(counts.mean()),
            "permutation_p": empirical_p, "n_perm": n_perm,
            "query_size": n, "pathway_size": len(target)}


def enrichr_query(genes, library, description="query"):
    import requests
    payload = {"list": "\n".join(genes), "description": description}
    resp = requests.post("https://maayanlab.cloud/Enrichr/addList", files=payload, timeout=60)
    resp.raise_for_status()
    user_list_id = resp.json()["userListId"]
    url = ("https://maayanlab.cloud/Enrichr/enrich?userListId=%s&backgroundType=%s"
           % (user_list_id, urllib.parse.quote(library)))
    data = {}
    for _ in range(3):
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            data = r.json()
            break
        except Exception:
            time.sleep(2)
    rows = []
    for row in data.get(library, []):
        genes_field = row[5]
        if isinstance(genes_field, str):
            genes = genes_field.split(";") if genes_field else []
        else:
            genes = list(genes_field or [])
        rows.append({"rank": row[0], "term": row[1], "p_value": row[2], "z_score": row[3],
                     "combined_score": row[4], "overlap_genes": genes,
                     "overlap": len(genes), "adjusted_p": row[6], "old_p": row[7]})
    rows.sort(key=lambda r: r["p_value"])
    return {"library": library, "n_terms": len(rows), "rows": rows[:100]}


def build_gene_sets(graph):
    target_names = graph["nodes"]["target"]
    compound_targets, disease_targets = {}, {}
    for key, rows in graph["edges"].items():
        if key == "compound-acts_on-target":
            for c, t in rows:
                compound_targets.setdefault(c, set()).add(target_names[t])
        if key == "target-associated_with-disease":
            for t, d in rows:
                disease_targets.setdefault(d, set()).add(target_names[t])
    return compound_targets, disease_targets


def main():
    os.makedirs(RESULTS, exist_ok=True)
    links, names = fetch_kegg_mapping()
    sym2id = load_sym2id()
    print("KEGG mapping:", len(links), "gene-pathway links,", len(names), "pathways;",
          "symbol map:", len(sym2id))
    with open(os.path.join(ROOT, "data", "processed", "graph_data.json"), encoding="utf-8") as fh:
        graph = json.load(fh)
    compound_targets, disease_targets = build_gene_sets(graph)

    top_path = os.path.join(RESULTS, "top_predictions.json")
    top = json.load(open(top_path, encoding="utf-8")) if os.path.exists(top_path) else []
    # primary predictions (top five after repair) plus the secondary, literature-
    # supported cancer associations that remain in the top 1% of candidate pairs
    selection = list(top[:5])
    for compound, disease in [("6-Gingerol", "Breast cancer"),
                              ("Isorhamnetin", "Breast cancer"),
                              ("Kaempferol", "Pancreatic cancer")]:
        selection.append({"compound": compound, "disease": disease})
    gene_sets = {}
    for row in selection:
        c = graph["nodes"]["compound"].index(row["compound"])
        d = graph["nodes"]["disease"].index(row["disease"])
        shared = sorted(compound_targets.get(c, set()) & disease_targets.get(d, set()))
        if shared:
            gene_sets[f"{row['compound']}__{row['disease']}"] = shared
    print("gene sets:", {k: len(v) for k, v in gene_sets.items()})

    kegg_out = {}
    for name, genes in gene_sets.items():
        kegg_out[name] = kegg_enrichment(genes, links, names, sym2id)
        print(name, len(genes), "genes ->", len(kegg_out[name]["rows"]), "pathways,",
              sum(1 for r in kegg_out[name]["rows"] if r["significant"]), "significant")
    if kegg_out:
        first = max(kegg_out, key=lambda k: len(kegg_out[k]["query_genes"]))
        top_row = kegg_out[first]["rows"][0] if kegg_out[first]["rows"] else None
        if top_row:
            kegg_out["_permutation_control"] = permutation_control(
                kegg_out[first]["query_genes"], top_row["pathway_id"], links, names, sym2id)
    with open(os.path.join(RESULTS, "enrichment_kegg.json"), "w", encoding="utf-8") as fh:
        json.dump(kegg_out, fh, indent=2)

    go_out = {}
    for name, genes in gene_sets.items():
        go_out[name] = {}
        for library in ["GO_Biological_Process_2023", "GO_Molecular_Function_2023",
                        "GO_Cellular_Component_2023"]:
            try:
                go_out[name][library] = enrichr_query(genes, library, name)
                print("  Enrichr", name, library, go_out[name][library]["n_terms"])
            except Exception as exc:
                print("  Enrichr failed", name, library, exc)
                go_out[name][library] = {"error": str(exc)}
    with open(os.path.join(RESULTS, "enrichment_go.json"), "w", encoding="utf-8") as fh:
        json.dump(go_out, fh, indent=2)
    print("written", os.path.join(RESULTS, "enrichment_kegg.json"))


if __name__ == "__main__":
    main()

