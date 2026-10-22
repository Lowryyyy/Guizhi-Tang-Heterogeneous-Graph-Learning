"""
Build the curated heterogeneous knowledge graph for Guizhi Tang.

Revision notes (v2.0)
---------------------
Compared with the code deposited with the original submission, this builder:
  * replaces the hard-coded PPI list with the STRING v12.0 interactions that are
    stored in data/raw/string_ppi_real.json (combined score >= 400);
  * removes the previously used compound-disease labels from message passing and
    keeps them exclusively as the supervised target;
  * replaces the synthetic drug-disease edges (which were inferred from shared
    targets and therefore circular) with real drug indications retrieved from
    ChEMBL (data/raw/drug_indications_chembl.json when available);
  * uses a deterministic node ordering so that the graph, splits and results are
    exactly reproducible.

Outputs
-------
data/processed/graph_data.json
    nodes          {node_type: [name, ...]}
    features       {node_type: [[float, ...], ...]}
    edges          {relation: [[src_idx, dst_idx], ...]}
    labels         [[compound_idx, disease_idx], ...]
    stats          summary of nodes / edges
    provenance     input files and versions
"""
from __future__ import annotations

import json
import os
from collections import Counter, defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
PROCESSED = os.path.join(ROOT, "data", "processed")

RELATIONS = [
    ("herb", "contains", "compound"),
    ("compound", "acts_on", "target"),
    ("target", "associated_with", "disease"),
    ("drug", "targets_protein", "target"),
    ("drug", "treats", "disease"),
    ("target", "interacts_with", "target"),
    ("compound", "may_treat", "disease"),
]

# Aliases used to match ChEMBL (MeSH/EFO) indication terms to the 31 disease
# nodes of the curated graph. Matching is case-insensitive and only exact
# matches after normalisation are accepted in order to avoid false links.
DISEASE_ALIASES = {
    "Rheumatoid arthritis": ["rheumatoid arthritis", "arthritis, rheumatoid"],
    "Hypertension": ["hypertension", "essential hypertension"],
    "Type 2 diabetes mellitus": ["type 2 diabetes mellitus", "diabetes mellitus, type 2",
                                 "diabetes mellitus type 2", "type ii diabetes mellitus"],
    "Parkinson disease": ["parkinson disease", "parkinson's disease", "parkinsonian disorders"],
    "Alzheimer disease": ["alzheimer disease", "alzheimer's disease"],
    "Pancreatic cancer": ["pancreatic cancer", "pancreatic neoplasms", "pancreatic carcinoma",
                          "pancreatic ductal carcinoma", "pancreatic ductal adenocarcinoma"],
    "Coronary artery disease": ["coronary artery disease", "coronary artery diseases",
                                "coronary atherosclerosis", "coronary heart disease"],
    "Allergic rhinitis": ["allergic rhinitis"],
    "COVID-19": ["covid-19", "covid19", "sars-cov-2 infection"],
    "Rheumatic disease": ["rheumatic diseases", "rheumatic disease", "rheumatism"],
    "Osteoarthritis": ["osteoarthritis", "osteoarthritis, knee"],
    "Systemic lupus erythematosus": ["systemic lupus erythematosus", "lupus erythematosus, systemic"],
    "Atopic dermatitis": ["atopic dermatitis", "dermatitis, atopic", "eczema, atopic"],
    "Gastric ulcer": ["gastric ulcer", "stomach ulcer", "ulcer, gastric"],
    "Schizophrenia": ["schizophrenia"],
    "Liver cirrhosis": ["liver cirrhosis", "cirrhosis, liver", "hepatic cirrhosis"],
    "Lung cancer": ["lung cancer", "lung neoplasms", "lung carcinoma", "non-small cell lung cancer",
                    "non-small-cell lung carcinoma", "small cell lung cancer"],
    "Breast cancer": ["breast cancer", "breast neoplasms", "breast carcinoma"],
    "Diabetic nephropathy": ["diabetic nephropathy", "diabetic kidney disease",
                             "diabetic nephropathies"],
    "Ulcerative colitis": ["ulcerative colitis", "colitis, ulcerative"],
    "Chronic obstructive pulmonary disease": ["chronic obstructive pulmonary disease", "copd",
                                               "pulmonary disease, chronic obstructive"],
    "Coronary restenosis": ["coronary restenosis", "restenosis, coronary"],
    "Chronic kidney disease": ["chronic kidney disease", "renal insufficiency, chronic",
                               "kidney failure, chronic"],
    "Hypertensive heart disease": ["hypertensive heart disease", "hypertensive heart diseases"],
    "Hematologic cancer": ["hematologic cancer", "hematologic neoplasms", "hematological malignancy",
                           "leukemia", "lymphoma", "multiple myeloma"],
    "Heart failure": ["heart failure", "cardiac failure", "heart decompensation"],
    "Sepsis": ["sepsis", "septicemia", "septic shock"],
    "Major depressive disorder": ["major depressive disorder", "depressive disorder, major",
                                  "depression, major"],
    "Crohn disease": ["crohn disease", "crohn's disease", "crohn's enteritis"],
    "Systemic sclerosis": ["systemic sclerosis", "scleroderma, systemic"],
    "Colorectal cancer": ["colorectal cancer", "colorectal neoplasms", "colon cancer",
                          "rectal cancer", "colon neoplasms"],
}


def load_json(name):
    path = os.path.join(RAW, name)
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def main():
    compounds_data = load_json("guizhi_tang_compounds.json")
    cti = load_json("compound_target_interactions.json")
    tda = load_json("target_disease_associations.json")
    ppi = load_json("string_ppi_real.json")

    indications_path = os.path.join(RAW, "drug_indications_chembl.json")
    if os.path.exists(indications_path):
        with open(indications_path, encoding="utf-8") as fh:
            chembl_indications = json.load(fh)
    else:  # pragma: no cover - only used when the cached ChEMBL query is absent
        chembl_indications = {}

    # ------------------------------------------------------------------ nodes
    herbs = list(compounds_data["herbs"].keys())

    compound_herbs = defaultdict(set)
    compound_obdl = {}
    for herb in herbs:
        for comp in compounds_data["herbs"][herb]["compounds"]:
            compound_herbs[comp["name"]].add(herb)
            compound_obdl.setdefault(comp["name"], (comp["ob"], comp["dl"]))
    compounds = sorted(compound_herbs)

    target_details = {t["gene"] for t in cti["target_details"]}
    cti_compounds = {c["name"] for c in cti["compounds"]}
    cti_targets = {t for c in cti["compounds"] for t in c["targets"]}
    disease_targets = {t for d in tda["diseases"] for t in d["associated_targets"]}
    targets = sorted(cti_targets | disease_targets | {t for t in target_details if t in cti_targets})

    diseases = [d["name"] for d in tda["diseases"]]
    drug_targets = {d["name"]: [t for t in d["targets"] if t in set(targets)]
                    for d in tda["drugbank_drugs_sharing_targets"]}
    drugs = sorted(drug_targets)

    node_order = {
        "herb": herbs,
        "compound": compounds,
        "target": targets,
        "disease": diseases,
        "drug": drugs,
    }
    index = {nt: {name: i for i, name in enumerate(names)} for nt, names in node_order.items()}

    # ------------------------------------------------------------------ edges
    edges = {f"{s}-{r}-{d}": [] for s, r, d in RELATIONS}
    seen = {key: set() for key in edges}

    def add_edge(src_type, rel, dst_type, src, dst):
        key = f"{src_type}-{rel}-{dst_type}"
        pair = (index[src_type][src], index[dst_type][dst])
        if pair not in seen[key]:
            seen[key].add(pair)
            edges[key].append(list(pair))

    for herb in herbs:
        for comp in compounds_data["herbs"][herb]["compounds"]:
            add_edge("herb", "contains", "compound", herb, comp["name"])

    for comp in cti["compounds"]:
        for target in comp["targets"]:
            add_edge("compound", "acts_on", "target", comp["name"], target)

    for disease in tda["diseases"]:
        for target in disease["associated_targets"]:
            add_edge("target", "associated_with", "disease", target, disease["name"])

    for drug in tda["drugbank_drugs_sharing_targets"]:
        for target in drug["targets"]:
            if target in index["target"]:
                add_edge("drug", "targets_protein", "target", drug["name"], target)

    # real indications from ChEMBL (only used when the fetch succeeded)
    indication_edges = 0
    disease_norm = {}
    for disease in diseases:
        for alias in DISEASE_ALIASES.get(disease, [disease.lower()]):
            disease_norm[alias] = disease
    for drug, rec in chembl_indications.items():
        if drug not in index["drug"]:
            continue
        for ind in rec.get("indications", []):
            for field in ("mesh_heading", "efo_term"):
                term = (ind.get(field) or "").strip().lower()
                if term in disease_norm:
                    add_edge("drug", "treats", "disease", drug, disease_norm[term])
                    indication_edges += 1
                    break

    # real STRING PPI, restricted to the 42 protein targets of the graph
    ppi_set = set()
    for row in ppi:
        a, b = row["preferredName_A"], row["preferredName_B"]
        if a in index["target"] and b in index["target"] and a != b:
            ppi_set.add(tuple(sorted((a, b))))
    # stored once per undirected pair; reverse edges are added when the PyG
    # HeteroData object is created (torch_geometric.transforms.ToUndirected)
    for a, b in sorted(ppi_set):
        add_edge("target", "interacts_with", "target", a, b)

    # known compound-disease links (supervision target only; never message-passed)
    labels = []
    for comp, disease_list in tda.get("compound_disease_known_links", {}).items():
        if comp not in index["compound"]:
            continue
        for disease in disease_list:
            if disease in index["disease"]:
                labels.append([index["compound"][comp], index["disease"][disease]])
    labels = sorted({tuple(x) for x in labels})
    labels = [list(x) for x in labels]

    # ---------------------------------------------------------------- features
    c_deg = Counter()
    t_deg = Counter()
    for c in cti["compounds"]:
        for target in c["targets"]:
            c_deg[c["name"]] += 1
            t_deg[target] += 1
    t_dis = Counter()
    for d in tda["diseases"]:
        for target in d["associated_targets"]:
            t_dis[target] += 1
    t_ppi = Counter()
    for a, b in ppi_set:
        t_ppi[a] += 1
        t_ppi[b] += 1
    d_drug = Counter()
    for drug, rec in chembl_indications.items():
        for ind in rec.get("indications", []):
            for field in ("mesh_heading", "efo_term"):
                term = (ind.get(field) or "").strip().lower()
                if term in disease_norm:
                    d_drug[disease_norm[term]] += 1
                    break
    drug_t_deg = Counter()
    for drug, tl in drug_targets.items():
        drug_t_deg[drug] = len(tl)

    def norm(counter, names):
        mx = max([counter[n] for n in names] + [1])
        return {n: counter[n] / mx for n in names}

    c_deg_n = norm(c_deg, compounds)
    t_deg_n = norm(t_deg, targets)
    t_dis_n = norm(t_dis, targets)
    t_ppi_n = norm(t_ppi, targets)
    d_target_deg = Counter()
    for d in tda["diseases"]:
        d_target_deg[d["name"]] = len(d["associated_targets"])
    d_deg_n = norm(d_target_deg, diseases)
    d_drug_n = norm(d_drug, diseases)
    drug_deg_n = norm(drug_t_deg, drugs)

    features = {}
    features["herb"] = np.eye(len(herbs), dtype=np.float32).tolist()
    features["compound"] = [
        [compound_obdl[c][0] / 100.0, compound_obdl[c][1],
         c_deg_n[c], len(compound_herbs[c]) / 5.0] for c in compounds
    ]
    features["target"] = [[t_deg_n[t], t_dis_n[t], t_ppi_n[t]] for t in targets]
    features["disease"] = [[d_deg_n[d], d_drug_n[d]] for d in diseases]
    features["drug"] = (
        [[drug_deg_n[d], 1.0 if d in chembl_indications else 0.0]
         + [1.0 if i == j else 0.0 for j in range(len(drugs))] for i, d in enumerate(drugs)]
    )

    stats = {
        "nodes": {k: len(v) for k, v in node_order.items()},
        "edges": {k: len(v) for k, v in edges.items()},
        "n_labels": len(labels),
        "indicators": {"chembl_indication_edges": indication_edges,
                       "string_ppi_pairs": len(ppi_set)},
    }
    stats["total_nodes"] = sum(stats["nodes"].values())
    stats["total_edges"] = sum(stats["edges"].values())

    graph = {
        "nodes": node_order,
        "features": features,
        "edges": edges,
        "labels": labels,
        "stats": stats,
        "provenance": {
            "builder": "src/graph_builder.py (v2.0 revision)",
            "inputs": ["guizhi_tang_compounds.json", "compound_target_interactions.json",
                       "target_disease_associations.json", "string_ppi_real.json",
                       "drug_indications_chembl.json"],
            "notes": ("Compound-disease labels are used only as supervision and are not "
                      "included in message passing. Drug-disease edges are real ChEMBL "
                      "indications, not inferred from shared targets."),
        },
    }
    os.makedirs(PROCESSED, exist_ok=True)
    with open(os.path.join(PROCESSED, "graph_data.json"), "w", encoding="utf-8") as fh:
        json.dump(graph, fh, indent=2)
    with open(os.path.join(PROCESSED, "graph_statistics.json"), "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)

    print(json.dumps(stats, indent=2))
    print("graph written to", os.path.join(PROCESSED, "graph_data.json"))


if __name__ == "__main__":
    main()

