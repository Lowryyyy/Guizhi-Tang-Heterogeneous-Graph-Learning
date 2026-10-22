# Guizhi Tang heterogeneous graph learning (revision v2.0)

Code and data for the revised manuscript:

> **Heterogeneous graph learning prioritises candidate multi-target mechanisms of Guizhi Tang**
> Scientific Reports, manuscript ID `aed4d08b-e589-4dda-9fa3-3cefc6abc0c1`

Repository URL: https://github.com/Lowryyyy/Guizhi-Tang-Heterogeneous-Graph-Learning

This release supersedes the code deposited with the original submission. Compared with the original
pipeline it (i) removes compound-disease label edges from message passing, (ii) replaces synthetic
drug-disease edges with curated ChEMBL indications, (iii) restores the real STRING PPI layer,
(iv) reports leakage-free five-seed and five-fold results with degree-only, shared-target,
label-permutation, edge-ablation and cold-start controls, (v) recomputes KEGG/GO enrichment from the
source APIs with annotation-density controls, and (vi) re-runs AutoDock Vina docking with decoy
comparisons, ligand-efficiency and molecular-weight/logP-adjusted analyses.

## Contents

| Path | Description |
|---|---|
| `src/graph_builder.py` | Builds the 189-node, 1,322-edge heterogeneous graph (5 herbs, 77 compounds, 47 targets, 31 diseases, 29 drugs) and the 154 compound-disease labels |
| `src/model.py` | MLP, homogeneous GCN, R-GCN and heterogeneous GNN (PyTorch Geometric `GATConv` per relation) |
| `src/train.py` | Five-seed training and leakage-free evaluation |
| `src/bootstrap_ci.py` | Bootstrap 95% confidence intervals and paired bootstrap comparisons |
| `src/degree_sensitivity.py` | Degree-weighted hard-negative sampling sensitivity analysis |
| `src/enrichment.py` | KEGG REST hypergeometric enrichment with BH-FDR, expected overlap, fold enrichment and permutation P values |
| `src/docking.py` | AutoDock Vina docking and decoy pipeline |
| `src/ligand_efficiency.py` | RDKit ligand descriptors, ligand efficiency and MW/logP-adjusted docking comparison |
| `src/figures.py` | Generates all publication figures |
| `data/processed/graph_data.json` | Final graph: nodes, typed edges and labels |
| `data/raw/` | Derived input tables (third-party raw records that cannot be redistributed are downloaded at run time) |
| `data/docking/` | Docking pairs and complete Vina run records |
| `results/` | All JSON/NPY/CSV result tables used in the manuscript |
| `figures/` | `Fig1.png`-`Fig8.png` (600 dpi) and editable SVG versions |
| `docs/DATA.md` | Data provenance and variable descriptions |
| `docs/DATA_LICENCES.md` | Source-by-source licence and redistribution statement |
| `docs/reference_verification.csv` | 70 candidate references with DOI and Crossref verification status |
| `zenodo_metadata.json` | Metadata for the Zenodo archival copy (DOI assigned at publication) |

## Quick start

```bash
python -m pip install -r requirements.txt
python src/graph_builder.py      # rebuild data/processed/graph_data.json
python src/train.py              # five-seed models and controls
python src/bootstrap_ci.py       # bootstrap confidence intervals
python src/degree_sensitivity.py # degree-weighted negative sampling sensitivity
python src/enrichment.py         # KEGG REST + Enrichr GO enrichment
python src/docking.py            # requires vina in tools/ (see tools/README.md)
python src/ligand_efficiency.py  # ligand-property confound analysis
python src/figures.py            # regenerate Fig1-Fig8
```

## Data and code availability

- The exact release used for every number in the manuscript is this repository at tag
  **`v2.0-revision`**.
- An archival copy with a DOI is prepared on Zenodo; the DOI is cited in the Code Availability
  section of the manuscript.
- Third-party raw data are not redistributed when their licences do not permit it. KEGG mappings are
  downloaded by `src/enrichment.py`; PDB structures and PubChem ligands are downloaded by
  `src/docking.py`. See `docs/DATA_LICENCES.md`.
- The study uses only public, de-identified aggregate data; no human participants or animals were
  involved and no ethics approval was required.

## Citation

If you use this code or data, please cite the manuscript and this repository (tag `v2.0-revision`).
See `CITATION.cff` for machine-readable metadata.

## Contact

Feng Luan, Xinjiang Second Medical College, Karamay 834000, Xinjiang, China.
E-mail: 783934728@qq.com
