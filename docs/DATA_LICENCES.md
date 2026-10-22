# Data provenance, licences and legal use

This file documents how each external data source used in the revision was obtained and how it is
redistributed. The authors confirm that all data were downloaded from legal public sources and used
in accordance with the providers' terms of use. Where a provider restricts redistribution, the
public release contains only processed/derived tables and download scripts, not the original
database records.

## Sources and terms

| Source | What was used | Terms / licence summary | What is redistributed |
|--------|---------------|--------------------------|-----------------------|
| TCMSP (https://tcmsp-e.com/) | herb lists, compounds, OB/DL, compound-target records | Free for academic research under TCMSP terms; cite Ru et al. 2014 | Curated per-herb compound/target table (small derived subset) with citation; users must also comply with TCMSP terms |
| STRING v12.0 (https://string-db.org/) | PPI pairs among the 47 targets | STRING data are available under a Creative Commons Attribution 4.0 licence (CC BY 4.0); cite Szklarczyk et al. | `string_ppi_real.json` (derived subset) with attribution |
| ChEMBL (https://www.ebi.ac.uk/chembl/) | approved-drug indications | ChEMBL data are released under a Creative Commons Attribution-ShareAlike licence; cite Zdrazil et al. | `drug_indications_chembl.json` (derived subset) with attribution and share-alike notice |
| DrugBank (https://go.drugbank.com/) | drug-target annotations | DrugBank data are subject to the DrugBank licence; academic use is permitted through the public database, but raw data may not be redistributed without permission | Only derived drug-target edges inside the released graph; no raw DrugBank records |
| DisGeNET (https://www.disgenet.org/) | gene-disease associations | DisGeNET data are free for academic/non-commercial use; redistribution of raw records is restricted | Only derived disease-target associations in the released graph; no raw DisGeNET records |
| OMIM (https://omim.org/) | disease-gene records | OMIM is copyrighted by Johns Hopkins University; use is subject to OMIM terms; raw redistribution is restricted | Only derived target-disease associations in the released graph; no raw OMIM records |
| KEGG (https://www.kegg.jp/) | pathway names, gene-pathway links, gene symbols | KEGG data are for academic use under the KEGG terms; redistribution of KEGG data requires permission | **No raw KEGG files are redistributed.** Scripts download the mapping directly from the KEGG REST API at run time; only derived enrichment statistics (p-values, fold enrichment, FDR) are reported |
| Enrichr (https://maayanlab.cloud/Enrichr/) | GO enrichment | Public web service; cite Kuleshov et al. | Only derived enrichment statistics |
| UniProt (https://www.uniprot.org/) | gene symbols/annotations | UniProt data are available under a Creative Commons Attribution 4.0 licence (CC BY 4.0) | Gene symbol mapping only |
| PubChem (https://pubchem.ncbi.nlm.nih.gov/) | 3D ligand structures | PubChem data are publicly available under the PubChem terms of use | Ligand SDF/PDBQT files are obtained at run time by the docking script; not redistributed |
| RCSB PDB (https://www.rcsb.org/) | protein structures | PDB data are available under the wwPDB data usage policy (open access) | Structures are downloaded at run time by the docking script; not redistributed |
| AutoDock Vina (https://github.com/ccsb-scripps/AutoDock-Vina) | docking engine | Apache License 2.0 | `tools/vina_1.2.7_win.exe` is redistributed with attribution under Apache 2.0 |
| Open Babel | chemical file conversion | GNU GPL v2 | Used as an external tool; not redistributed |
| RDKit | chemistry toolkit | BSD 3-Clause | Used as an installed package; not redistributed |

## Human data and ethics

No human participants, human material or animals were involved. All human-related data are
aggregated, de-identified records from public databases (DisGeNET, OMIM, DrugBank). No individual-
level or identifiable data are included or redistributed. No ethics approval or informed consent is
required for secondary analysis of public, de-identified aggregate data; this is stated in the
manuscript's Ethics statement.

## Code licence

Our original code is released under the MIT licence (see `LICENSE`). Third-party tools retain their
own licences as listed above. Users of the released data are responsible for complying with the
terms of the original databases and for citing them.

## Reproducibility without redistribution

`src/enrichment.py` downloads the KEGG mapping from the KEGG REST API when the cached files are not
present. `src/docking.py` downloads PDB structures from RCSB and ligand conformers from PubChem at
run time. `src/fetch_drug_indications.py` downloads ChEMBL indications. `docs/DATA.md` lists all
access dates and query parameters. This allows exact regeneration of the analysed data by users who
accept the providers' terms.
