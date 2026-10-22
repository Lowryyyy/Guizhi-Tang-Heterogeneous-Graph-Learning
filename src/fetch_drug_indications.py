"""Fetch real approved-drug indications from ChEMBL for the drug layer of the graph."""
import json
import os
import time
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 research-script"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def main():
    tda = json.load(open(os.path.join(RAW, "target_disease_associations.json"), encoding="utf-8"))
    drugs = []
    seen = set()
    for d in tda["drugbank_drugs_sharing_targets"]:
        if d["name"] not in seen:
            seen.add(d["name"])
            drugs.append(d)
    overrides = {"Dasatinib": "CHEMBL1421", "Methotrexate": "CHEMBL34259"}
    out = {}
    for d in drugs:
        name = d["name"]
        rec = {"query": name, "indications": []}
        if name in overrides:
            cid = overrides[name]
            rec["chembl_id"] = cid
            ind = get(f"https://www.ebi.ac.uk/chembl/api/data/drug_indication.json?molecule_chembl_id={cid}&limit=1000")
            rec["indications"] = [
                {"mesh_id": x.get("mesh_id"), "mesh_heading": x.get("mesh_heading"),
                 "efo_id": x.get("efo_id"), "efo_term": x.get("efo_term"),
                 "max_phase": x.get("max_phase_for_ind")}
                for x in ind.get("drug_indications", [])
            ]
            out[name] = rec
            print(f"{name:14s} {cid:12s} indications={len(rec['indications']):3d} (override)")
            time.sleep(0.2)
            continue
        try:
            q = urllib.parse.quote(name)
            search = get(f"https://www.ebi.ac.uk/chembl/api/data/molecule/search.json?q={q}&limit=10")
            rows = search.get("molecules", [])
            exact = [x for x in rows if (x.get("pref_name") or "").lower() == name.lower()]
            pick = exact[0] if exact else None
            if pick is None:
                # tolerate minor spelling variants, but only when unambiguous
                cands = [x for x in rows if (x.get("pref_name") or "").lower().startswith(name.lower()[:6])]
                pick = cands[0] if len(cands) == 1 else None
            if pick:
                cid = pick["molecule_chembl_id"]
                rec["chembl_id"] = cid
                rec["pref_name"] = pick.get("pref_name")
                ind = get(f"https://www.ebi.ac.uk/chembl/api/data/drug_indication.json?molecule_chembl_id={cid}&limit=1000")
                rec["indications"] = [
                    {"mesh_id": x.get("mesh_id"), "mesh_heading": x.get("mesh_heading"),
                     "efo_id": x.get("efo_id"), "efo_term": x.get("efo_term"),
                     "max_phase": x.get("max_phase_for_ind")}
                    for x in ind.get("drug_indications", [])
                ]
            else:
                rec["error"] = "no unambiguous ChEMBL drug match"
        except Exception as exc:  # pragma: no cover - network robustness
            rec["error"] = str(exc)[:200]
        out[name] = rec
        print(f"{name:14s} {rec.get('chembl_id','-'):12s} indications={len(rec['indications']):3d} "
              f"{rec.get('error','')}")
        time.sleep(0.2)
    with open(os.path.join(RAW, "drug_indications_chembl.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("saved", len(out), "drug records")


if __name__ == "__main__":
    main()

