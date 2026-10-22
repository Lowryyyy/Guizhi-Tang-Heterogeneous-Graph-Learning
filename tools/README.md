# External docking engine

The analysis used **AutoDock Vina 1.2.7** (Apache License 2.0), which is not
redistributed in this repository. Download the Windows executable from:

https://github.com/ccsb-scripps/AutoDock-Vina/releases

Place `vina_1.2.7_win.exe` in this folder and run `python src/docking.py`.
Receptor PDB files and ligand SDF files are downloaded by the script from RCSB PDB
and PubChem; they are not redistributed here.
