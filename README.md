Cold-pool (and maybe rain visuals) for RICO
=============================

This tool makes a basic plot of cold-pools. Rain plots will be added. 

Setup
- NetCDF file in a folder (default needs to be changed!).
- Install: `pip install -r requirements.txt`.

Run
- Edit `main.py` by adjusting `config`.
- Run with `python main.py`.

Notes
- If `rico.r.nc` is missing, rain is set to zero.
- The second PNG uses wind anomalies (mean wind removed).
