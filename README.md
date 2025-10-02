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
- Cold-pool detection (optional in `main.py`) now seeds rainy regions from near-surface rain-water mixing ratio `q_r` (from `rico.r.nc`) using a threshold in kg/kg; centroids can be advected by mean low-level wind; boundaries come from the zero-contour of the radial second derivative of density potential temperature at a 10-minute lag.

Sample gif (tracking mean winds)

![Cold pools tracking](cold_pools_sample.gif)
