"""Run cold-pool panel plotting with a simple config.

Edit the config dict below and run:

    python main.py

This produces two PDFs: one with actual winds, and one with mean-wind-subtracted winds.
"""

from typing import Optional, Tuple
import os

from src.plot_cold_pools import main as run_panels


# config
config = {
    # where the RICO files live
    "data_root": os.environ.get("RICO_DATA", "/Users/jure/PhD/coding/RICO_1hr"),

    # vertical level index
    "z_index": 1,

    # time index of the first panel and the time in indices between panels
    "start_index": 0,
    "time_stride": 1,
    "n_panels": 3, 

    # buoyancy reference; if True use clear air (ql + qr < threshold), else domain mean
    "use_clear_air_reference": True,
    # threshold in kg/kg (default is 0.00005 kg/kg)
    "threshold_kgkg": 1e-5,

    # rain file name; warn and assume zero if missing
    "rain_filename": "rico.r.nc",

    # optional zoom in k (-X,+X). None for full domain
    "xlim_km": (-2,2),
    "ylim_km": (-2,2),

    # plot style
    "colormap": None,        # None is auto
    "arrow_subsample": 4,    # arrow density
    "arrow_scale": 40,      # arrow scale

    # outputs
    "outfile": "cold_pools_panels.png",
    "outfile_wind_anom": "cold_pools_panels_anom.png",
}


def main():
    # check data_root path; fail fast
    if not os.path.isdir(config["data_root"]):
        raise FileNotFoundError(f"data_root does not exist: {config['data_root']}")

    run_panels(
        data_root=config["data_root"],
        start_index=config["start_index"],
        time_stride=config["time_stride"],
        n_panels=config["n_panels"],
        z_index=config["z_index"],
        xlim_km=config["xlim_km"],
        ylim_km=config["ylim_km"],
        use_clear_air_reference=config["use_clear_air_reference"],
        threshold_kgkg=config["threshold_kgkg"],
        rain_filename=config["rain_filename"],
        colormap=config["colormap"],
        arrow_subsample=config["arrow_subsample"],
        arrow_scale=config["arrow_scale"],
        outfile=config["outfile"],
        outfile_wind_anom=config["outfile_wind_anom"],
    )


if __name__ == "__main__":
    main()
