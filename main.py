"""Run cold-pool panel plotting with a simple config.

Edit the config dict below and run:

    python main.py

This produces two PDFs: one with actual winds, and one with mean-wind-subtracted winds.
"""

from typing import Optional, Tuple
import os

from src.plot_cold_pools import main as run_panels
from src.cold_pool_detect import run_detection, track_pools_over_time


# config
config = {
    # where the RICO files live
    "data_root": os.environ.get("RICO_DATA", "/Users/jure/PhD/coding/RICO_1hr"),

    # vertical level index
    "z_index": 2,

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
    
    # gif options
    "make_tracking_gif": True,           # single-panel, minute-by-minute GIF
    "gif_minutes": 20,                    # number of frames (minutes)
    # region center and half-window in km in domain coords
    "gif_center_km": (0.0, 0.0),          # (x0_km, y0_km)
    "gif_half_window_km": (2.0, 2.0),     # (hx_km, hy_km)
    "gif_outfile": "cold_pools_tracking.gif",
    
    # detection and tracking options
    "detect_pools": True,
    "detect_minutes": 60,                # how many consecutive minutes to process
    # seeding from rain-water mixing ratio
    "qr_thresh_kgkg": None,              # None -> adaptive threshold per minute
    "near_surface_levels": 1,            # used if qr_max_height_m is None
    "qr_max_height_m": 400.0,            # max over z <= this height (m)
    "sigma_rain_smooth_m": 150.0,        # Gaussian smoothing for qr_max (m)
    "min_pool_area_km2": 0.2,            # minimum rainy region area (km^2)
    # lag and hessian
    "lag_minutes": 7,
    "hessian_sigma_m": 150.0,            # Gaussian sigma for Hessian (m)
    "use_advection_correction": True,    # advect by mean low-level wind
    # acceptance gates
    "proximity_factor": 1.5,
    "cover_rainy_min": 0.60,
    "cover_poly_min": 0.10,
    "aspect_min": 0.40,
    "solidity_min": 0.55,
    # tracking gates
    "track_max_dist_factor": 2.0,        # × previous eq. radius
    "track_min_overlap": 0.30,
    # output
    "detect_output_prefix": "pools",    # prefix for diagnostic plots
    # rain vs recognition GIF
    "make_rain_vs_recognition_gif": True,
    "rvrs_minutes": 60,
    "rvrs_outfile": "rain_vs_recognition.gif",
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
        make_tracking_gif=config["make_tracking_gif"],
        gif_minutes=config["gif_minutes"],
        gif_center_km=config["gif_center_km"],
        gif_half_window_km=config["gif_half_window_km"],
        gif_outfile=config["gif_outfile"],
    )

    # Optional: detect and track cold pools
    if config.get("detect_pools", False):
        pools_by_time = run_detection(
            data_root=config["data_root"],
            start_index=config["start_index"],
            minutes=config["detect_minutes"],
            z_index=config["z_index"],
            qr_thresh_kgkg=config["qr_thresh_kgkg"],
            near_surface_levels=config["near_surface_levels"],
            qr_max_height_m=config["qr_max_height_m"],
            min_area_km2=config["min_pool_area_km2"],
            lag_minutes=config["lag_minutes"],
            hessian_sigma_m=config["hessian_sigma_m"],
            sigma_rain_smooth_m=config["sigma_rain_smooth_m"],
            use_advection_correction=config["use_advection_correction"],
            proximity_factor=config["proximity_factor"],
            cover_rainy_min=config["cover_rainy_min"],
            cover_poly_min=config["cover_poly_min"],
            aspect_min=config["aspect_min"],
            solidity_min=config["solidity_min"],
            output_prefix=config["detect_output_prefix"],
            make_plots=True,
            colormap=config["colormap"],
            arrow_subsample=config["arrow_subsample"],
            arrow_scale=config["arrow_scale"],
        )
        tracks = track_pools_over_time(
            pools_by_time,
            data_root=config["data_root"],
            z_index=config["z_index"],
            use_advection_correction=config["use_advection_correction"],
            min_overlap=config["track_min_overlap"],
            track_max_dist_factor=config["track_max_dist_factor"],
        )
        print(f"Detected tracks: {len(tracks)}")

    # Optional: side-by-side rain vs recognition GIF
    if config.get("make_rain_vs_recognition_gif", False):
        from src.cold_pool_detect import render_rain_vs_recognition_gif
        gif_path = render_rain_vs_recognition_gif(
            data_root=config["data_root"],
            start_index=config["start_index"],
            minutes=config["rvrs_minutes"],
            z_index=config["z_index"],
            qr_max_height_m=config["qr_max_height_m"],
            sigma_rain_smooth_m=config["sigma_rain_smooth_m"],
            min_area_km2=config["min_pool_area_km2"],
            lag_minutes=config["lag_minutes"],
            hessian_sigma_m=config["hessian_sigma_m"],
            use_advection_correction=config["use_advection_correction"],
            proximity_factor=config["proximity_factor"],
            cover_rainy_min=config["cover_rainy_min"],
            cover_poly_min=config["cover_poly_min"],
            aspect_min=config["aspect_min"],
            solidity_min=config["solidity_min"],
            arrow_subsample=config["arrow_subsample"],
            arrow_scale=config["arrow_scale"],
            outfile=config["rvrs_outfile"],
            colormap=config["colormap"],
        )
        print(f"Saved: {gif_path}")


if __name__ == "__main__":
    main()
