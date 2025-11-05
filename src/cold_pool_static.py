"""Single-timestep cold pool detection for large static fields.

This module detects cold pools in individual snapshots without temporal tracking.
Designed for large domains (e.g., 50km×50km) with hourly or sparse temporal resolution.

Usage:
    python -m src.cold_pool_static --data_root /path/to/data --start_index 0 --n_steps 5
"""

from __future__ import annotations
import os
from typing import List, Optional, Tuple

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath

from .cold_pool_detect import (
    _rain_mask_from_qr,
    _theta_rho_field,
    _winds_at_level_by_index,
    _d2r_field,
    _zero_contour_paths_km,
    _region_properties,
    _time_values_and_count,
    Pool,
)
from .physics import relative_humidity_from_p_T_qv, _ensure_kgkg


def _open(path: str) -> xr.Dataset:
    return xr.open_dataset(path, decode_times=False)


def detect_cold_pools_static(
    data_root: str,
    time_index: int,
    z_index: int = 0,
    # Rain seeding parameters
    near_surface_levels: int = 1,
    qr_thresh_kgkg: Optional[float] = None,
    qr_max_height_m: float = 400.0,
    sigma_rain_smooth_m: float = 150.0,
    min_pool_area_km2: float = 0.2,
    # Boundary detection parameters
    hessian_sigma_m: float = 150.0,
    auto_tune_sigma: bool = True,
    # Acceptance gates
    proximity_factor: float = 1.5,
    cover_rainy_min: float = 0.60,
    cover_poly_min: float = 0.10,
    aspect_min: float = 0.40,
    solidity_min: float = 0.55,
    # Diagnostics
    plot_diagnostics: bool = False,
    diagnostic_output_dir: str = "plots/diagnostics",
) -> Tuple[List[Pool], List[Tuple[float, float]]]:
    """Detect cold pools in a single timestep.
    
    Args:
        data_root: Path to NetCDF data directory
        time_index: Time index to process
        z_index: Vertical level index (default: 0, lowest level)
        near_surface_levels: Number of near-surface levels for rain detection
        qr_thresh_kgkg: Rain threshold in kg/kg (None = adaptive)
        qr_max_height_m: Maximum height for rain detection
        sigma_rain_smooth_m: Gaussian smoothing sigma for rain field
        min_pool_area_km2: Minimum cold pool area
        hessian_sigma_m: Gaussian sigma for Hessian computation
        auto_tune_sigma: Automatically tune hessian_sigma_m from data
        proximity_factor: Centroid proximity gate (× rain equivalent diameter)
        cover_rainy_min: Minimum fraction of rain region covered by polygon
        cover_poly_min: Minimum fraction of polygon covering rain
        aspect_min: Minimum aspect ratio for polygon acceptance
        solidity_min: Minimum solidity for polygon acceptance
        plot_diagnostics: Generate diagnostic plots for first few centroids
        diagnostic_output_dir: Where to save diagnostic plots
    
    Returns:
        Tuple of (detected pools, rain centroids)
    """
    from scipy import ndimage
    from scipy.ndimage import binary_fill_holes
    
    km = 1000.0
    
    # Get grid and resolution
    with _open(os.path.join(data_root, "rico.p.nc")) as dsp:
        xt = dsp["xt"].values
        yt = dsp["yt"].values
    dx = float(np.mean(np.diff(xt)))
    dy = float(np.mean(np.diff(yt)))
    
    # 1) Rain seeding from q_r with smoothing + adaptive threshold
    rainy, xt_m, yt_m, diag = _rain_mask_from_qr(
        data_root=data_root,
        t_index=time_index,
        near_surface_levels=near_surface_levels,
        qr_thresh_kgkg=qr_thresh_kgkg,
        qr_max_height_m=qr_max_height_m,
        sigma_rain_smooth_m=sigma_rain_smooth_m,
        min_area_km2=min_pool_area_km2,
    )
    
    print(f"SEED t={time_index}: post comps={diag['post_components']} pix={diag['post_pixels']} "
          f"area_km2={diag['post_area_km2']:.3f}; thresh={diag['qr_thresh']:.2e}")
    
    # Four-connected components and fill holes per region
    structure = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=int)
    labeled, nlab = ndimage.label(rainy, structure=structure)
    
    if nlab == 0:
        print(f"No rainy regions found at t={time_index}")
        return [], []
    
    # Remove small regions and fill holes
    mask_keep = np.zeros_like(rainy, dtype=bool)
    for lab in range(1, nlab+1):
        region = (labeled == lab)
        region = binary_fill_holes(region)
        area_km2 = region.sum() * dx * dy / (km*km)
        if area_km2 >= min_pool_area_km2:
            mask_keep |= region
    
    labeled, nlab = ndimage.label(mask_keep, structure=structure)
    if nlab == 0:
        print(f"No rainy regions after filtering at t={time_index}")
        return [], []
    
    # Centroids of rainy regions
    centroids_xy_km: List[Tuple[float, float]] = []
    for lab in range(1, nlab+1):
        region = (labeled == lab)
        if not np.any(region):
            continue
        iy, ix = np.nonzero(region)
        if ix.size == 0:
            continue
        x0_km = float(np.mean(xt[ix])) / km
        y0_km = float(np.mean(yt[iy])) / km
        centroids_xy_km.append((x0_km, y0_km))
    
    print(f"Found {len(centroids_xy_km)} rainy centroids at t={time_index}")
    
    # 2) Get theta_rho field at same time (no lag for static detection)
    theta_rho, xt_m, yt_m = _theta_rho_field(data_root, time_index, z_index)
    xt_km = xt_m / km
    yt_km = yt_m / km
    
    # Auto-tune hessian_sigma_m if requested
    if auto_tune_sigma and len(centroids_xy_km) > 0:
        from .cold_pool_diagnostics import auto_tune_hessian_sigma
        hessian_sigma_m = auto_tune_hessian_sigma(
            data_root, time_index, z_index, centroids_xy_km, max_centroids=5
        )
    
    # Generate diagnostic plots for first few centroids if requested
    if plot_diagnostics and len(centroids_xy_km) > 0:
        from .cold_pool_diagnostic_detailed import plot_detection_pipeline
        os.makedirs(diagnostic_output_dir, exist_ok=True)
        
        acceptance_params = {
            'proximity_factor': proximity_factor,
            'cover_rainy_min': cover_rainy_min,
            'cover_poly_min': cover_poly_min,
            'aspect_min': aspect_min,
            'solidity_min': solidity_min,
        }
        
        max_diag = min(5, len(centroids_xy_km))  # Diagnose up to 5 centroids
        for i, centroid_km in enumerate(centroids_xy_km[:max_diag]):
            outfile = os.path.join(
                diagnostic_output_dir, 
                f"pipeline_t{time_index:04d}_centroid{i+1}.png"
            )
            plot_detection_pipeline(
                data_root=data_root,
                time_index=time_index,
                z_index=z_index,
                centroid_km=centroid_km,
                centroid_index=i+1,
                rain_mask=rainy,
                labeled=labeled,
                xt_km=xt_km,
                yt_km=yt_km,
                xt_m=xt_m,
                yt_m=yt_m,
                hessian_sigma_m=hessian_sigma_m,
                acceptance_params=acceptance_params,
                output_file=outfile,
            )
            print(f"  Saved pipeline diagnostic: {outfile}")
    
    # 3) For each rainy centroid, build boundary from zero-contour of second radial derivative
    sigma_x_pix = float(hessian_sigma_m / dx)
    sigma_y_pix = float(hessian_sigma_m / dy)
    
    detected_pools: List[Pool] = []
    Xk, Yk = np.meshgrid(xt_km, yt_km)
    
    for i_cent, cxy_km in enumerate(centroids_xy_km):
        cx_km, cy_km = cxy_km
        
        # Compute radial second derivative field and its zero contours
        d2r = _d2r_field(theta_rho, (cx_km*km, cy_km*km), xt_m, yt_m,  (sigma_y_pix, sigma_x_pix))
        
        # Debug: check d2r field statistics
        d2r_finite = d2r[np.isfinite(d2r)]
        d2r_min = np.min(d2r_finite) if d2r_finite.size > 0 else 0.0
        d2r_max = np.max(d2r_finite) if d2r_finite.size > 0 else 0.0
        has_neg = np.any(d2r_finite < 0)
        has_pos = np.any(d2r_finite > 0)
        
        contour_paths = _zero_contour_paths_km(d2r, xt_km, yt_km)
        
        # Fallback: if no d2r contours, use theta_rho threshold
        if len(contour_paths) == 0:
            print(f"  Centroid {i_cent+1} at ({cx_km:.2f}, {cy_km:.2f}): no d2r contours "
                  f"[d2r range: {d2r_min:.2e} to {d2r_max:.2e}], trying theta_rho threshold fallback")
            
            # Use theta_rho anomaly: find region around centroid that's colder than surroundings
            theta_anom = theta_rho - np.nanmean(theta_rho)
            
            # Define threshold as coldest 10% within search radius
            R = np.sqrt((Xk - cx_km)**2 + (Yk - cy_km)**2)
            search_radius_km = 5.0  # Look within 5km of centroid
            local_region = (R <= search_radius_km)
            
            if np.any(local_region):
                local_vals = theta_anom[local_region]
                threshold = float(np.percentile(local_vals[np.isfinite(local_vals)], 10))
                
                # Find contours of this threshold
                all_contour_paths = _zero_contour_paths_km(theta_anom - threshold, xt_km, yt_km)
                
                # Filter: only keep contours that contain or are near the centroid
                contour_paths = []
                for seg_km in all_contour_paths:
                    # Check if centroid is inside this contour
                    path = MplPath(seg_km)
                    if path.contains_point((cx_km, cy_km)):
                        contour_paths.append(seg_km)
                        continue
                    
                    # Or check if contour is close to centroid (within search radius)
                    seg_arr = np.array(seg_km)
                    min_dist = np.min(np.sqrt((seg_arr[:,0] - cx_km)**2 + (seg_arr[:,1] - cy_km)**2))
                    if min_dist <= search_radius_km:
                        contour_paths.append(seg_km)
                
                if len(contour_paths) > 0:
                    print(f"    Found {len(contour_paths)} contours near centroid (from {len(all_contour_paths)} total) using theta_rho threshold={threshold:.3f} K")
                else:
                    print(f"    No contours near centroid (checked {len(all_contour_paths)} total)")
                    continue
            else:
                continue
        
        if len(contour_paths) == 0:
            continue
        
        print(f"  Centroid {i_cent+1} at ({cx_km:.2f}, {cy_km:.2f}): {len(contour_paths)} candidate contours")
        
        # Identify rainy region mask for overlap/proximity tests
        dists = []
        for lab in range(1, nlab+1):
            region = (labeled == lab)
            iy_reg, ix_reg = np.nonzero(region)
            if ix_reg.size == 0:
                continue
            x_km = float(np.mean(xt_m[ix_reg])) / km
            y_km = float(np.mean(yt_m[iy_reg])) / km
            dists.append(((cx_km - x_km)**2 + (cy_km - y_km)**2, lab))
        dists.sort()
        rain_lab = dists[0][1]
        rain_mask = (labeled == rain_lab)
        rain_area_km2 = rain_mask.sum() * dx * dy / (km*km)
        rain_eq_diam_km = 2.0 * np.sqrt(rain_area_km2 / np.pi)
        
        # Evaluate each candidate zero-contour polygon
        best_poly: Optional[Pool] = None
        best_centroid_dist = np.inf
        n_candidates = len(contour_paths)
        n_rejected_proximity = 0
        n_rejected_overlap = 0
        n_rejected_shape = 0
        n_accepted = 0
        
        # Pre-crop grid to region around centroid for faster point-in-polygon tests
        window_km = 10.0  # 10 km window around centroid
        x_crop_mask = (xt_km >= cx_km - window_km) & (xt_km <= cx_km + window_km)
        y_crop_mask = (yt_km >= cy_km - window_km) & (yt_km <= cy_km + window_km)
        xt_crop = xt_km[x_crop_mask]
        yt_crop = yt_km[y_crop_mask]
        Xk_crop, Yk_crop = np.meshgrid(xt_crop, yt_crop)
        
        for seg_km in contour_paths:
            path = MplPath(seg_km)
            # Test only points in cropped region
            inside_crop = path.contains_points(np.column_stack([Xk_crop.ravel(), Yk_crop.ravel()]))
            poly_mask_crop = inside_crop.reshape(Xk_crop.shape)
            
            # Map back to full grid
            poly_mask = np.zeros_like(Xk, dtype=bool)
            poly_mask[np.ix_(y_crop_mask, x_crop_mask)] = poly_mask_crop
            
            if poly_mask.sum() == 0:
                continue
            
            # Centroid of polygon
            iy, ix = np.nonzero(poly_mask)
            if ix.size == 0:
                continue
            cx_poly_km = float(np.mean(xt_km[ix]))
            cy_poly_km = float(np.mean(yt_km[iy]))
            cen_dist = np.hypot(cx_poly_km - cx_km, cy_poly_km - cy_km)
            
            # Proximity test
            if cen_dist > (proximity_factor * rain_eq_diam_km):
                n_rejected_proximity += 1
                continue
            
            # Overlap tests
            inter = (poly_mask & rain_mask).sum()
            rain_px = rain_mask.sum()
            poly_px = poly_mask.sum()
            if rain_px == 0 or poly_px == 0:
                continue
            cover_rainy = inter / rain_px
            cover_poly = inter / poly_px
            if not (cover_rainy >= cover_rainy_min and cover_poly >= cover_poly_min):
                n_rejected_overlap += 1
                continue
            
            # Shape quality
            props = _region_properties(poly_mask, dx, dy)
            if not (props["aspect"] >= aspect_min and props["solidity"] >= solidity_min):
                n_rejected_shape += 1
                continue
            
            n_accepted += 1
            
            # Compute fields within polygon
            theta_vals = theta_rho[poly_mask]
            mean_tr = float(np.nanmean(theta_vals))
            min_tr = float(np.nanmin(theta_vals))
            area_km2 = props["area"] / (km*km)
            eq_radius_km = props["eq_radius"] / km
            
            pool = Pool(
                time_index=int(time_index),
                centroid_km=(cx_poly_km, cy_poly_km),
                area_km2=float(area_km2),
                eq_radius_km=float(eq_radius_km),
                mean_theta_rho=float(mean_tr),
                min_theta_rho=float(min_tr),
                boundary_xy_km=seg_km,
            )
            
            if cen_dist < best_centroid_dist:
                best_poly = pool
                best_centroid_dist = cen_dist
        
        # Report rejection statistics for this centroid
        print(f"    Candidates: {n_candidates}, Rejected: proximity={n_rejected_proximity}, "
              f"overlap={n_rejected_overlap}, shape={n_rejected_shape}, Accepted={n_accepted}")
        
        if best_poly is not None:
            detected_pools.append(best_poly)
            print(f"    ✓ Pool detected: area={best_poly.area_km2:.2f} km2")
    
    print(f"Detected {len(detected_pools)} cold pools at t={time_index}")
    return detected_pools, centroids_xy_km


def _plot_full_field_overview(
    data_root: str,
    time_index: int,
    z_index: int,
    centroids_xy_km: List[Tuple[float, float]],
    pools: List[Pool],
    output_dir: str,
    colormap: Optional[str] = None,
):
    """Plot full-field theta_rho overview with rain centroids and detected pools."""
    km = 1000.0
    
    # Get theta_rho field
    theta_rho, xt_m, yt_m = _theta_rho_field(data_root, time_index, z_index)
    xt_km = xt_m / km
    yt_km = yt_m / km
    
    # Set up colormap
    if colormap is None:
        colormap = "cmo.thermal" if "cmo.thermal" in plt.colormaps() else "viridis"
    
    fig, ax = plt.subplots(figsize=(12, 10), constrained_layout=True)
    
    X, Y = np.meshgrid(xt_km, yt_km)
    
    # Plot theta_rho field
    vmin = np.nanpercentile(theta_rho, 2)
    vmax = np.nanpercentile(theta_rho, 98)
    im = ax.pcolormesh(X, Y, theta_rho, cmap=colormap, vmin=vmin, vmax=vmax, shading="auto")
    
    # Plot all rain centroids
    if len(centroids_xy_km) > 0:
        cx_all = [c[0] for c in centroids_xy_km]
        cy_all = [c[1] for c in centroids_xy_km]
        ax.scatter(cx_all, cy_all, c='yellow', marker='x', s=100, linewidths=2, 
                  label=f'{len(centroids_xy_km)} rain centroids', zorder=3)
    
    # Plot detected cold pool boundaries
    for i, pool in enumerate(pools):
        bnd = pool.boundary_xy_km
        label = 'Detected cold pool' if i == 0 else None
        ax.plot(bnd[:, 0], bnd[:, 1], 'r-', linewidth=2, label=label, zorder=4)
        ax.plot([pool.centroid_km[0]], [pool.centroid_km[1]], 'r*', markersize=15, zorder=5)
    
    ax.set_xlabel("x (km)", fontsize=12)
    ax.set_ylabel("y (km)", fontsize=12)
    ax.set_title(f"Full-field θ_ρ (K) at t={time_index}, z={z_index}\n"
                 f"{len(centroids_xy_km)} rain regions, {len(pools)} cold pools detected", 
                 fontsize=14, fontweight='bold')
    ax.set_aspect("equal")
    plt.colorbar(im, ax=ax, label="θ_ρ (K)")
    if len(centroids_xy_km) > 0 or len(pools) > 0:
        ax.legend(loc='upper right')
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    outfile = os.path.join(output_dir, f"fullfield_t{time_index:04d}.png")
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved full-field overview: {outfile}")
    return outfile


def _plot_static_detection(
    data_root: str,
    time_index: int,
    z_index: int,
    pools: List[Pool],
    output_dir: str,
    colormap: Optional[str] = None,
    arrow_subsample: int = 8,
    arrow_scale: float = 100,
):
    """Create a diagnostic plot showing theta_v and RH with detected cold pool boundaries."""
    km = 1000.0
    
    # Get theta_rho field
    theta_rho, xt_m, yt_m = _theta_rho_field(data_root, time_index, z_index)
    xt_km = xt_m / km
    yt_km = yt_m / km
    
    # Get winds
    u_full, v_full, u_mean, v_mean = _winds_at_level_by_index(data_root, time_index, z_index, xt_m, yt_m)
    
    # Get RH field
    with _open(os.path.join(data_root, "rico.p.nc")) as dsp:
        p2d = dsp["p"].isel(time=time_index, zt=z_index).load().values.astype(float)
        
    with _open(os.path.join(data_root, "rico.t.nc")) as dst:
        thl2d = dst["t"].isel(time=time_index, zt=z_index).load().values.astype(float)
        
    with _open(os.path.join(data_root, "rico.q.nc")) as dsq:
        qt2d = dsq["q"].isel(time=time_index, zt=z_index).load().values.astype(float)
        
    with _open(os.path.join(data_root, "rico.l.nc")) as dsl:
        ql2d = dsl["l"].isel(time=time_index, zt=z_index).load().values.astype(float)
    
    # Get rain field
    r_path = os.path.join(data_root, "rico.r.nc")
    if os.path.exists(r_path):
        with _open(r_path) as dsr:
            qr2d = dsr["r"].isel(time=time_index, zt=z_index).load().values.astype(float)
    else:
        qr2d = np.zeros_like(qt2d)
    
    # Unit conversions
    qt2d = _ensure_kgkg(qt2d)
    ql2d = _ensure_kgkg(ql2d)
    qr2d = _ensure_kgkg(qr2d)
    
    # Reconstruct T and compute RH
    from .physics import r_d, c_pd, l_v, p0, r_v, c_pv
    
    theta_l = thl2d
    p = p2d
    qv2d = np.clip(qt2d - ql2d - qr2d, 0.0, None)
    kappa = (r_d / c_pd) * ((1.0 + qv2d / (r_d / r_v)) / (1.0 + qv2d * (c_pv / c_pd)))
    T = theta_l * (c_pd / (c_pd - l_v * ql2d)) * (p0 / p) ** (-kappa)
    
    rh2d = relative_humidity_from_p_T_qv(p, T, qv2d)
    
    # Set up colormap
    if colormap is None:
        colormap = "cmo.thermal" if "cmo.thermal" in plt.colormaps() else "viridis"
    
    # Create figure with two panels: theta_v and RH
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    
    X, Y = np.meshgrid(xt_km, yt_km)
    
    # Subsample arrows
    step = arrow_subsample
    Xs = X[::step, ::step]
    Ys = Y[::step, ::step]
    Us = u_full[::step, ::step]
    Vs = v_full[::step, ::step]
    
    # Panel 1: theta_v
    ax = axes[0]
    vmin = np.nanpercentile(theta_rho, 2)
    vmax = np.nanpercentile(theta_rho, 98)
    im = ax.pcolormesh(X, Y, theta_rho, cmap=colormap, vmin=vmin, vmax=vmax, shading="auto")
    ax.quiver(Xs, Ys, Us, Vs, scale=arrow_scale, color="white", alpha=0.6, width=0.003)
    
    # Plot cold pool boundaries
    for pool in pools:
        bnd = pool.boundary_xy_km
        ax.plot(bnd[:, 0], bnd[:, 1], 'r-', linewidth=2, label='Cold pool boundary')
        ax.plot([pool.centroid_km[0]], [pool.centroid_km[1]], 'r*', markersize=10)
    
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_title(f"θ_ρ (K) - t_index={time_index} - {len(pools)} pools detected")
    ax.set_aspect("equal")
    plt.colorbar(im, ax=ax, label="θ_ρ (K)")
    
    # Panel 2: RH
    ax = axes[1]
    rh_percent = rh2d * 100.0
    vmin_rh = np.nanpercentile(rh_percent, 2)
    vmax_rh = np.nanpercentile(rh_percent, 98)
    im_rh = ax.pcolormesh(X, Y, rh_percent, cmap="cmo.haline" if "cmo.haline" in plt.colormaps() else "viridis",
                          vmin=vmin_rh, vmax=vmax_rh, shading="auto")
    ax.quiver(Xs, Ys, Us, Vs, scale=arrow_scale, color="white", alpha=0.6, width=0.003)
    
    # Plot cold pool boundaries
    for pool in pools:
        bnd = pool.boundary_xy_km
        ax.plot(bnd[:, 0], bnd[:, 1], 'r-', linewidth=2)
        ax.plot([pool.centroid_km[0]], [pool.centroid_km[1]], 'r*', markersize=10)
    
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_title(f"Relative Humidity (%) - t_index={time_index}")
    ax.set_aspect("equal")
    plt.colorbar(im_rh, ax=ax, label="RH (%)")
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    outfile = os.path.join(output_dir, f"static_detection_t{time_index:04d}.png")
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outfile}")
    return outfile


def run_static_detection(
    data_root: str,
    start_index: int = 0,
    n_steps: int = 1,
    z_index: int = 0,
    # Rain seeding parameters
    near_surface_levels: int = 1,
    qr_thresh_kgkg: Optional[float] = None,
    qr_max_height_m: float = 400.0,
    sigma_rain_smooth_m: float = 150.0,
    min_pool_area_km2: float = 0.2,
    # Boundary detection parameters
    hessian_sigma_m: float = 150.0,
    auto_tune_sigma: bool = True,
    # Acceptance gates
    proximity_factor: float = 1.5,
    cover_rainy_min: float = 0.60,
    cover_poly_min: float = 0.10,
    aspect_min: float = 0.40,
    solidity_min: float = 0.55,
    # Output
    output_dir: str = "plots",
    make_plots: bool = True,
    plot_diagnostics: bool = False,
    colormap: Optional[str] = None,
    arrow_subsample: int = 8,
    arrow_scale: float = 100,
):
    """Run static detection for multiple timesteps.
    
    Args:
        data_root: Path to NetCDF data directory
        start_index: Starting time index
        n_steps: Number of timesteps to process
        z_index: Vertical level index
        near_surface_levels: Number of near-surface levels for rain detection
        qr_thresh_kgkg: Rain threshold in kg/kg (None = adaptive)
        qr_max_height_m: Maximum height for rain detection
        sigma_rain_smooth_m: Gaussian smoothing sigma for rain field
        min_pool_area_km2: Minimum cold pool area
        hessian_sigma_m: Gaussian sigma for Hessian computation
        proximity_factor: Centroid proximity gate
        cover_rainy_min: Minimum rain coverage
        cover_poly_min: Minimum polygon coverage
        aspect_min: Minimum aspect ratio
        solidity_min: Minimum solidity
        output_dir: Directory for output plots
        make_plots: Whether to generate diagnostic plots
        colormap: Colormap name for plots
        arrow_subsample: Arrow density for plots
        arrow_scale: Arrow scale for plots
    """
    # Get time values
    t_values, ntime = _time_values_and_count(data_root)
    
    # Generate time indices to process
    time_indices = range(start_index, min(start_index + n_steps, ntime))
    
    print(f"Running static detection for {len(list(time_indices))} timesteps starting at index {start_index}")
    
    all_results = {}
    all_rejection_stats = {
        'total_centroids': 0,
        'total_candidates': 0,
        'rejected_proximity': 0,
        'rejected_overlap': 0,
        'rejected_shape': 0,
        'accepted': 0,
    }
    
    for ti in time_indices:
        print(f"\n{'='*60}")
        print(f"Processing time index {ti}")
        print(f"{'='*60}")
        
        # Store centroids for full-field plot
        centroids_for_plot: List[Tuple[float, float]] = []
        
        # Detect pools and get centroids
        pools, centroids_for_plot = detect_cold_pools_static(
            data_root=data_root,
            time_index=ti,
            z_index=z_index,
            near_surface_levels=near_surface_levels,
            qr_thresh_kgkg=qr_thresh_kgkg,
            qr_max_height_m=qr_max_height_m,
            sigma_rain_smooth_m=sigma_rain_smooth_m,
            min_pool_area_km2=min_pool_area_km2,
            hessian_sigma_m=hessian_sigma_m,
            auto_tune_sigma=auto_tune_sigma,
            proximity_factor=proximity_factor,
            cover_rainy_min=cover_rainy_min,
            cover_poly_min=cover_poly_min,
            aspect_min=aspect_min,
            solidity_min=solidity_min,
            plot_diagnostics=plot_diagnostics,
        )
        
        all_results[ti] = pools
        
        # Always generate full-field overview plot
        if make_plots:
            _plot_full_field_overview(
                data_root=data_root,
                time_index=ti,
                z_index=z_index,
                centroids_xy_km=centroids_for_plot,
                pools=pools,
                output_dir=output_dir,
                colormap=colormap,
            )
        
        # Generate detailed plot if cold pools detected
        if make_plots and len(pools) > 0:
            _plot_static_detection(
                data_root=data_root,
                time_index=ti,
                z_index=z_index,
                pools=pools,
                output_dir=output_dir,
                colormap=colormap,
                arrow_subsample=arrow_subsample,
                arrow_scale=arrow_scale,
            )
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    total_pools = sum(len(pools) for pools in all_results.values())
    print(f"Total cold pools detected: {total_pools}")
    for ti, pools in all_results.items():
        if len(pools) > 0:
            print(f"  t_index={ti}: {len(pools)} pool(s)")
            for i, pool in enumerate(pools):
                print(f"    Pool {i+1}: area={pool.area_km2:.2f} km², "
                      f"centroid=({pool.centroid_km[0]:.2f}, {pool.centroid_km[1]:.2f}) km, "
                      f"mean_θ_ρ={pool.mean_theta_rho:.2f} K")
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    p = argparse.ArgumentParser(description="Static cold pool detection from RICO LES outputs")
    p.add_argument("--data_root", type=str, default=None, help="Path to data directory (or set RICO_DATA env var)")
    p.add_argument("--start_index", type=int, default=0, help="Starting time index")
    p.add_argument("--n_steps", type=int, default=1, help="Number of timesteps to process")
    p.add_argument("--z_index", type=int, default=0, help="Vertical level index")
    p.add_argument("--output_dir", type=str, default="plots", help="Output directory for plots")
    p.add_argument("--no_plots", action="store_true", help="Disable plot generation")
    p.add_argument("--diagnostics", action="store_true", help="Generate diagnostic plots (saved to plots/diagnostics/)")
    
    args = p.parse_args()
    
    data_root = args.data_root or os.environ.get("RICO_DATA")
    if not data_root or not os.path.isdir(data_root):
        raise SystemExit("Provide --data_root or set RICO_DATA environment variable")
    
    run_static_detection(
        data_root=data_root,
        start_index=args.start_index,
        n_steps=args.n_steps,
        z_index=args.z_index,
        output_dir=args.output_dir,
        make_plots=not args.no_plots,
        plot_diagnostics=args.diagnostics,
    )
