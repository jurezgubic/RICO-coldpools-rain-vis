"""Diagnostic tools for cold pool detection parameter tuning.

This module helps visualize fields and auto-tune detection parameters.
"""

from __future__ import annotations
import os
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from .cold_pool_detect import _theta_rho_field, _d2r_field, _zero_contour_paths_km


def estimate_cold_pool_length_scale(
    theta_rho: np.ndarray,
    xt_km: np.ndarray,
    yt_km: np.ndarray,
    centroid_km: Tuple[float, float],
    search_radius_km: float = 5.0,
) -> dict:
    """Estimate characteristic length scale of theta_rho anomaly around a centroid.
    
    Args:
        theta_rho: 2D field of density potential temperature
        xt_km: x coordinates in km
        yt_km: y coordinates in km
        centroid_km: (x, y) centroid position in km
        search_radius_km: Radius to analyze around centroid
    
    Returns:
        Dictionary with length scale estimates
    """
    cx_km, cy_km = centroid_km
    
    # Create mesh
    X, Y = np.meshgrid(xt_km, yt_km)
    
    # Distance from centroid
    R = np.sqrt((X - cx_km)**2 + (Y - cy_km)**2)
    
    # Extract region around centroid
    mask = R <= search_radius_km
    if not np.any(mask):
        return {"length_scale_km": None, "anomaly_strength_K": 0.0}
    
    # Compute anomaly relative to local mean
    local_mean = np.nanmean(theta_rho[mask])
    anomaly = theta_rho - local_mean
    
    # Find where anomaly is negative (cold pool interior)
    cold_mask = (anomaly < 0) & mask
    
    if not np.any(cold_mask):
        return {"length_scale_km": None, "anomaly_strength_K": 0.0}
    
    # Characteristic radius: radius containing 86% of cold area (1-sigma for Gaussian)
    cold_r = R[cold_mask]
    r_sorted = np.sort(cold_r)
    idx_86 = int(0.86 * len(r_sorted))
    length_scale_km = float(r_sorted[idx_86]) if idx_86 < len(r_sorted) else float(r_sorted[-1])
    
    # Anomaly strength
    anomaly_strength = float(np.abs(np.nanmin(anomaly[cold_mask])))
    
    return {
        "length_scale_km": length_scale_km,
        "anomaly_strength_K": anomaly_strength,
        "cold_area_km2": float(cold_mask.sum() * (xt_km[1] - xt_km[0]) * (yt_km[1] - yt_km[0])),
    }


def plot_theta_rho_diagnostic(
    data_root: str,
    time_index: int,
    z_index: int,
    centroid_km: Tuple[float, float],
    hessian_sigma_m: float,
    window_km: float = 10.0,
    output_file: str = None,
):
    """Plot theta_rho field and d2r field around a centroid for diagnostics.
    
    Args:
        data_root: Path to data directory
        time_index: Time index
        z_index: Vertical level
        centroid_km: (x, y) position in km
        hessian_sigma_m: Smoothing scale for Hessian
        window_km: Half-width of plot window
        output_file: Where to save plot (None = show)
    """
    km = 1000.0
    cx_km, cy_km = centroid_km
    
    # Get theta_rho field
    theta_rho, xt_m, yt_m = _theta_rho_field(data_root, time_index, z_index)
    xt_km = xt_m / km
    yt_km = yt_m / km
    
    # Compute grid resolution
    dx = float(np.mean(np.diff(xt_m)))
    dy = float(np.mean(np.diff(yt_m)))
    
    # Crop to window around centroid
    x_mask = (xt_km >= cx_km - window_km) & (xt_km <= cx_km + window_km)
    y_mask = (yt_km >= cy_km - window_km) & (yt_km <= cy_km + window_km)
    
    xt_crop = xt_km[x_mask]
    yt_crop = yt_km[y_mask]
    theta_rho_crop = theta_rho[np.ix_(y_mask, x_mask)]
    
    # Compute d2r field
    sigma_x_pix = hessian_sigma_m / dx
    sigma_y_pix = hessian_sigma_m / dy
    d2r = _d2r_field(theta_rho, (cx_km*km, cy_km*km), xt_m, yt_m, (sigma_y_pix, sigma_x_pix))
    d2r_crop = d2r[np.ix_(y_mask, x_mask)]
    
    # Get zero contours
    contour_paths = _zero_contour_paths_km(d2r, xt_km, yt_km)
    
    # Estimate length scale
    length_info = estimate_cold_pool_length_scale(theta_rho, xt_km, yt_km, centroid_km)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    X, Y = np.meshgrid(xt_crop, yt_crop)
    
    # Panel 1: theta_rho
    ax = axes[0]
    vmin = np.nanpercentile(theta_rho_crop, 5)
    vmax = np.nanpercentile(theta_rho_crop, 95)
    im = ax.pcolormesh(X, Y, theta_rho_crop, cmap='RdBu_r', vmin=vmin, vmax=vmax, shading='auto')
    ax.plot(cx_km, cy_km, 'k*', markersize=15, label='Rain centroid')
    ax.set_xlabel('x (km)')
    ax.set_ylabel('y (km)')
    ax.set_title(f'theta_rho (K) at t={time_index}, z={z_index}')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, label='theta_rho (K)')
    ax.legend()
    
    # Panel 2: theta_rho anomaly
    ax = axes[1]
    anomaly = theta_rho_crop - np.nanmean(theta_rho_crop)
    vext = np.nanmax(np.abs(anomaly))
    im = ax.pcolormesh(X, Y, anomaly, cmap='RdBu_r', vmin=-vext, vmax=vext, shading='auto')
    ax.plot(cx_km, cy_km, 'k*', markersize=15)
    if length_info['length_scale_km'] is not None:
        circle = plt.Circle((cx_km, cy_km), length_info['length_scale_km'], 
                           fill=False, color='green', linewidth=2, 
                           label=f'Length scale: {length_info["length_scale_km"]:.1f} km')
        ax.add_patch(circle)
    ax.set_xlabel('x (km)')
    ax.set_ylabel('y (km)')
    ax.set_title('theta_rho anomaly (K)')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, label='Anomaly (K)')
    ax.legend()
    
    # Panel 3: d2r field with contours
    ax = axes[2]
    vext = np.nanmax(np.abs(d2r_crop))
    im = ax.pcolormesh(X, Y, d2r_crop, cmap='RdBu_r', vmin=-vext, vmax=vext, shading='auto')
    ax.plot(cx_km, cy_km, 'k*', markersize=15)
    
    # Plot zero contours if they exist
    for path in contour_paths:
        # Only plot if within window
        path_x, path_y = path[:, 0], path[:, 1]
        if np.any((path_x >= cx_km - window_km) & (path_x <= cx_km + window_km) &
                  (path_y >= cy_km - window_km) & (path_y <= cy_km + window_km)):
            ax.plot(path_x, path_y, 'lime', linewidth=2)
    
    ax.set_xlabel('x (km)')
    ax.set_ylabel('y (km)')
    ax.set_title(f'd2_theta_rho/dr2 (sigma={hessian_sigma_m}m)\n{len(contour_paths)} zero-contours')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, label='d2r')
    
    # Add text with diagnostics
    info_text = f"Length scale: {length_info['length_scale_km']:.1f} km\n"
    info_text += f"Anomaly: {length_info['anomaly_strength_K']:.2f} K\n"
    info_text += f"Suggested sigma: {length_info['length_scale_km']*1000*0.5:.0f} m"
    fig.text(0.02, 0.02, info_text, fontsize=10, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved diagnostic plot: {output_file}")
        plt.close()
    else:
        plt.show()
    
    return length_info


def auto_tune_hessian_sigma(
    data_root: str,
    time_index: int,
    z_index: int,
    rain_centroids_km: list,
    max_centroids: int = 5,
) -> float:
    """Automatically determine appropriate hessian_sigma_m from cold pool features.
    
    Args:
        data_root: Path to data
        time_index: Time index
        z_index: Vertical level
        rain_centroids_km: List of (x, y) rain centroids in km
        max_centroids: Maximum centroids to analyze
    
    Returns:
        Suggested hessian_sigma_m in meters
    """
    km = 1000.0
    
    # Get theta_rho field
    theta_rho, xt_m, yt_m = _theta_rho_field(data_root, time_index, z_index)
    xt_km = xt_m / km
    yt_km = yt_m / km
    
    # Analyze a subset of centroids
    length_scales = []
    for i, centroid_km in enumerate(rain_centroids_km[:max_centroids]):
        info = estimate_cold_pool_length_scale(theta_rho, xt_km, yt_km, centroid_km)
        if info['length_scale_km'] is not None and info['anomaly_strength_K'] > 0.1:
            length_scales.append(info['length_scale_km'])
    
    if len(length_scales) == 0:
        print("WARNING: No clear cold pool features detected. Using default sigma=150m")
        return 150.0
    
    # Use median length scale, convert to sigma (assume ~0.5 * radius is good smoothing)
    median_length_km = float(np.median(length_scales))
    suggested_sigma_m = median_length_km * 1000.0 * 0.5
    
    print(f"Auto-tuned hessian_sigma_m: {suggested_sigma_m:.0f} m "
          f"(from {len(length_scales)} features, median scale={median_length_km:.1f} km)")
    
    return suggested_sigma_m


if __name__ == "__main__":
    import argparse
    
    p = argparse.ArgumentParser(description="Diagnostic tools for cold pool detection")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--time_index", type=int, required=True)
    p.add_argument("--z_index", type=int, default=0)
    p.add_argument("--centroid", type=float, nargs=2, required=True, 
                   help="x y coordinates of centroid in km")
    p.add_argument("--hessian_sigma_m", type=float, default=150.0)
    p.add_argument("--output", type=str, default="diagnostic.png")
    
    args = p.parse_args()
    
    plot_theta_rho_diagnostic(
        data_root=args.data_root,
        time_index=args.time_index,
        z_index=args.z_index,
        centroid_km=tuple(args.centroid),
        hessian_sigma_m=args.hessian_sigma_m,
        output_file=args.output,
    )
