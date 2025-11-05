"""Detailed diagnostic plots for cold pool detection pipeline."""

from __future__ import annotations
import os
from typing import List, Optional, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from scipy import ndimage

from .cold_pool_detect import _theta_rho_field, _d2r_field, _zero_contour_paths_km, _region_properties


def plot_detection_pipeline(
    data_root: str,
    time_index: int,
    z_index: int,
    centroid_km: Tuple[float, float],
    centroid_index: int,
    rain_mask: np.ndarray,
    labeled: np.ndarray,
    xt_km: np.ndarray,
    yt_km: np.ndarray,
    xt_m: np.ndarray,
    yt_m: np.ndarray,
    hessian_sigma_m: float,
    acceptance_params: dict,
    output_file: str,
) -> Dict:
    """Plot full detection pipeline for one centroid.
    
    Shows 2x3 panel layout:
    Row 1: theta_rho, theta_rho anomaly, rain mask
    Row 2: d2r field, fallback threshold, final result with rejection info
    
    Returns:
        Dictionary with statistics about this centroid's detection
    """
    km = 1000.0
    cx_km, cy_km = centroid_km
    window_km = 10.0  # Plot window size
    
    # Get theta_rho field
    theta_rho, _, _ = _theta_rho_field(data_root, time_index, z_index)
    
    # Compute grid resolution
    dx = float(np.mean(np.diff(xt_m)))
    dy = float(np.mean(np.diff(yt_m)))
    
    # Crop to window around centroid
    x_mask = (xt_km >= cx_km - window_km) & (xt_km <= cx_km + window_km)
    y_mask = (yt_km >= cy_km - window_km) & (yt_km <= cy_km + window_km)
    
    xt_crop = xt_km[x_mask]
    yt_crop = yt_km[y_mask]
    theta_rho_crop = theta_rho[np.ix_(y_mask, x_mask)]
    rain_mask_crop = rain_mask[np.ix_(y_mask, x_mask)]
    
    # Compute fields
    theta_anom = theta_rho - np.nanmean(theta_rho)
    theta_anom_crop = theta_anom[np.ix_(y_mask, x_mask)]
    
    # d2r field
    sigma_x_pix = hessian_sigma_m / dx
    sigma_y_pix = hessian_sigma_m / dy
    d2r = _d2r_field(theta_rho, (cx_km*km, cy_km*km), xt_m, yt_m, (sigma_y_pix, sigma_x_pix))
    d2r_crop = d2r[np.ix_(y_mask, x_mask)]
    
    # Try to find contours with d2r
    d2r_contours = _zero_contour_paths_km(d2r, xt_km, yt_km)
    used_fallback = len(d2r_contours) == 0
    
    # Fallback threshold method
    Xk, Yk = np.meshgrid(xt_km, yt_km)
    R = np.sqrt((Xk - cx_km)**2 + (Yk - cy_km)**2)
    search_radius_km = 5.0
    local_region = (R <= search_radius_km)
    
    fallback_contours = []
    threshold_val = None
    if used_fallback and np.any(local_region):
        local_vals = theta_anom[local_region]
        threshold_val = float(np.percentile(local_vals[np.isfinite(local_vals)], 10))
        all_fallback_contours = _zero_contour_paths_km(theta_anom - threshold_val, xt_km, yt_km)
        
        # Filter: only keep contours near centroid (same as main detection)
        for seg_km in all_fallback_contours:
            path = MplPath(seg_km)
            if path.contains_point((cx_km, cy_km)):
                fallback_contours.append(seg_km)
                continue
            seg_arr = np.array(seg_km)
            min_dist = np.min(np.sqrt((seg_arr[:,0] - cx_km)**2 + (seg_arr[:,1] - cy_km)**2))
            if min_dist <= search_radius_km:
                fallback_contours.append(seg_km)
    
    # Use whichever method produced contours
    contour_paths = d2r_contours if not used_fallback else fallback_contours
    
    # Evaluate candidates against acceptance criteria
    results = _evaluate_candidates(
        contour_paths, centroid_km, rain_mask, labeled, 
        xt_km, yt_km, dx, dy, acceptance_params
    )
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    X, Y = np.meshgrid(xt_crop, yt_crop)
    
    # === ROW 1: INPUT FIELDS ===
    
    # Panel A: theta_rho
    ax = axes[0, 0]
    vmin = np.nanpercentile(theta_rho_crop, 5)
    vmax = np.nanpercentile(theta_rho_crop, 95)
    im = ax.pcolormesh(X, Y, theta_rho_crop, cmap='RdBu_r', vmin=vmin, vmax=vmax, shading='auto')
    ax.plot(cx_km, cy_km, 'k*', markersize=15, label='Rain centroid')
    ax.set_xlabel('x (km)')
    ax.set_ylabel('y (km)')
    ax.set_title(f'(A) θ_ρ field (K)', fontweight='bold')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, label='θ_ρ (K)')
    ax.legend()
    
    # Panel B: theta_rho anomaly
    ax = axes[0, 1]
    vext = np.nanmax(np.abs(theta_anom_crop))
    im = ax.pcolormesh(X, Y, theta_anom_crop, cmap='RdBu_r', vmin=-vext, vmax=vext, shading='auto')
    ax.plot(cx_km, cy_km, 'k*', markersize=15)
    # Draw search radius
    circle = plt.Circle((cx_km, cy_km), search_radius_km, 
                       fill=False, color='green', linewidth=2, linestyle='--',
                       label=f'Search radius: {search_radius_km} km')
    ax.add_patch(circle)
    ax.set_xlabel('x (km)')
    ax.set_ylabel('y (km)')
    ax.set_title(f'(B) θ_ρ anomaly (K)', fontweight='bold')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, label='Anomaly (K)')
    ax.legend()
    
    # Panel C: Rain mask
    ax = axes[0, 2]
    ax.pcolormesh(X, Y, rain_mask_crop.astype(float), cmap='Blues', shading='auto', alpha=0.6)
    ax.plot(cx_km, cy_km, 'r*', markersize=15, label='Rain centroid')
    ax.set_xlabel('x (km)')
    ax.set_ylabel('y (km)')
    ax.set_title(f'(C) Rain mask region', fontweight='bold')
    ax.set_aspect('equal')
    ax.legend()
    
    # === ROW 2: DETECTION STEPS ===
    
    # Panel D: d2r field with contours
    ax = axes[1, 0]
    vext = np.nanmax(np.abs(d2r_crop))
    im = ax.pcolormesh(X, Y, d2r_crop, cmap='RdBu_r', vmin=-vext, vmax=vext, shading='auto')
    ax.plot(cx_km, cy_km, 'k*', markersize=15)
    
    # Plot d2r zero contours if they exist
    for path in d2r_contours:
        path_x, path_y = path[:, 0], path[:, 1]
        if np.any((path_x >= cx_km - window_km) & (path_x <= cx_km + window_km)):
            ax.plot(path_x, path_y, 'lime', linewidth=2)
    
    ax.set_xlabel('x (km)')
    ax.set_ylabel('y (km)')
    title_d = f'(D) d²θ_ρ/dr² field\n{len(d2r_contours)} zero-contours'
    ax.set_title(title_d, fontweight='bold')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, label='d²r')
    
    # Panel E: Fallback threshold method
    ax = axes[1, 1]
    if used_fallback:
        threshold_field = theta_anom_crop - threshold_val if threshold_val is not None else theta_anom_crop
        vext = np.nanmax(np.abs(threshold_field))
        im = ax.pcolormesh(X, Y, threshold_field, cmap='RdBu_r', vmin=-vext, vmax=vext, shading='auto')
        ax.plot(cx_km, cy_km, 'k*', markersize=15)
        
        # Plot fallback contours
        for path in fallback_contours:
            path_x, path_y = path[:, 0], path[:, 1]
            if np.any((path_x >= cx_km - window_km) & (path_x <= cx_km + window_km)):
                ax.plot(path_x, path_y, 'lime', linewidth=2)
        
        title_e = f'(E) Fallback: θ_ρ threshold\nthresh={threshold_val:.3f} K, {len(fallback_contours)} contours'
        ax.set_title(title_e, fontweight='bold', color='orange')
    else:
        ax.text(0.5, 0.5, 'Fallback not used\n(d²r method succeeded)', 
               ha='center', va='center', transform=ax.transAxes, fontsize=14)
        title_e = '(E) Fallback (not used)'
        ax.set_title(title_e, fontweight='bold')
    
    ax.set_xlabel('x (km)')
    ax.set_ylabel('y (km)')
    ax.set_aspect('equal')
    if used_fallback:
        plt.colorbar(im, ax=ax, label='θ_ρ - threshold')
    
    # Panel F: Final result with acceptance info
    ax = axes[1, 2]
    ax.pcolormesh(X, Y, theta_anom_crop, cmap='RdBu_r', vmin=-vext, vmax=vext, shading='auto', alpha=0.3)
    ax.plot(cx_km, cy_km, 'k*', markersize=15)
    
    # Plot all candidate contours with color-coding
    for i, (path, status) in enumerate(zip(results['candidate_paths'], results['statuses'])):
        path_x, path_y = path[:, 0], path[:, 1]
        if not np.any((path_x >= cx_km - window_km) & (path_x <= cx_km + window_km)):
            continue
        
        if status == 'accepted':
            ax.plot(path_x, path_y, 'g-', linewidth=3, label='Accepted' if i == 0 else None)
        elif 'proximity' in status:
            ax.plot(path_x, path_y, 'orange', linewidth=1, linestyle='--', alpha=0.5, 
                   label='Rejected: Proximity' if 'proximity' not in [results['statuses'][j] for j in range(i)] else None)
        elif 'overlap' in status:
            ax.plot(path_x, path_y, 'purple', linewidth=1, linestyle='--', alpha=0.5,
                   label='Rejected: Overlap' if 'overlap' not in [results['statuses'][j] for j in range(i)] else None)
        elif 'shape' in status:
            ax.plot(path_x, path_y, 'red', linewidth=1, linestyle='--', alpha=0.5,
                   label='Rejected: Shape' if 'shape' not in [results['statuses'][j] for j in range(i)] else None)
    
    ax.set_xlabel('x (km)')
    ax.set_ylabel('y (km)')
    title_f = f'(F) Final result\n{results["n_accepted"]} accepted, {results["n_rejected"]} rejected'
    ax.set_title(title_f, fontweight='bold', color='green' if results["n_accepted"] > 0 else 'red')
    ax.set_aspect('equal')
    if len(results['candidate_paths']) > 0:
        ax.legend(loc='upper right', fontsize=9)
    
    # Add overall summary text
    summary = f"Centroid {centroid_index} at ({cx_km:.1f}, {cy_km:.1f}) km\n"
    summary += f"Method: {'Fallback (θ_ρ threshold)' if used_fallback else 'd²θ_ρ/dr² zero-contour'}\n"
    summary += f"Candidates: {len(contour_paths)} | "
    summary += f"Rejected: prox={results['n_proximity']}, ovlp={results['n_overlap']}, shape={results['n_shape']} | "
    summary += f"Accepted: {results['n_accepted']}"
    
    fig.suptitle(f"Detection Pipeline - t={time_index}, z={z_index}", 
                fontsize=16, fontweight='bold', y=0.995)
    fig.text(0.5, 0.01, summary, ha='center', fontsize=11, 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.99])
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return results


def _evaluate_candidates(
    contour_paths: List[np.ndarray],
    centroid_km: Tuple[float, float],
    rain_mask: np.ndarray,
    labeled: np.ndarray,
    xt_km: np.ndarray,
    yt_km: np.ndarray,
    dx: float,
    dy: float,
    params: dict,
) -> Dict:
    """Evaluate all candidate contours against acceptance criteria."""
    km = 1000.0
    cx_km, cy_km = centroid_km
    
    # Find rain region associated with this centroid
    nlab = int(np.max(labeled)) if labeled.size > 0 else 0
    dists = []
    for lab in range(1, nlab+1):
        region = (labeled == lab)
        iy_reg, ix_reg = np.nonzero(region)
        if ix_reg.size == 0:
            continue
        x_km = float(np.mean(xt_km[ix_reg]))
        y_km = float(np.mean(yt_km[iy_reg]))
        dists.append(((cx_km - x_km)**2 + (cy_km - y_km)**2, lab))
    
    if not dists:
        return {
            'candidate_paths': [],
            'statuses': [],
            'n_accepted': 0,
            'n_rejected': 0,
            'n_proximity': 0,
            'n_overlap': 0,
            'n_shape': 0,
        }
    
    dists.sort()
    rain_lab = dists[0][1]
    rain_mask_local = (labeled == rain_lab)
    rain_area_km2 = rain_mask_local.sum() * dx * dy / (km*km)
    rain_eq_diam_km = 2.0 * np.sqrt(rain_area_km2 / np.pi)
    
    # Evaluate each candidate
    Xk, Yk = np.meshgrid(xt_km, yt_km)
    statuses = []
    n_proximity = 0
    n_overlap = 0
    n_shape = 0
    n_accepted = 0
    
    # Pre-crop grid to region around centroid for faster point-in-polygon tests
    window_km = 10.0
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
            statuses.append('empty')
            continue
        
        # Centroid test
        iy, ix = np.nonzero(poly_mask)
        if ix.size == 0:
            statuses.append('empty')
            continue
        
        cx_poly_km = float(np.mean(xt_km[ix]))
        cy_poly_km = float(np.mean(yt_km[iy]))
        cen_dist = np.hypot(cx_poly_km - cx_km, cy_poly_km - cy_km)
        
        # Proximity test
        if cen_dist > (params['proximity_factor'] * rain_eq_diam_km):
            statuses.append('rejected_proximity')
            n_proximity += 1
            continue
        
        # Overlap test
        inter = (poly_mask & rain_mask_local).sum()
        rain_px = rain_mask_local.sum()
        poly_px = poly_mask.sum()
        
        if rain_px == 0 or poly_px == 0:
            statuses.append('empty')
            continue
        
        cover_rainy = inter / rain_px
        cover_poly = inter / poly_px
        
        if not (cover_rainy >= params['cover_rainy_min'] and cover_poly >= params['cover_poly_min']):
            statuses.append('rejected_overlap')
            n_overlap += 1
            continue
        
        # Shape test
        props = _region_properties(poly_mask, dx, dy)
        if not (props["aspect"] >= params['aspect_min'] and props["solidity"] >= params['solidity_min']):
            statuses.append('rejected_shape')
            n_shape += 1
            continue
        
        statuses.append('accepted')
        n_accepted += 1
    
    return {
        'candidate_paths': contour_paths,
        'statuses': statuses,
        'n_accepted': n_accepted,
        'n_rejected': n_proximity + n_overlap + n_shape,
        'n_proximity': n_proximity,
        'n_overlap': n_overlap,
        'n_shape': n_shape,
    }
