import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import xarray as xr
from scipy import ndimage
from scipy.ndimage import binary_fill_holes
from scipy.ndimage import gaussian_filter
from scipy.ndimage import grey_closing, grey_opening
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

from .physics import _ensure_kgkg, density_potential_temperature, g


def _open(path: str) -> xr.Dataset:
    return xr.open_dataset(path, decode_times=False)


def _time_values_and_count(data_root: str) -> Tuple[np.ndarray, int]:
    with _open(os.path.join(data_root, "rico.p.nc")) as ds:
        t = ds["time"].values
    return t, int(t.shape[0])


def _winds_at_level_by_index(data_root: str, t_index: int, z_index: int,
                             xt: np.ndarray, yt: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
    # Read u and v, interpolate to centers using xarray built-in interp like in plot module
    with _open(os.path.join(data_root, "rico.u.nc")) as dsu:
        u_da = dsu["u"].isel(time=t_index, zt=z_index)
        if "xm" in u_da.dims:
            u_da = u_da.interp(xm=xt)
        u = u_da.load().values.astype(float)
        fill = u_da.attrs.get("_FillValue", None)
        if fill is not None:
            u = np.where(u == fill, np.nan, u)

    with _open(os.path.join(data_root, "rico.v.nc")) as dsv:
        v_da = dsv["v"].isel(time=t_index, zt=z_index)
        if "ym" in v_da.dims:
            v_da = v_da.interp(ym=yt)
        v = v_da.load().values.astype(float)
        fill = v_da.attrs.get("_FillValue", None)
        if fill is not None:
            v = np.where(v == fill, np.nan, v)

    return u, v, float(np.nanmean(u)), float(np.nanmean(v))


def _theta_rho_field(data_root: str, t_index: int, z_index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with _open(os.path.join(data_root, "rico.p.nc")) as dsp:
        p2d = dsp["p"].isel(time=t_index, zt=z_index).load().values.astype(float)
        xt = dsp["xt"].values
        yt = dsp["yt"].values

    with _open(os.path.join(data_root, "rico.t.nc")) as dst:
        thl2d = dst["t"].isel(time=t_index, zt=z_index).load().values.astype(float)

    with _open(os.path.join(data_root, "rico.q.nc")) as dsq:
        qt2d = dsq["q"].isel(time=t_index, zt=z_index).load().values.astype(float)

    with _open(os.path.join(data_root, "rico.l.nc")) as dsl:
        ql2d = dsl["l"].isel(time=t_index, zt=z_index).load().values.astype(float)

    # optional rain mixing ratio
    qr2d = None
    r_path = os.path.join(data_root, "rico.r.nc")
    if os.path.exists(r_path):
        try:
            with _open(r_path) as dsr:
                qr2d = dsr["r"].isel(time=t_index, zt=z_index).load().values.astype(float)
        except Exception:
            qr2d = None

    # Optional rain mixing ratio not used for theta_rho; assume no ice
    qt2d = _ensure_kgkg(qt2d)
    ql2d = _ensure_kgkg(ql2d)
    qr2d = _ensure_kgkg(qr2d) if qr2d is not None else 0.0
    # derive qv = qt - ql - qr
    qv2d = np.clip(qt2d - ql2d - (qr2d if isinstance(qr2d, np.ndarray) else qr2d), 0.0, None)

    # We need temperature T; use existing physics conversion by reconstructing from theta_l
    # Quick consistent T reconstruction as used in physics.calculate_physics_variables
    # Reuse that code indirectly: approximate T via theta_l and p with moist kappa factor
    # For density potential temperature we only need theta = T (p0/p)^(R_d/c_pd),
    # but T below follows the same approach as calculate_physics_variables.
    from .physics import r_d, c_pd, l_v, p0, r_v, c_pv

    theta_l = thl2d
    p = p2d
    q_v = qv2d
    kappa = (r_d / c_pd) * ((1.0 + q_v / (r_d / r_v)) / (1.0 + q_v * (c_pv / c_pd)))
    T = theta_l * (c_pd / (c_pd - l_v * ql2d)) * (p0 / p) ** (-kappa)

    theta_rho = density_potential_temperature(p, T, q_v, ql2d, 0.0, qr2d)
    return theta_rho, xt, yt


def _rain_mask_from_qr(
    data_root: str,
    t_index: int,
    near_surface_levels: int,
    qr_thresh_kgkg: Optional[float],
    qr_max_height_m: Optional[float],
    sigma_rain_smooth_m: float,
    min_area_km2: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Build rainy mask from q_r with smoothing, adaptive threshold, and diagnostics.

    Returns (rainy_mask_post, xt, yt, diag)
    diag includes: qr_field (smoothed), p95,p98,p99, qr_thresh, pre/post counts and areas, centroids_xy_km
    """
    # base grid and resolution
    with _open(os.path.join(data_root, "rico.p.nc")) as dsp:
        xt = np.array(dsp["xt"].values)
        yt = np.array(dsp["yt"].values)
    dx = float(np.mean(np.diff(xt)))
    dy = float(np.mean(np.diff(yt)))
    km = 1000.0
    path = os.path.join(data_root, "rico.r.nc")
    if not os.path.exists(path):
        print("no rain mixing ratio file; treating q_r as zero")
        diag = {"qr_field": np.zeros((len(yt), len(xt)), dtype=float),
                "p95": 0.0, "p98": 0.0, "p99": 0.0, "qr_thresh": float(qr_thresh_kgkg or 0.0),
                "pre_components": 0, "pre_pixels": 0, "pre_area_km2": 0.0,
                "post_components": 0, "post_pixels": 0, "post_area_km2": 0.0,
                "centroids_xy_km": []}
        return np.zeros((len(yt), len(xt)), dtype=bool), xt, yt, diag
    with _open(path) as ds:
        r_var = ds["r"]
        r_da = r_var.isel(time=t_index)
        # Choose vertical selection by height if provided, else by a fixed number of lowest levels
        if qr_max_height_m is not None and "zt" in r_da.dims and "zt" in ds:
            z = np.asarray(ds["zt"].values, dtype=float)
            idx = np.where(z <= float(qr_max_height_m))[0]
            if idx.size == 0:
                idx = np.array([0])
            r_da = r_da.isel(zt=idx)
        else:
            nz = int(ds.sizes.get("zt", r_da.sizes.get("zt", r_da.shape[0])))
            nlev = max(1, min(int(near_surface_levels), nz))
            r_da = r_da.isel(zt=slice(0, nlev))
        r_sub = r_da.load().values.astype(float)
        # mask fill/missing and clamp negatives
        fill = r_var.attrs.get("_FillValue", None)
        miss = r_var.attrs.get("missing_value", None)
        if fill is not None:
            r_sub = np.where(r_sub == float(fill), np.nan, r_sub)
        if miss is not None:
            r_sub = np.where(r_sub == float(miss), np.nan, r_sub)
        r_sub = np.where(r_sub < 0, 0.0, r_sub)
    qr = _ensure_kgkg(r_sub)
    # max over selected levels
    qr_max = np.nanmax(qr, axis=0)
    # smooth with physical sigma
    sigx = float(sigma_rain_smooth_m / dx) if dx > 0 else 1.0
    sigy = float(sigma_rain_smooth_m / dy) if dy > 0 else 1.0
    qr_smooth = gaussian_filter(qr_max, sigma=(sigy, sigx))
    # small morphology close->open on greyscale field
    footprint = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=int)
    qr_smooth = grey_closing(qr_smooth, footprint=footprint)
    qr_smooth = grey_opening(qr_smooth, footprint=footprint)
    # diagnostics percentiles
    finite_vals = qr_smooth[np.isfinite(qr_smooth)]
    if finite_vals.size == 0:
        p95 = p98 = p99 = 0.0
    else:
        p95 = float(np.nanpercentile(finite_vals, 95))
        p98 = float(np.nanpercentile(finite_vals, 98))
        p99 = float(np.nanpercentile(finite_vals, 99))
    # adaptive threshold unless explicit threshold provided
    if qr_thresh_kgkg is None:
        qr_thresh = max(min(p98, 5e-5), 2e-6)
    else:
        qr_thresh = float(qr_thresh_kgkg)
    rainy0 = qr_smooth >= qr_thresh
    # pre-filter counts
    structure = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=int)
    labeled_pre, nlab_pre = ndimage.label(rainy0, structure=structure)
    pre_pixels = int(rainy0.sum())
    pre_area_km2 = pre_pixels * dx * dy / (km*km)
    # fill holes per region and drop small areas
    mask_keep = np.zeros_like(rainy0, dtype=bool)
    for lab in range(1, nlab_pre+1):
        region = (labeled_pre == lab)
        region = binary_fill_holes(region)
        area_km2 = region.sum() * dx * dy / (km*km)
        if area_km2 >= float(min_area_km2):
            mask_keep |= region
    rainy = mask_keep
    labeled_post, nlab_post = ndimage.label(rainy, structure=structure)
    post_pixels = int(rainy.sum())
    post_area_km2 = post_pixels * dx * dy / (km*km)
    # centroids after filtering
    cents: List[Tuple[float, float]] = []
    for lab in range(1, nlab_post+1):
        region = (labeled_post == lab)
        if not np.any(region):
            continue
        iy, ix = np.nonzero(region)
        if ix.size == 0:
            continue
        x0_km = float(np.mean(xt[ix]))/km
        y0_km = float(np.mean(yt[iy]))/km
        cents.append((x0_km, y0_km))

    diag = {
        "qr_field": qr_smooth,
        "p95": p95, "p98": p98, "p99": p99,
        "qr_thresh": float(qr_thresh),
        "pre_components": int(nlab_pre), "pre_pixels": pre_pixels, "pre_area_km2": float(pre_area_km2),
        "post_components": int(nlab_post), "post_pixels": post_pixels, "post_area_km2": float(post_area_km2),
        "centroids_xy_km": cents,
        "labeled_post": labeled_post,
    }
    return rainy, xt, yt, diag


def _mask_from_polygon_contour(mask: np.ndarray) -> List[np.ndarray]:
    """Extract polygon boundaries from a binary mask using matplotlib contours."""
    cs = plt.contour(mask.astype(float), levels=[0.5])
    paths = []
    for c in cs.allsegs[0]:
        paths.append(c)
    plt.clf()
    return paths


def _region_properties(mask: np.ndarray, dx: float, dy: float) -> Dict[str, float]:
    area_px = int(mask.sum())
    area = area_px * dx * dy
    # bounding box aspect ratio
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return {"area": 0.0, "eq_radius": 0.0, "aspect": 0.0, "solidity": 0.0}
    w = xs.max() - xs.min() + 1
    h = ys.max() - ys.min() + 1
    aspect = min(w, h) / max(w, h)
    # solidity via convex hull
    pts = np.column_stack([xs, ys])
    if pts.shape[0] >= 3:
        hull = ConvexHull(pts)
        hull_area = hull.area  # perimeter in 2D? For 2D, ConvexHull.area returns perimeter; use volume for area.
        hull_area = hull.volume  # in 2D, volume is area
        solidity = (area_px / hull_area) if hull_area > 0 else 0.0
    else:
        solidity = 0.0
    eq_radius = np.sqrt(area / np.pi)
    return {"area": area, "eq_radius": eq_radius, "aspect": float(aspect), "solidity": float(solidity)}


def _d2r_field(
    theta_rho: np.ndarray,
    centroid_xy_m: Tuple[float, float],
    xt_m: np.ndarray,
    yt_m: np.ndarray,
    sigma_pix_yx: Tuple[float, float],
) -> np.ndarray:
    """Second radial derivative field u^T H u about centroid, smoothed with anisotropic sigma in pixel units.

    sigma_pix_yx is (sigma_y_pixels, sigma_x_pixels) corresponding to array axes (yt, xt).
    """
    sy, sx = sigma_pix_yx
    # Hessian components (orders along (y, x))
    dxx = gaussian_filter(theta_rho, sigma=(sy, sx), order=(0, 2))
    dyy = gaussian_filter(theta_rho, sigma=(sy, sx), order=(2, 0))
    dxy = gaussian_filter(theta_rho, sigma=(sy, sx), order=(1, 1))

    X, Y = np.meshgrid(xt_m, yt_m)
    rx = X - centroid_xy_m[0]
    ry = Y - centroid_xy_m[1]
    rnorm = np.hypot(rx, ry)
    cos_t = np.where(rnorm > 0, rx / rnorm, 0.0)
    sin_t = np.where(rnorm > 0, ry / rnorm, 0.0)
    d2r = (dxx * cos_t * cos_t) + (2.0 * dxy * sin_t * cos_t) + (dyy * sin_t * sin_t)
    return d2r


def _zero_contour_paths_km(d2r: np.ndarray, xt_km: np.ndarray, yt_km: np.ndarray) -> List[np.ndarray]:
    """Return closed zero-level contour paths of d2r in km coordinates (list of Nx2 arrays)."""
    Xk, Yk = np.meshgrid(xt_km, yt_km)
    cs = plt.contour(Xk, Yk, d2r, levels=[0.0])
    paths: List[np.ndarray] = []
    try:
        for seg in cs.allsegs[0]:
            if seg.shape[0] < 20:
                continue
            # Keep only closed paths (first ~ last within small tol)
            if np.hypot(*(seg[0] - seg[-1])) > 1e-6:
                continue
            paths.append(seg)
    finally:
        plt.clf()
    return paths


def _plot_rain_distribution(
    xt_km: np.ndarray,
    yt_km: np.ndarray,
    diag: dict,
    prefix: str,
    t_seed: int,
):
    """Diagnostic plot of smoothed qr_max with adaptive threshold and kept components."""
    qr = diag.get("qr_field")
    thresh = diag.get("qr_thresh", 0.0)
    labeled_post = diag.get("labeled_post")
    p95, p98, p99 = diag.get("p95", 0.0), diag.get("p98", 0.0), diag.get("p99", 0.0)
    Xk, Yk = np.meshgrid(xt_km, yt_km)
    fig, ax = plt.subplots(figsize=(6,6))
    im = ax.pcolormesh(Xk, Yk, qr, shading="auto", cmap="viridis")
    try:
        cs = ax.contour(Xk, Yk, qr, levels=[thresh], colors="white", linewidths=1.0)
    except Exception:
        cs = None
    # overlay labels/centroids
    if labeled_post is not None:
        comps = int(np.max(labeled_post)) if labeled_post.size > 0 else 0
        for lab in range(1, comps+1):
            mask = (labeled_post == lab)
            if not np.any(mask):
                continue
            iy, ix = np.nonzero(mask)
            cx = float(np.mean(xt_km[ix]))
            cy = float(np.mean(yt_km[iy]))
            ax.plot(cx, cy, marker="+", color="white", markersize=8, mew=1.5)
    ax.set_title(f"qr_max smoothed @ t={t_seed}; p95={p95:.2e}, p98={p98:.2e}, p99={p99:.2e}; thr={thresh:.2e}")
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    cbar = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02)
    cbar.set_label("q_r (kg/kg)")
    out = f"{prefix}_rain_{int(t_seed):05d}.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)


def _plot_theta_rho_anom_with_candidates(
    xt_km: np.ndarray,
    yt_km: np.ndarray,
    theta_rho: np.ndarray,
    u_full: np.ndarray,
    v_full: np.ndarray,
    u_mean: float,
    v_mean: float,
    contour_paths: List[np.ndarray],
    best_poly: Optional[Pool],
    prefix: str,
    t_lag_idx: int,
    colormap: Optional[str],
    arrow_subsample: int,
    arrow_scale: float,
):
    """Plot theta_rho' with wind anomalies, all contours and highlight accepted polygon."""
    X, Y = np.meshgrid(xt_km, yt_km)
    theta_rho_pert = theta_rho - np.nanmean(theta_rho)
    vmin = np.nanpercentile(theta_rho_pert, 5)
    vmax = np.nanpercentile(theta_rho_pert, 95)
    fig, ax = plt.subplots(figsize=(6,6))
    im = ax.pcolormesh(X, Y, theta_rho_pert, shading="auto", cmap=(colormap or "viridis"), vmin=vmin, vmax=vmax)
    step = max(1, int(arrow_subsample))
    ax.quiver(X[::step, ::step], Y[::step, ::step], (u_full - u_mean)[::step, ::step], (v_full - v_mean)[::step, ::step],
              color="black", scale=arrow_scale, width=0.002)
    # all candidates
    for seg in contour_paths:
        ax.plot(seg[:,0], seg[:,1], color="white", linewidth=0.7, alpha=0.9)
    title = f"theta_rho' and candidates at t_lag={t_lag_idx}"
    # highlight accepted polygon
    if best_poly is not None:
        bx = best_poly.boundary_xy_km[:, 0]
        by = best_poly.boundary_xy_km[:, 1]
        ax.plot(bx, by, color="white", linewidth=2.0)
    else:
        title += " — NO_POLYGON_ACCEPTED"
    ax.set_title(title)
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    cbar = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02)
    cbar.set_label("theta_rho' (K)")
    out = f"{prefix}_trho_{int(t_lag_idx):05d}.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)


@dataclass
class Pool:
    time_index: int
    centroid_km: Tuple[float, float]
    area_km2: float
    eq_radius_km: float
    mean_theta_rho: float
    min_theta_rho: float
    boundary_xy_km: np.ndarray  # Nx2 array


def run_detection(
    data_root: str,
    start_index: int,
    minutes: int,
    z_index: int,
    qr_thresh_kgkg: Optional[float] = None,
    near_surface_levels: int = 1,
    qr_max_height_m: Optional[float] = 400.0,
    min_area_km2: float = 0.2,
    lag_minutes: int = 7,
    hessian_sigma_m: float = 150.0,
    sigma_rain_smooth_m: float = 150.0,
    use_advection_correction: bool = True,
    proximity_factor: float = 1.5,
    cover_rainy_min: float = 0.60,
    cover_poly_min: float = 0.10,
    aspect_min: float = 0.40,
    solidity_min: float = 0.55,
    output_prefix: Optional[str] = None,
    make_plots: bool = True,
    colormap: Optional[str] = None,
    arrow_subsample: int = 8,
    arrow_scale: float = 100,
) -> Dict[str, List[Pool]]:
    """Detect cold pools per the provided procedure and return pools per time.

    This function operates time-by-time; tracking is returned by a separate helper.
    """
    t_values, ntime = _time_values_and_count(data_root)
    # minute-by-minute indices based on time axis (seconds)
    t_values, ntime = _time_values_and_count(data_root)
    t0 = t_values[start_index]
    target_times = t0 + 60.0 * np.arange(minutes)
    idx = np.array([int(np.argmin(np.abs(t_values - tt))) for tt in target_times], dtype=int)
    idx = idx[idx < ntime]

    # get grid and resolution
    with _open(os.path.join(data_root, "rico.p.nc")) as dsp:
        xt = dsp["xt"].values
        yt = dsp["yt"].values
    dx = float(np.mean(np.diff(xt)))
    dy = float(np.mean(np.diff(yt)))
    km = 1000.0

    # Collect results keyed by lagged time index; create on demand
    results: Dict[str, List[Pool]] = {}

    # determine default colormap
    if colormap is None:
        colormap = "cmo.thermal" if "cmo.thermal" in plt.colormaps() else ("cmocean.thermal" if "cmocean.thermal" in plt.colormaps() else "viridis")

    for t_seed in idx:
        # 1) Rain seeding from q_r with smoothing + adaptive threshold at t_seed
        rainy, xt_m, yt_m, diag = _rain_mask_from_qr(
            data_root=data_root,
            t_index=t_seed,
            near_surface_levels=near_surface_levels,
            qr_thresh_kgkg=qr_thresh_kgkg,
            qr_max_height_m=qr_max_height_m,
            sigma_rain_smooth_m=sigma_rain_smooth_m,
            min_area_km2=min_area_km2,
        )
        # Seeding diagnostics
        print(f"SEED t={t_seed}: pre comps={diag['pre_components']} pix={diag['pre_pixels']} area_km2={diag['pre_area_km2']:.3f}; "
              f"post comps={diag['post_components']} pix={diag['post_pixels']} area_km2={diag['post_area_km2']:.3f}; "
              f"qr_max p95/p98/p99={diag['p95']:.2e}/{diag['p98']:.2e}/{diag['p99']:.2e}; thresh={diag['qr_thresh']:.2e}")
        if diag["post_components"] == 0:
            print(f"SEEDING_EMPTY at t={t_seed}")
        if make_plots and output_prefix:
            try:
                _plot_rain_distribution(xt_m/1000.0, yt_m/1000.0, diag, output_prefix, t_seed)
            except Exception:
                pass
        # four-connected components and fill holes per region
        structure = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=int)
        labeled, nlab = ndimage.label(rainy, structure=structure)
        if nlab == 0:
            continue
        # remove small regions (< min_area_km2)
        mask_keep = np.zeros_like(rainy, dtype=bool)
        labels = np.arange(1, nlab+1)
        for lab in labels:
            region = (labeled == lab)
            region = binary_fill_holes(region)
            area_km2 = region.sum() * dx * dy / (km*km)
            if area_km2 >= min_area_km2:
                mask_keep |= region
        labeled, nlab = ndimage.label(mask_keep, structure=structure)
        if nlab == 0:
            continue

        # centroids of rainy regions
        centroids_xy_km: List[Tuple[float, float]] = []
        for lab in range(1, nlab+1):
            region = (labeled == lab)
            if not np.any(region):
                continue
            iy, ix = np.nonzero(region)
            if ix.size == 0:
                continue
            # map to km using coordinate means (no rounding)
            x0_km = float(np.mean(xt[ix])) / km
            y0_km = float(np.mean(yt[iy])) / km
            centroids_xy_km.append((x0_km, y0_km))

        # 2) Ten-minute lag and optional advection of centroids
        t_lag_target = t_values[t_seed] + lag_minutes * 60.0
        # nearest index to lag time
        t_all = t_values
        t_lag_idx = int(np.argmin(np.abs(t_all - t_lag_target)))
        if t_lag_idx >= ntime:
            continue
        if use_advection_correction and len(centroids_xy_km) > 0:
            # domain-mean low-level wind at t_seed across near-surface levels
            try:
                with _open(os.path.join(data_root, "rico.u.nc")) as dsu, _open(os.path.join(data_root, "rico.v.nc")) as dsv:
                    nz_u = dsu.sizes.get("zt", 1)
                    nz_v = dsv.sizes.get("zt", 1)
                    nlev = max(1, min(int(near_surface_levels), min(nz_u, nz_v)))
                    u_da = dsu["u"].isel(time=t_seed, zt=slice(0, nlev))
                    v_da = dsv["v"].isel(time=t_seed, zt=slice(0, nlev))
                    # interpolate to centers
                    if "xm" in u_da.dims:
                        u_da = u_da.interp(xm=xt)
                    if "ym" in v_da.dims:
                        v_da = v_da.interp(ym=yt)
                    u_mean = float(np.nanmean(u_da.values))
                    v_mean = float(np.nanmean(v_da.values))
            except Exception:
                u_mean = 0.0
                v_mean = 0.0
            dt_s = float((t_values[t_lag_idx] - t_values[t_seed]))
            dx_km_adv = (u_mean * dt_s) / 1000.0
            dy_km_adv = (v_mean * dt_s) / 1000.0
            centroids_xy_km = [(cx + dx_km_adv, cy + dy_km_adv) for (cx, cy) in centroids_xy_km]

        # theta_rho at lag time
        theta_rho, xt_m, yt_m = _theta_rho_field(data_root, t_lag_idx, z_index)
        xt_km = xt_m / km
        yt_km = yt_m / km

        # wind means for anomalies in plots
        u_full, v_full, u_mean, v_mean = _winds_at_level_by_index(data_root, t_lag_idx, z_index, xt_m, yt_m)

        # domain-mean subtract for theta_rho perturbation if preferred
        theta_rho_pert = theta_rho - np.nanmean(theta_rho)

        # 3) For each rainy centroid build boundary from zero-contour of second radial derivative
        # compute sigma in pixel units from meters
        sigma_x_pix = float(hessian_sigma_m / dx)
        sigma_y_pix = float(hessian_sigma_m / dy)
        for cxy_km in centroids_xy_km:
            cx_km, cy_km = cxy_km
            # compute radial second derivative field and its zero contours
            d2r = _d2r_field(theta_rho, (cx_km*km, cy_km*km), xt_m, yt_m, (sigma_y_pix, sigma_x_pix))
            contour_paths = _zero_contour_paths_km(d2r, xt_km, yt_km)
            if len(contour_paths) == 0:
                continue

            # identify rainy region mask for overlap/proximity tests
            # Build rainy region around centroid: choose the label whose centroid is nearest to cxy
            # recompute centroids to find matching label index
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

            # Evaluate each candidate zero-contour polygon on native grid
            best_poly: Optional[Pool] = None
            best_centroid_dist = np.inf
            Xk, Yk = np.meshgrid(xt_km, yt_km)
            from matplotlib.path import Path as MplPath
            for seg_km in contour_paths:
                path = MplPath(seg_km)
                inside = path.contains_points(np.column_stack([Xk.ravel(), Yk.ravel()]))
                poly_mask = inside.reshape(Xk.shape)
                if poly_mask.sum() == 0:
                    continue

                # centroid of polygon on native grid in physical coordinates (no rounding)
                iy, ix = np.nonzero(poly_mask)
                if ix.size == 0:
                    continue
                cx_poly_km = float(np.mean(xt_km[ix]))
                cy_poly_km = float(np.mean(yt_km[iy]))
                cen_dist = np.hypot(cx_poly_km - cx_km, cy_poly_km - cy_km)

                # Proximity test (within proximity_factor × equivalent diameter of rainy region)
                if cen_dist > (proximity_factor * rain_eq_diam_km):
                    continue

                # Overlap tests (mask-based)
                inter = (poly_mask & rain_mask).sum()
                rain_px = rain_mask.sum()
                poly_px = poly_mask.sum()
                if rain_px == 0 or poly_px == 0:
                    continue
                cover_rainy = inter / rain_px
                cover_poly = inter / poly_px
                if not (cover_rainy >= cover_rainy_min and cover_poly >= cover_poly_min):
                    continue

                # Shape quality
                props = _region_properties(poly_mask, dx, dy)
                if not (props["aspect"] >= aspect_min and props["solidity"] >= solidity_min):
                    continue

                # Candidate passes; compute fields within polygon
                theta_vals = theta_rho[poly_mask]
                mean_tr = float(np.nanmean(theta_vals))
                min_tr = float(np.nanmin(theta_vals))
                area_km2 = props["area"] / (km*km)
                eq_radius_km = props["eq_radius"] / km

                # boundary from the contour segment (already in km coords)
                boundary_xy_km = seg_km

                pool = Pool(
                    time_index=int(t_lag_idx),
                    centroid_km=(cx_poly_km, cy_poly_km),
                    area_km2=float(area_km2),
                    eq_radius_km=float(eq_radius_km),
                    mean_theta_rho=float(mean_tr),
                    min_theta_rho=float(min_tr),
                    boundary_xy_km=boundary_xy_km,
                )

                if cen_dist < best_centroid_dist:
                    best_poly = pool
                    best_centroid_dist = cen_dist

            if best_poly is not None:
                k = str(int(t_lag_idx))
                results.setdefault(k, []).append(best_poly)

            # Diagnostic plot of theta_rho' and candidates
            if make_plots and output_prefix:
                try:
                    _plot_theta_rho_anom_with_candidates(
                        xt_km, yt_km, theta_rho, u_full, v_full, u_mean, v_mean,
                        contour_paths, best_poly, output_prefix, t_lag_idx, colormap,
                        arrow_subsample, arrow_scale,
                    )
                except Exception:
                    pass

    return results


def track_pools_over_time(
    pools_by_time: Dict[str, List[Pool]],
    data_root: Optional[str] = None,
    z_index: int = 0,
    use_advection_correction: bool = False,
    min_overlap: float = 0.30,
    track_max_dist_factor: float = 2.0,
) -> List[Dict]:
    """Link pools between consecutive times by centroid distance and mutual overlap.

    Optionally advect previous polygons by domain-mean low-level wind before overlap.
    Uses native grid for rasterization if data_root is provided; otherwise infers a coarse grid.
    """
    # time keys as sorted integers
    times = sorted([int(k) for k in pools_by_time.keys()])
    tracks: List[Dict] = []
    active: Dict[int, Dict] = {}
    next_track_id = 1

    # grid for rasterization
    if data_root is not None:
        with _open(os.path.join(data_root, "rico.p.nc")) as dsp:
            xt_m = dsp["xt"].values
            yt_m = dsp["yt"].values
            t_vals = dsp["time"].values
        xt_km = xt_m / 1000.0
        yt_km = yt_m / 1000.0
    else:
        # fallback: infer from boundaries
        has_any = any(len(pools_by_time[str(t)]) > 0 for t in times)
        if not has_any:
            return []
        all_pts = np.vstack([pool.boundary_xy_km for t in times for pool in pools_by_time[str(t)]])
        x_min, x_max = all_pts[:, 0].min(), all_pts[:, 0].max()
        y_min, y_max = all_pts[:, 1].min(), all_pts[:, 1].max()
        xt_km = np.linspace(x_min, x_max, 256)
        yt_km = np.linspace(y_min, y_max, 256)
        t_vals = np.array(times, dtype=float)

    # helper: polygon mask rasterization on native grid
    from matplotlib.path import Path as MplPath
    Xk, Yk = np.meshgrid(xt_km, yt_km)
    pts = np.column_stack([Xk.ravel(), Yk.ravel()])

    def polygon_mask_km(vertices_km: np.ndarray) -> np.ndarray:
        path = MplPath(vertices_km)
        inside = path.contains_points(pts)
        return inside.reshape(Xk.shape)

    # iterate times
    prev_time = None
    for t in times:
        pools = pools_by_time[str(t)]
        masks = [polygon_mask_km(p.boundary_xy_km) for p in pools]
        if prev_time is None:
            for p in pools:
                tracks.append({
                    "id": next_track_id,
                    "pools": [p],
                    "start": t,
                    "end": t,
                })
                active[next_track_id] = tracks[-1]
                next_track_id += 1
        else:
            used = set()
            for tid, tr in list(active.items()):
                p_prev = tr["pools"][-1]
                # optionally advect previous polygon by mean wind
                prev_vertices = p_prev.boundary_xy_km
                if use_advection_correction and data_root is not None:
                    # mean wind at previous time on low level
                    try:
                        with _open(os.path.join(data_root, "rico.u.nc")) as dsu, _open(os.path.join(data_root, "rico.v.nc")) as dsv:
                            u_da = dsu["u"].isel(time=prev_time, zt=z_index)
                            v_da = dsv["v"].isel(time=prev_time, zt=z_index)
                            if "xm" in u_da.dims:
                                u_da = u_da.interp(xm=xt_m)
                            if "ym" in v_da.dims:
                                v_da = v_da.interp(ym=yt_m)
                            u_mean = float(np.nanmean(u_da.values))
                            v_mean = float(np.nanmean(v_da.values))
                    except Exception:
                        u_mean = 0.0
                        v_mean = 0.0
                    dt_s = float(t_vals[t] - t_vals[prev_time]) if data_root is not None else 60.0
                    dx_km = (u_mean * dt_s) / 1000.0
                    dy_km = (v_mean * dt_s) / 1000.0
                    prev_vertices = prev_vertices + np.array([dx_km, dy_km])

                m_prev = polygon_mask_km(prev_vertices)

                # find best candidate
                best_j = None
                best_d = np.inf
                for j, p in enumerate(pools):
                    if j in used:
                        continue
                    d = np.hypot(p.centroid_km[0] - p_prev.centroid_km[0], p.centroid_km[1] - p_prev.centroid_km[1])
                    if d <= (track_max_dist_factor * p_prev.eq_radius_km) and d < best_d:
                        m_curr = masks[j]
                        inter = (m_prev & m_curr).sum()
                        if inter == 0:
                            continue
                        o_prev = inter / m_prev.sum()
                        o_curr = inter / m_curr.sum()
                        if o_prev >= min_overlap and o_curr >= min_overlap:
                            best_d = d
                            best_j = j
                if best_j is not None:
                    tr["pools"].append(pools[best_j])
                    tr["end"] = t
                    used.add(best_j)
                else:
                    active.pop(tid, None)
            for j, p in enumerate(pools):
                if j in used:
                    continue
                tracks.append({
                    "id": next_track_id,
                    "pools": [p],
                    "start": t,
                    "end": t,
                })
                active[next_track_id] = tracks[-1]
                next_track_id += 1
        prev_time = t

    # enrich tracks with summary stats
    for tr in tracks:
        pools = tr["pools"]
        tr["lifetime_minutes"] = len(pools)
        if len(pools) >= 2:
            x = np.arange(len(pools))
            y = np.array([p.area_km2 for p in pools])
            coef = np.polyfit(x, y, 1)
            tr["area_growth_km2_per_min"] = float(coef[0])
            dists = [np.hypot(pools[i+1].centroid_km[0]-pools[i].centroid_km[0], pools[i+1].centroid_km[1]-pools[i].centroid_km[1]) for i in range(len(pools)-1)]
            tr["propagation_km_per_min"] = float(np.mean(dists))
        else:
            tr["area_growth_km2_per_min"] = 0.0
            tr["propagation_km_per_min"] = 0.0

    return tracks
