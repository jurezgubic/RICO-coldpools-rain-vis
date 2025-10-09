import os
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from .physics import (
    calculate_physics_variables,
    _ensure_kgkg,
    g,
    relative_humidity_from_p_T_qv,
)


def _open(path: str) -> xr.Dataset:
    # keep numeric time axis in seconds; I only need relative minutes
    return xr.open_dataset(path, decode_times=False)


def _nearest_index(values: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(values - target)))


def _load_time_coord(ds: xr.Dataset) -> np.ndarray:
    t = ds["time"].values
    return t


def _time_values_and_count(data_root: str) -> Tuple[np.ndarray, int]:
    """Raw time values and count from a base file."""
    with _open(os.path.join(data_root, "rico.p.nc")) as ds:
        t = ds["time"].values
    return t, int(t.shape[0])


def _get2d_by_index(ds_path: str, var: str, t_index: int, z_index: int,
                    x_name: str = "xt", y_name: str = "yt") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Single (yt, xt) 2D slice at time/z index."""
    with _open(ds_path) as ds:
        da = ds[var].isel(time=t_index, zt=z_index)
        x = np.array(ds[x_name].values)
        y = np.array(ds[y_name].values)
        arr = da.load().values.astype(float)
        # Mask fill values if present
        fill = da.attrs.get("_FillValue", None)
        if fill is not None:
            arr = np.where(arr == fill, np.nan, arr)
    return arr, x, y


def _interp_to_center(u_da: xr.DataArray, xt: np.ndarray = None, yt: np.ndarray = None) -> xr.DataArray:
    """Interpolate staggered winds to centers so arrows line up with scalars."""
    kwargs = {}
    if xt is not None and "xm" in u_da.dims:
        kwargs["xm"] = xt
    if yt is not None and "ym" in u_da.dims:
        kwargs["ym"] = yt
    if not kwargs:
        return u_da
    return u_da.interp(**kwargs)


def _winds_at_level_by_index(data_root: str, t_index: int, z_index: int,
                             xt: np.ndarray, yt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """u,v on (yt, xt) at time/z index."""
    # u: dims (time, zt, yt, xm)
    with _open(os.path.join(data_root, "rico.u.nc")) as dsu:
        u_da = dsu["u"].isel(time=t_index, zt=z_index)
        u_da = _interp_to_center(u_da, xt=xt)
        u = u_da.load().values.astype(float)
        fill = u_da.attrs.get("_FillValue", None)
        if fill is not None:
            u = np.where(u == fill, np.nan, u)

    # v: dims (time, zt, ym, xt)
    with _open(os.path.join(data_root, "rico.v.nc")) as dsv:
        v_da = dsv["v"].isel(time=t_index, zt=z_index)
        v_da = _interp_to_center(v_da, yt=yt)
        v = v_da.load().values.astype(float)
        fill = v_da.attrs.get("_FillValue", None)
        if fill is not None:
            v = np.where(v == fill, np.nan, v)

    return u, v


def _format_minutes_since(t0: float, t: float) -> int:
    return int(round((t - t0) / 60.0))


def compute_fields_for_indices(
    data_root: str,
    t_indices: Sequence[int],
    z_index: int,
    xlim_km: Optional[Tuple[float, float]] = None,
    ylim_km: Optional[Tuple[float, float]] = None,
    use_clear_air_reference: bool = True,
    threshold_kgkg: float = 5e-5,
    rain_filename: str = "rico.r.nc",
    compute_rh: bool = False,
) -> List[dict]:
    """theta_v, density, winds, buoyancy for a set of time indices at one level.
    I can crop to the same subdomain across fields.
    """
    # Grid from pressure file
    with _open(os.path.join(data_root, "rico.p.nc")) as dsp:
        z = dsp["zt"].values
        xt = dsp["xt"].values
        yt = dsp["yt"].values

    # Determine crop indices if any
    x_km = xt / 1000.0
    y_km = yt / 1000.0
    if xlim_km is None:
        ix0, ix1 = 0, len(xt)
    else:
        ix0 = int(np.searchsorted(x_km, xlim_km[0], side="left"))
        ix1 = int(np.searchsorted(x_km, xlim_km[1], side="right"))
    if ylim_km is None:
        iy0, iy1 = 0, len(yt)
    else:
        iy0 = int(np.searchsorted(y_km, ylim_km[0], side="left"))
        iy1 = int(np.searchsorted(y_km, ylim_km[1], side="right"))

    xt_sub = xt[ix0:ix1]
    yt_sub = yt[iy0:iy1]

    rain_path = os.path.join(data_root, rain_filename)
    have_rain = os.path.exists(rain_path)
    if not have_rain:
        print("no rain file")

    results = []
    for ti in t_indices:
        # Scalars at centers and crop
        p2d, _, _ = _get2d_by_index(os.path.join(data_root, "rico.p.nc"), "p", ti, z_index)
        p2d = p2d[iy0:iy1, ix0:ix1]

        thl2d, _, _ = _get2d_by_index(os.path.join(data_root, "rico.t.nc"), "t", ti, z_index)
        thl2d = thl2d[iy0:iy1, ix0:ix1]

        qt2d, _, _ = _get2d_by_index(os.path.join(data_root, "rico.q.nc"), "q", ti, z_index)
        qt2d = qt2d[iy0:iy1, ix0:ix1]

        ql2d, _, _ = _get2d_by_index(os.path.join(data_root, "rico.l.nc"), "l", ti, z_index)
        ql2d = ql2d[iy0:iy1, ix0:ix1]

        qr2d = None
        if have_rain:
            try:
                qr2d, _, _ = _get2d_by_index(rain_path, "r", ti, z_index)
                qr2d = qr2d[iy0:iy1, ix0:ix1]
            except Exception:
                qr2d = None

        # unit conversions for water species
        qt2d = _ensure_kgkg(qt2d)
        ql2d = _ensure_kgkg(ql2d)
        qr2d = _ensure_kgkg(qr2d) if qr2d is not None else 0.0

        # physics
        T2d, rho2d, thv2d, _B_domain = calculate_physics_variables(p2d, thl2d, ql2d, qt2d, qr2d)

        # Optional: relative humidity at this level
        rh2d = None
        if compute_rh:
            qv2d = np.clip(qt2d - ql2d - (qr2d if isinstance(qr2d, np.ndarray) else qr2d), 0.0, None)
            rh2d = relative_humidity_from_p_T_qv(p2d, T2d, qv2d)

        # reference for buoyancy
        if use_clear_air_reference:
            thr = float(threshold_kgkg)
            mask = (ql2d + (qr2d if isinstance(qr2d, np.ndarray) else qr2d)) < thr
            if np.any(mask):
                thv_ref = np.nanmean(thv2d[mask])
            else:
                thv_ref = np.nanmean(thv2d)
        else:
            thv_ref = np.nanmean(thv2d)

        B2d = g * (thv2d - thv_ref) / thv_ref

        # Winds: load full domain, compute domain-mean, then crop
        u_full, v_full = _winds_at_level_by_index(data_root, ti, z_index, xt=xt, yt=yt)
        u_mean = float(np.nanmean(u_full))
        v_mean = float(np.nanmean(v_full))
        u2d = u_full[iy0:iy1, ix0:ix1]
        v2d = v_full[iy0:iy1, ix0:ix1]

        result_item = {
            "time_index": ti,
            "theta_v": thv2d,
            "rho": rho2d,
            "B": B2d,
            "u": u2d,
            "v": v2d,
            "u_mean": u_mean,
            "v_mean": v_mean,
            "xt": xt_sub,
            "yt": yt_sub,
            "z": z[z_index],
        }
        if compute_rh:
            result_item["rh"] = rh2d
        results.append(result_item)

    return results


def panel_plot(
    results: List[dict],
    time_values: np.ndarray,
    outfile: str = "cold_pools_panels.png",
    colormap: Optional[str] = None,
    arrow_subsample: int = 8,
    arrow_scale: float = 100,
    subtract_mean_wind: bool = False,
    scalar_field: str = "theta_v",
):
    n = len(results)
    ncols = 3 if n >= 3 else n
    nrows = int(np.ceil(n / ncols))

    # Choose scalar to display (theta_v or RH)
    scalar_field = scalar_field.lower()
    if scalar_field not in ("theta_v", "rh"):
        raise ValueError("scalar_field must be 'theta_v' or 'rh'")

    # color limits shared across panels
    if scalar_field == "theta_v":
        all_vals = np.concatenate([np.ravel(r["theta_v"]) for r in results])
    else:
        # RH in percent for display
        all_vals = np.concatenate([np.ravel(r.get("rh")) for r in results]) * 100.0
    vmin = np.nanpercentile(all_vals, 2)
    vmax = np.nanpercentile(all_vals, 98)

    # Choose default colormap if not specified by config
    if colormap is None:
        if scalar_field == "theta_v":
            colormap = (
                "cmo.thermal"
                if "cmo.thermal" in plt.colormaps()
                else ("cmocean.thermal" if "cmocean.thermal" in plt.colormaps() else "viridis")
            )
        else:
            colormap = (
                "cmo.haline"
                if "cmo.haline" in plt.colormaps()
                else ("cmocean.haline" if "cmocean.haline" in plt.colormaps() else "viridis")
            )

    # thin the arrow field for readability
    def subsample(arr, step=arrow_subsample):
        return arr[::step, ::step]

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8*ncols, 4.8*nrows), constrained_layout=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.ravel()

    t0 = time_values[results[0]["time_index"]]

    for ax, r in zip(axes, results):
        xt = r["xt"]/1000.0
        yt = r["yt"]/1000.0
        X, Y = np.meshgrid(xt, yt)

        if scalar_field == "theta_v":
            scalar = r["theta_v"]
        else:
            if "rh" not in r:
                raise KeyError("RH not present in results; set compute_rh=True when computing fields.")
            scalar = r["rh"] * 100.0  # display as percent
        im = ax.pcolormesh(X, Y, scalar, shading="auto", cmap=colormap, vmin=vmin, vmax=vmax)

        # Buoyancy zero contour (cold-pool edge)
        try:
            cs = ax.contour(X, Y, r["B"], levels=[0.0], colors="white", linewidths=1.0)
        except Exception:
            cs = None

        # winds (thin sampling) plotted as arrows
        step = max(1, int(arrow_subsample))
        u_s = subsample(r["u"], step)
        v_s = subsample(r["v"], step)
        if subtract_mean_wind:
            u_s = u_s - r["u_mean"]
            v_s = v_s - r["v_mean"]
        Xs = X[::step, ::step]
        Ys = Y[::step, ::step]
        ax.quiver(Xs, Ys, u_s, v_s, color="black", scale=arrow_scale, width=0.002)

        # Title with timestep index and relative time
        ti = r["time_index"]
        minutes = int(round((time_values[ti] - t0) / 60.0))
        ax.set_title(f"timestep {ti} (t = {minutes} min), z ~ {r['z']:.1f} m")
        ax.set_xlabel("x (km)")
        ax.set_ylabel("y (km)")

    # Colorbar
    cbar = fig.colorbar(im, ax=axes.tolist(), shrink=0.95, pad=0.02)
    cbar.set_label("theta_v (K)" if scalar_field == "theta_v" else "RH (%)")

    fig.savefig(outfile, dpi=400)
    return outfile


def _advect_window(center_xy_km: Tuple[float, float], minutes_since_start: float,
                   mean_wind_ms: Tuple[float, float]) -> Tuple[float, float]:
    """Return new center (km) after advection with mean wind for given minutes.

    mean_wind_ms is (u_mean, v_mean) in m/s on domain. Convert to km/min.
    """
    u_ms, v_ms = mean_wind_ms
    factor = 60.0  # seconds per minute
    dx_km = (u_ms * minutes_since_start * factor) / 1000.0
    dy_km = (v_ms * minutes_since_start * factor) / 1000.0
    xk, yk = center_xy_km
    return xk + dx_km, yk + dy_km


def render_tracking_gif(
    data_root: str,
    start_index: int,
    minutes: int,
    z_index: int,
    center_km: Tuple[float, float],
    half_window_km: Tuple[float, float],
    outfile: str,
    colormap: Optional[str] = None,
    arrow_subsample: int = 8,
    arrow_scale: float = 100,
):
    """GIF with a fixed-size window advected by mean wind.
    One panel per frame; arrows are anomalies (mean removed)."""
    import imageio.v2 as imageio

    # Helper: robustly convert a Matplotlib figure to an RGB image array across backends
    def _fig_to_rgb(fig) -> np.ndarray:
        fig.canvas.draw()
        h, w = fig.canvas.get_width_height()[1], fig.canvas.get_width_height()[0]
        # Try RGB first (Agg backends)
        if hasattr(fig.canvas, "tostring_rgb"):
            buf = fig.canvas.tostring_rgb()
            arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
            return arr
        # Fallback: ARGB (QtAgg)
        if hasattr(fig.canvas, "tostring_argb"):
            buf = fig.canvas.tostring_argb()
            arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
            # ARGB -> RGB (drop alpha, reorder)
            return arr[:, :, 1:4]
        # Last resort: buffer_rgba if available
        if hasattr(fig.canvas, "buffer_rgba"):
            buf = fig.canvas.buffer_rgba()
            arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
            return arr[:, :, :3]
        raise RuntimeError("Unsupported Matplotlib canvas for GIF rendering.")

    # Load time axis
    t_values, ntime = _time_values_and_count(data_root)
    # Map minutes to indices by assuming 60 s per minute time axis spacing in seconds.
    # Here we interpret "minute-by-minute" as contiguous time indices, as files are in seconds.
    idx = start_index + np.arange(minutes)
    if np.any(idx >= ntime):
        raise IndexError("Chosen GIF frames exceed available timesteps.")

    # Grid from a base file to get coordinates
    with _open(os.path.join(data_root, "rico.p.nc")) as dsp:
        xt_full = dsp["xt"].values
        yt_full = dsp["yt"].values

    # helper to convert km window to index slices, clamped to domain
    def window_indices(x_center_km: float, y_center_km: float, hx_km: float, hy_km: float):
        x_km = xt_full / 1000.0
        y_km = yt_full / 1000.0
        x0 = x_center_km - hx_km
        x1 = x_center_km + hx_km
        y0 = y_center_km - hy_km
        y1 = y_center_km + hy_km
        ix0 = int(np.searchsorted(x_km, x0, side="left"))
        ix1 = int(np.searchsorted(x_km, x1, side="right"))
        iy0 = int(np.searchsorted(y_km, y0, side="left"))
        iy1 = int(np.searchsorted(y_km, y1, side="right"))
        ix0 = max(0, min(ix0, len(xt_full)-1))
        ix1 = max(ix0+1, min(ix1, len(xt_full)))
        iy0 = max(0, min(iy0, len(yt_full)-1))
        iy1 = max(iy0+1, min(iy1, len(yt_full)))
        return (ix0, ix1, iy0, iy1)

    # Prepare frames
    frames = []

    # Determine default colormap once
    if colormap is None:
        colormap = "cmo.thermal" if "cmo.thermal" in plt.colormaps() else ("cmocean.thermal" if "cmocean.thermal" in plt.colormaps() else "viridis")

    # Track center advected by domain-mean wind using instantaneous means each frame
    xck, yck = center_km
    hxk, hyk = half_window_km

    t0 = t_values[idx[0]]

    for t_i in idx:
        # Compute fields on full domain at this time to get means and subdomain
        results = compute_fields_for_indices(
            data_root=data_root,
            t_indices=[t_i],
            z_index=z_index,
            xlim_km=None,
            ylim_km=None,
            use_clear_air_reference=True,
            threshold_kgkg=5e-5,
            rain_filename="rico.r.nc",
        )
        r = results[0]

        # Update advection based on domain-mean wind up to current time
        minutes_since_start = int(round((t_values[t_i] - t0) / 60.0))
        xck, yck = _advect_window(center_xy_km=(center_km[0], center_km[1]),
                                  minutes_since_start=minutes_since_start,
                                  mean_wind_ms=(r["u_mean"], r["v_mean"]))

        ix0, ix1, iy0, iy1 = window_indices(xck, yck, hxk, hyk)

        # Crop fields
        xt = r["xt"][ix0:ix1] / 1000.0
        yt = r["yt"][iy0:iy1] / 1000.0
        X, Y = np.meshgrid(xt, yt)
        thv = r["theta_v"][iy0:iy1, ix0:ix1]
        B = r["B"][iy0:iy1, ix0:ix1]
        u = r["u"][iy0:iy1, ix0:ix1] - r["u_mean"]
        v = r["v"][iy0:iy1, ix0:ix1] - r["v_mean"]

        # Figure
        fig, ax = plt.subplots(figsize=(5.2, 5.2))
        vmin = np.nanpercentile(thv, 2)
        vmax = np.nanpercentile(thv, 98)
        im = ax.pcolormesh(X, Y, thv, shading="auto", cmap=colormap, vmin=vmin, vmax=vmax)
        try:
            ax.contour(X, Y, B, levels=[0.0], colors="white", linewidths=1.0)
        except Exception:
            pass

        step = max(1, int(arrow_subsample))
        ax.quiver(X[::step, ::step], Y[::step, ::step], u[::step, ::step], v[::step, ::step],
                  color="black", scale=arrow_scale, width=0.002)

        minutes_rel = int(round((t_values[t_i] - t0) / 60.0))
        ax.set_title(f"timestep {t_i} (t = {minutes_rel} min), z ~ {r['z']:.1f} m")
        ax.set_xlabel("x (km)")
        ax.set_ylabel("y (km)")
        cbar = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02)
        cbar.set_label("theta_v (K)")

        # Convert figure to image array (backend-agnostic)
        fig_array = _fig_to_rgb(fig)
        frames.append(fig_array)
        plt.close(fig)

    # Save GIF (10 fps => 6s for 60 frames; we want 1 fps)
    imageio.mimsave(outfile, frames, fps=1)
    return outfile


def choose_time_indices(start_index: int, time_stride: int, n_panels: int, data_root: str) -> Tuple[np.ndarray, np.ndarray]:
    t_values, ntime = _time_values_and_count(data_root)
    idx = start_index + np.arange(n_panels) * time_stride
    if np.any(idx >= ntime):
        raise IndexError("Chosen time indices exceed available timesteps.")
    return idx, t_values


def main(
    data_root: str,
    start_index: int = 0,
    time_stride: int = 10,
    n_panels: int = 6,
    z_index: int = 2,
    xlim_km: Optional[Tuple[float, float]] = None,
    ylim_km: Optional[Tuple[float, float]] = None,
    use_clear_air_reference: bool = True,
    threshold_kgkg: float = 5e-5,
    rain_filename: str = "rico.r.nc",
    colormap: Optional[str] = None,
    arrow_subsample: int = 8,
    arrow_scale: float = 100,
    outfile: str = "cold_pools_panels.png",
    outfile_wind_anom: Optional[str] = None,
    make_tracking_gif: bool = False,
    gif_minutes: int = 0,
    gif_center_km: Tuple[float, float] = (0.0, 0.0),
    gif_half_window_km: Tuple[float, float] = (2.0, 2.0),
    gif_outfile: Optional[str] = None,
    panel_scalar: str = "theta_v",
):
    idx, t_values = choose_time_indices(
        start_index=start_index,
        time_stride=time_stride,
        n_panels=n_panels,
        data_root=data_root,
    )

    results = compute_fields_for_indices(
        data_root=data_root,
        t_indices=idx,
        z_index=z_index,
        xlim_km=xlim_km,
        ylim_km=ylim_km,
        use_clear_air_reference=use_clear_air_reference,
        threshold_kgkg=threshold_kgkg,
        rain_filename=rain_filename,
        compute_rh=(panel_scalar.lower() == "rh"),
    )

    out = panel_plot(
        results,
        t_values,
        outfile=outfile,
        colormap=colormap,
        arrow_subsample=arrow_subsample,
        arrow_scale=arrow_scale,
        subtract_mean_wind=False,
        scalar_field=panel_scalar,
    )
    print(f"Saved: {out}")

    if outfile_wind_anom is not None:
        out2 = panel_plot(
            results,
            t_values,
            outfile=outfile_wind_anom,
            colormap=colormap,
            arrow_subsample=arrow_subsample,
            arrow_scale=arrow_scale,
            subtract_mean_wind=True,
            scalar_field=panel_scalar,
        )
        print(f"Saved: {out2}")

    if make_tracking_gif and gif_minutes > 0 and gif_outfile:
        gif_path = render_tracking_gif(
            data_root=data_root,
            start_index=start_index,
            minutes=gif_minutes,
            z_index=z_index,
            center_km=gif_center_km,
            half_window_km=gif_half_window_km,
            outfile=gif_outfile,
            colormap=colormap,
            arrow_subsample=arrow_subsample,
            arrow_scale=arrow_scale,
        )
        print(f"Saved: {gif_path}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Cold-pool panel plot from RICO LES outputs")
    p.add_argument("--out", type=str, default="cold_pools_panels.png", help="Output PNG path")
    p.add_argument("--out_anom", type=str, default=None, help="Optional PNG with mean-wind-subtracted arrows")
    args = p.parse_args()

    dr = os.environ.get("RICO_DATA")
    if not dr or not os.path.isdir(dr):
        raise SystemExit("set RICO_DATA to a valid data directory or use main.py")

    main(data_root=dr, outfile=args.out, outfile_wind_anom=args.out_anom)
