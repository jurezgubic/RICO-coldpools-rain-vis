import numpy as np


# constants I keep lowercase
r_d = 287.04       # gas constant dry air [J/kg/K]
r_v = 461.5        # gas constant water vapor [J/kg/K]
c_pd = 1005.0      # specific heat dry air at constant pressure [J/kg/K]
c_pv = 1850.0      # specific heat water vapor at constant pressure [J/kg/K]
l_v = 2.5e6        # latent heat of vaporization [J/kg]
p0 = 100000.0      # reference pressure [Pa]
epsilon = 0.622    # R_d / R_v ratio
rho_l = 1000.0     # density of liquid water [kg/m^3]
g = 9.80665        # gravity [m/s^2]


def _ensure_kgkg(q):
    """Keep everything in kg/kg; if values look like g/kg, convert.
    Only print when I actually convert.
    """
    if q is None:
        return None
    # typical q_t in tropics is approx 10 g/kg -> 0.01 kg/kg
    # If values exceed 0.2, they are probably g/kg.
    max_abs = np.nanmax(np.abs(q))
    # Print only when conversion occurs; otherwise stay quiet
    if not np.isfinite(max_abs):
        return q
    if max_abs > 0.2:
        try:
            print(f"ensure_kgkg: max={float(max_abs):.4g} -> converting g/kg to kg/kg")
        except Exception:
            pass
        return q / 1000.0
    return q


def calculate_physics_variables(p_values, theta_l_values, q_l_values, q_t_values,
                                q_r_values=None):
    """Compute T, rho, theta_v, and B from theta_l/p and water species.
    q_r is optional (defaults to 0).
    """

    # Convert g/kg -> kg/kg if needed
    q_l = _ensure_kgkg(np.asarray(q_l_values, dtype=float))
    q_t = _ensure_kgkg(np.asarray(q_t_values, dtype=float))
    q_r = _ensure_kgkg(np.asarray(q_r_values, dtype=float)) if q_r_values is not None else 0.0

    # Vapor as remainder. If rain exists, assume q_t = q_v + q_l + q_r.
    q_v = np.clip(q_t - q_l - (q_r if isinstance(q_r, np.ndarray) else q_r), 0.0, None)

    p = np.asarray(p_values, dtype=float)
    theta_l = np.asarray(theta_l_values, dtype=float)

    # moist-adjusted kappa
    kappa = (r_d / c_pd) * ((1.0 + q_v / epsilon) / (1.0 + q_v * (c_pv / c_pd)))

    # temperature from theta_l and p
    T = theta_l * (c_pd / (c_pd - l_v * q_l)) * (p0 / p) ** (-kappa)

    # partial pressure of vapor from mixing ratio
    p_v = (q_v / (q_v + epsilon)) * p

    # density: dry air + vapor; then add condensate mass explicitly
    rho = (p - p_v) / (r_d * T) + (p_v / (r_v * T))
    # add condensate mass explicitly (liquid + rain) with bulk density of water
    rho = rho + (q_l + (q_r if isinstance(q_r, np.ndarray) else q_r)) * rho_l

    # virtual temperature and theta_v
    q_r_term = q_r if isinstance(q_r, np.ndarray) else q_r
    T_v = T * (1.0 + 0.61 * q_v - q_l - q_r_term)
    theta_v = T_v * (p0 / p) ** (r_d / c_pd)

    # buoyancy relative to planar-mean theta_v
    theta_v_mean = np.nanmean(theta_v)
    B = g * (theta_v - theta_v_mean) / theta_v_mean

    return T, rho, theta_v, B


def saturation_vapor_pressure_pa(T: np.ndarray) -> np.ndarray:
    """Saturation vapor pressure over liquid water (Pa).

    Uses an old Bolton? formula, should be okay for boundary-layer temperatures.
    """
    T = np.asarray(T, dtype=float)
    Tc = T - 273.15
    # e_s in hPa, then convert to Pa
    e_s_hpa = 6.112 * np.exp(17.67 * Tc / (Tc + 243.5))
    return e_s_hpa * 100.0


def relative_humidity_from_p_T_qv(p: np.ndarray, T: np.ndarray, q_v_values: np.ndarray) -> np.ndarray:
    """Relative humidity (0-1) from total pressure p (Pa), temperature T (K), and vapor mixing ratio q_v (kg/kg).

    RH = e / e_s(T), where e = p_v = (q_v/(epsilon + q_v)) * p.
    Values are clipped at [0, 1.05] to avoid numerical artifacts.
    """
    q_v = _ensure_kgkg(np.asarray(q_v_values, dtype=float))
    p = np.asarray(p, dtype=float)
    T = np.asarray(T, dtype=float)
    # vapor partial pressure from mixing ratio
    p_v = (q_v / (q_v + epsilon)) * p
    e_s = saturation_vapor_pressure_pa(T)
    with np.errstate(divide="ignore", invalid="ignore"):
        RH = p_v / e_s
    # clip for safety
    RH = np.clip(RH, 0.0, 1.05)
    return RH


def theta_from_T_p(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Dry potential temperature from T and p (theta = T * (p0/p)^(R_d/c_pd))."""
    return np.asarray(T, dtype=float) * (p0 / np.asarray(p, dtype=float)) ** (r_d / c_pd)


from typing import Union


def density_potential_temperature(p_values,
                                  T_values,
                                  q_v_values,
                                  q_l_values: Union[float, np.ndarray] = 0.0,
                                  q_i_values: Union[float, np.ndarray] = 0.0,
                                  q_r_values: Union[float, np.ndarray] = 0.0):
    """Include rain in both mass loading and condensate term.
    
    theta_rho = theta * (1 + (R_v/R_d) q_v - q_l - q_i) / (1 + q_t)
    
    where theta = T * (p0/p)^(R_d/c_pd),
    
    q_t = q_v + q_l + q_i, and all q are in kg/kg.
    """
    
    q_v = _ensure_kgkg(np.asarray(q_v_values, dtype=float))
    q_l = _ensure_kgkg(np.asarray(q_l_values, dtype=float)) if q_l_values is not None else 0.0
    q_i = _ensure_kgkg(np.asarray(q_i_values, dtype=float)) if q_i_values is not None else 0.0

    q_r = _ensure_kgkg(np.asarray(q_r_values, dtype=float)) if q_r_values is not None else 0.0

    theta = theta_from_T_p(T_values, p_values)
    # include rain water in total mass loading
    q_t = (
        q_v
        + (q_l if isinstance(q_l, np.ndarray) else q_l)
        + (q_i if isinstance(q_i, np.ndarray) else q_i)
        + (q_r if isinstance(q_r, np.ndarray) else q_r)
    )
    # subtract condensate (including rain) from virtual component multiplier
    q_l_term = (q_l if isinstance(q_l, np.ndarray) else q_l)
    q_i_term = (q_i if isinstance(q_i, np.ndarray) else q_i)
    q_r_term = (q_r if isinstance(q_r, np.ndarray) else q_r)
    factor = 1.0 + (r_v / r_d) * q_v - q_l_term - q_i_term - q_r_term
    theta_rho = theta * factor / (1.0 + q_t)
    return theta_rho
