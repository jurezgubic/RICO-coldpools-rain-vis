import numpy as np


# physical constants (lowercase by preference)
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
    """Convert mixing ratio that may be provided in g/kg to kg/kg.

    quick check: print max and guess units, then convert if needed.
    """
    if q is None:
        return None
    # typical q_t in tropics is approx 10 g/kg -> 0.01 kg/kg
    # If values exceed 0.2, they are probably g/kg.
    max_abs = np.nanmax(np.abs(q))
    # Print about units guess, but don't fail
    try:
        print(f"ensure_kgkg: max={float(max_abs):.4g} -> guessing {'g/kg' if max_abs > 0.2 else 'kg/kg'}")
    except Exception:
        pass
    if max_abs is np.nan:
        return q
    if max_abs > 0.2:
        return q / 1000.0
    return q


def calculate_physics_variables(p_values, theta_l_values, q_l_values, q_t_values,
                                q_r_values=None):
    """Vectorized temperature, density, and theta_v from theta_l.

    Parameters
    ----------
    p_values : ndarray
        Pressure [Pa]
    theta_l_values : ndarray
        Liquid water potential temperature [K]
    q_l_values : ndarray
        Cloud liquid water mixing ratio [kg/kg] or [g/kg]
    q_t_values : ndarray
        Total (non-precipitating) water mixing ratio [kg/kg] or [g/kg].
        In many LES setups this is q_v + q_l.
    q_r_values : ndarray, optional
        Rain water mixing ratio [kg/kg] or [g/kg]. If not provided, assumed 0.

    Returns
    -------
    T : ndarray
        Temperature [K]
    rho : ndarray
        Air density [kg/m^3]
    theta_v : ndarray
        Virtual potential temperature [K]
    B : ndarray
        Buoyancy relative to planar mean [m/s^2]
    """

    # Convert g/kg -> kg/kg if needed
    q_l = _ensure_kgkg(np.asarray(q_l_values, dtype=float))
    q_t = _ensure_kgkg(np.asarray(q_t_values, dtype=float))
    q_r = _ensure_kgkg(np.asarray(q_r_values, dtype=float)) if q_r_values is not None else 0.0

    # Vapor as remainder. If rain exists, assume q_t = q_v + q_l + q_r.
    q_v = np.clip(q_t - q_l - (q_r if isinstance(q_r, np.ndarray) else q_r), 0.0, None)

    p = np.asarray(p_values, dtype=float)
    theta_l = np.asarray(theta_l_values, dtype=float)

    # moist-adjusted kappa (same as cloudtracker physics)
    kappa = (r_d / c_pd) * ((1.0 + q_v / epsilon) / (1.0 + q_v * (c_pv / c_pd)))

    # temperature from theta_l and p (same as cloudtracker physics)
    T = theta_l * (c_pd / (c_pd - l_v * q_l)) * (p0 / p) ** (-kappa)

    # partial pressure of vapor from mixing ratio (same as cloudtracker physics)
    p_v = (q_v / (q_v + epsilon)) * p

    # density: dry air + vapor + suspended liquid + rain (same as cloudtracker physics)
    rho = (p - p_v) / (r_d * T) + (p_v / (r_v * T))
    # add condensate mass explicitly (liquid + rain) with bulk density of water
    rho = rho + (q_l + (q_r if isinstance(q_r, np.ndarray) else q_r)) * rho_l

    # virtual temperature and theta_v
    q_r_term = q_r if isinstance(q_r, np.ndarray) else q_r
    T_v = T * (1.0 + 0.61 * q_v - q_l - q_r_term)
    theta_v = T_v * (p0 / p) ** (r_d / c_pd)

    # buoyancy about planar mean theta_v (B = g * theta_v'/theta_v_mean)
    theta_v_mean = np.nanmean(theta_v)
    B = g * (theta_v - theta_v_mean) / theta_v_mean

    return T, rho, theta_v, B
