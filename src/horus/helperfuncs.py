import numpy as np
from scipy.integrate import solve_ivp
from pyoculus.problems import CylindricalBfield # type: ignore
from simsopt.field import MagneticField # type: ignore


def normalize(v: np.ndarray) -> np.ndarray:
    """Compute the normalized vector of v."""
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


### Magnetic field line tracing using solve_ip ###


def trace(bobject, tf, xx, **kwargs):
    if isinstance(bobject, MagneticField):

        def unit_Bfield(t, xx):
            bobject.set_points(xx.reshape((-1, 3)))
            return normalize(bobject.B().flatten())

    elif isinstance(bobject, CylindricalBfield):
        kwargs["is_cylindrical"] = True

        def unit_Bfield(t, xx):
            return bobject.f_RZ(t, [xx[0], xx[1]])

    else:
        raise ValueError("bobject must be a MagneticField or a CylindricalBfield")
    return _trace(unit_Bfield, tf, xx, **kwargs)


def _trace(unit_Bfield, tf, xx, **kwargs):
    """Compute the curve of the magnetic field line in 3d in the forward direction from the initial point xx
    using scipy to solve the Initial Value Problem.

    Args:
        bs (simsopt.MagneticField): Magnetic Field object, for instance simsopt.BiotSavart
        tw (tuple): time window (tmax, nsteps)
        xx (np.ndarray): initial point in 3d cartesian coordinates
        t_eval (np.ndarray): time points to evaluate the solution,
            if None, use np.linspace(0, tf, steps)

    Returns:
        np.ndarray: the trace of the magnetic field line
    """
    options = {
        "rtol": 1e-10,
        "atol": 1e-10,
        "t_eval": None,
        "steps": int(tf * 1000),
        "method": "DOP853",
        "is_cylindrical": False,
    }
    options.update(kwargs)

    if options["t_eval"] is None:
        options["t_eval"] = np.linspace(0, tf, options["steps"])

    out = solve_ivp(
        unit_Bfield,
        [0, tf],
        xx,
        t_eval=options["t_eval"],
        method=options["method"],
        rtol=options["rtol"],
        atol=options["atol"],
    )

    if options["is_cylindrical"]:
        gamma = np.array(
            [
                [r * np.cos(phi), r * np.sin(phi), z]
                for r, phi, z in zip(out.y[0], out.t, out.y[1])
            ]
        ).T
    else:
        gamma = out.y

    return gamma, out
