import numpy as np
from scipy.integrate import solve_ivp
from pyoculus.problems import CylindricalBfield # type: ignore
from simsopt.field import MagneticField # type: ignore
from pathlib import Path
import matplotlib.pyplot as plt
plt.style.use('lateky')
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np

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

def plot_q_profile(r, q, ax, r_shift):
    r = r + r_shift
    ax.plot(r, q, marker=".", linestyle="-", color="black")
    # ax.set_xlabel(r"Minor radius $\rho$")
    ax.set_ylabel(r"Safety factor $q$", fontsize=16)
    return ax.get_figure(), ax

def plot_iota_profile(r, iota, ax, r_shift):
    r = r + r_shift
    ax.plot(r, iota, marker=".", linestyle="-", color="black")
    ax.set_xlabel(r"Minor radius $\rho$", fontsize=16)
    ax.set_ylabel(r"Rotationnal transform $\iota/2\pi$", fontsize=16)
    return ax.get_figure(), ax

def plot_iota_q(r, iota, q,  bbox = None, r_shift=-6):
    if bbox is None:
        bbox = (.15, .07, .4, .3)

    fig, ax = plt.subplots()
    # bbox = (.55, .6, .4, .35)
   
    plot_iota_profile(r, iota, ax, r_shift=args.r_shift)
    axins = inset_axes(ax, width="100%", height="100%", 
                       bbox_to_anchor=bbox,
                       bbox_transform=ax.transAxes, loc=3)
    plot_q_profile(r, q, axins, r_shift=r_shift)
    return fig, ax

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Compute the q/iota plot from the poincare data.")
#     parser.add_argument("folder", help="Folder containing the poincare data.", default="squared-profile")
#     # parser.add_argument("--bb", type=tuple, default=(.1, .07, .4, .3), help="bbox_to_anchor for the inset axes.")
#     parser.add_argument("--r_shift", type=float, default=-6, help="Shift the minor radius.")
#     args = parser.parse_args()

#     folder = Path(args.folder)
#     r = np.loadtxt(folder / "r-squared.txt")
#     q = np.loadtxt(folder / "q-squared.txt")
#     iota = np.loadtxt(folder / "iota-squared.txt")
