import numpy as np
from multiprocessing import Pool
from scipy.integrate import solve_ivp
from scipy.optimize import toms748
from simsopt.configs import get_ncsx_data, get_w7x_data
from simsopt.field import (
    MagneticField,
    BiotSavart,
    InterpolatedField,
    coils_via_symmetries,
    SurfaceClassifier,
    compute_fieldlines,
    LevelsetStoppingCriterion,
)
from simsopt.geo import SurfaceRZFourier

from pyoculus.problems import CartesianBfield, CylindricalBfield
from pyoculus.solvers import FixedPoint, PoincarePlot

import matplotlib.pyplot as plt
from matplotlib import cm

### Helper functions for Horus ###

# def cyltocart(r, theta, z):
#     return np.array(
#         [
#             [np.cos(theta), -np.sin(theta), 0],
#             [-r * np.sin(theta), -r * np.cos(theta), 0],
#             [0, 0, 1],
#         ]
#     )


# def carttocyl(x, y, z):
#     r = np.sqrt(x**2 + y**2)
#     theta = np.arctan2(y, x)
#     return np.linalg.inv(cyltocart(r, theta, z))


def normalize(v: np.ndarray) -> np.ndarray:
    """Compute the normalized vector of v."""
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


### Simsopt magnetic field problem class ###
class SimsoptBfieldProblem(CartesianBfield):
    def __init__(self, R0, Z0, Nfp, bs):
        super().__init__(R0, Z0, Nfp)

        if not isinstance(bs, MagneticField):
            raise ValueError("bs must be a MagneticField object")

        self.bs = bs

    # The return of the B field for the two following methods is not the same as the calls are :
    #   - CartesianBfield.f_RZ which does :
    #   line 37     B = np.array([self.B(xyz, *args)]).T
    #   - CartesianBfield.f_RZ_tangent which does :
    #   line 68     B, dBdX = self.dBdX(xyz, *args)
    #   line 69     B = np.array(B).T
    # and both should result in a (3,1) array
    def B(self, xyz):
        xyz = np.reshape(xyz, (-1, 3))
        self.bs.set_points(xyz)
        return self.bs.B().flatten()

    def dBdX(self, xyz):
        B = self.B(xyz)
        return [B], self.bs.dB_by_dX().reshape(3, 3)

    def B_many(self, x1arr, x2arr, x3arr, input1D=True):
        if input1D:
            xyz = np.array([x1arr, x2arr, x3arr], dtype=np.float64).T
        else:
            xyz = np.meshgrid(x1arr, x2arr, x3arr)
            xyz = np.array(
                [xyz[0].flatten(), xyz[1].flatten(), xyz[2].flatten()], dtype=np.float64
            ).T

        xyz = np.ascontiguousarray(xyz, dtype=np.float64)
        self.bs.set_points(xyz)

        return self.bs.B()

    def dBdX_many(self, x1arr, x2arr, x3arr, input1D=True):
        B = self.B_many(x1arr, x2arr, x3arr, input1D=input1D)
        return [B], self.bs.dB_by_dX()


### Stellerator configurations ###


def ncsx():
    """Get the NCSX stellarator configuration."""
    return stellarator(*get_ncsx_data(), nfp=3)


def w7x():
    """Get the W7-X stellarator configuration."""
    return stellarator(*get_w7x_data(), nfp=5, surface_radius=2)


def stellarator(curves, currents, ma, nfp, **kwargs):
    """Set up a stellarator configuration and returns the magnetic field and the interpolated magnetic field as well as coils and the magnetic axis.

    Args:
        curves (simsopt.CurveXYZFourier): list of curves
        currents (list of simsopt.Current): list of currents
        ma (simsopt.CurveRZFourier): magnetic axis
        nfp (int): number of field periods

    Keyword Args:
        degree (int): degree of the interpolating polynomial
        n (int): number of points in the radial direction
        mpol (int): number of poloidal modes
        ntor (int): number of toroidal modes
        stellsym (bool): whether to exploit stellarator symmetry
        surface_radius (float): radius of the surface

    Returns:
        tuple: (Biot-Savart object, InterpolatedField object, (nfp, coils, ma, sc_fieldline))
    """
    options = {
        "degree": 2,
        "surface_radius": 0.7,
        "n": 20,
        "mpol": 5,
        "ntor": 5,
        "stellsym": True,
    }
    options.update(kwargs)

    # Load the NCSX data and create the coils
    coils = coils_via_symmetries(curves, currents, nfp, True)
    curves = [c.curve for c in coils]

    # Create the Biot-Savart object
    bs = BiotSavart(coils)

    # Create the surface and the surface classifier
    s = SurfaceRZFourier.from_nphi_ntheta(
        mpol=options["mpol"],
        ntor=options["ntor"],
        stellsym=options["stellsym"],
        nfp=nfp,
        range="full torus",
        nphi=64,
        ntheta=24,
    )
    s.fit_to_curve(ma, options["surface_radius"], flip_theta=False)
    sc_fieldline = SurfaceClassifier(s, h=0.03, p=2)

    # Bounds for the interpolated magnetic field chosen so that the surface is
    # entirely contained in it
    rs = np.linalg.norm(s.gamma()[:, :, 0:2], axis=2)
    zs = s.gamma()[:, :, 2]
    rrange = (np.min(rs), np.max(rs), options["n"])
    phirange = (0, 2 * np.pi / nfp, options["n"] * 2)
    # exploit stellarator symmetry and only consider positive z values:
    zrange = (0, np.max(zs), options["n"] // 2)

    def skip(rs, phis, zs):
        # The RegularGrindInterpolant3D class allows us to specify a function that
        # is used in order to figure out which cells to be skipped.  Internally,
        # the class will evaluate this function on the nodes of the regular mesh,
        # and if *all* of the eight corners are outside the domain, then the cell
        # is skipped.  Since the surface may be curved in a way that for some
        # cells, all mesh nodes are outside the surface, but the surface still
        # intersects with a cell, we need to have a bit of buffer in the signed
        # distance (essentially blowing up the surface a bit), to avoid ignoring
        # cells that shouldn't be ignored
        rphiz = np.asarray([rs, phis, zs]).T.copy()
        dists = sc_fieldline.evaluate_rphiz(rphiz)
        skip = list((dists < -0.05).flatten())
        return skip

    bsh = InterpolatedField(
        bs,
        options["degree"],
        rrange,
        phirange,
        zrange,
        True,
        nfp=nfp,
        stellsym=True,
        skip=skip,
    )

    return bs, bsh, (nfp, coils, ma, sc_fieldline)


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
        gamma = np.array([[r*np.cos(phi), r*np.sin(phi), z] for r, phi, z in zip(out.y[0], out.t, out.y[1])]).T
    else:
        gamma = out.y

    return gamma, out


### Drawing of a Poincare section ###

class PoincarePlanes():

    @classmethod
    def from_ivp(cls, out):
        cls.out = out

    @classmethod
    def from_simsopt(cls, fieldlines_tys, fieldlines_phi_hits):
        cls.tys = fieldlines_tys
        cls.phi_hits = fieldlines_phi_hits

    @classmethod
    def from_record(cls, record):
        cls.record = record

    @property
    def hits(self):
        if hasattr(self, "out"):
            return self.out
        elif hasattr(self, "phi_hits"):
            return self.phi_hits
        elif hasattr(self, "record"):
            return self.record


def plot_poincare_data(
    fieldlines_phi_hits,
    phis,
    filename=None,
    mark_lost=False,
    aspect="equal",
    dpi=300,
    xlims=None,
    ylims=None,
    surf=None,
    s=2,
    marker="o",
):
    """
    Create a poincare plot. Usage:

    .. code-block::

        phis = np.linspace(0, 2*np.pi/nfp, nphis, endpoint=False)
        res_tys, res_phi_hits = compute_fieldlines(
            bsh, R0, Z0, tmax=1000, phis=phis, stopping_criteria=[])
        plot_poincare_data(res_phi_hits, phis, '/tmp/fieldlines.png')

    Requires matplotlib to be installed.

    """
    from math import ceil, sqrt

    nrowcol = ceil(sqrt(len(phis)))
    plt.figure()
    fig, axs = plt.subplots(nrowcol, nrowcol, figsize=(8, 5))
    if len(phis) == 1:
        axs = np.array([[axs]])
    for ax in axs.ravel():
        ax.set_aspect(aspect)
    color = None
    for i in range(len(phis)):
        row = i // nrowcol
        col = i % nrowcol
        if i != len(phis) - 1:
            axs[row, col].set_title(
                f"$\\phi = {phis[i]/np.pi:.2f}\\pi$ ", loc="left", y=0.0
            )
        else:
            axs[row, col].set_title(
                f"$\\phi = {phis[i]/np.pi:.2f}\\pi$ ", loc="right", y=0.0
            )
        if row == nrowcol - 1:
            axs[row, col].set_xlabel("$r$")
        if col == 0:
            axs[row, col].set_ylabel("$z$")
        if col == 1:
            axs[row, col].set_yticklabels([])
        if xlims is not None:
            axs[row, col].set_xlim(xlims)
        if ylims is not None:
            axs[row, col].set_ylim(ylims)
        for j in range(len(fieldlines_phi_hits)):
            lost = fieldlines_phi_hits[j][-1, 1] < 0
            if mark_lost:
                color = "r" if lost else "g"
            data_this_phi = fieldlines_phi_hits[j][
                np.where(fieldlines_phi_hits[j][:, 1] == i)[0], :
            ]
            if data_this_phi.size == 0:
                continue
            r = np.sqrt(data_this_phi[:, 2] ** 2 + data_this_phi[:, 3] ** 2)
            axs[row, col].scatter(
                r, data_this_phi[:, 4], marker=marker, s=s, linewidths=0, c=color
            )

        plt.rc("axes", axisbelow=True)
        axs[row, col].grid(True, linewidth=0.5)

        # if passed a surface, plot the plasma surface outline
        if surf is not None:
            cross_section = surf.cross_section(phi=phis[i])
            r_interp = np.sqrt(cross_section[:, 0] ** 2 + cross_section[:, 1] ** 2)
            z_interp = cross_section[:, 2]
            axs[row, col].plot(r_interp, z_interp, linewidth=1, c="k")

    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, dpi=dpi)
    return fig, axs


def poincare(
    bs, RZstart, phis, sc_fieldline=None, engine="simsopt", plot=True, **kwargs
):
    if engine == "simsopt":
        fieldlines_tys, fieldlines_phi_hits = poincare_simsopt(
            bs, RZstart, phis, sc_fieldline, **kwargs
        )
        pplane = PoincarePlanes.from_simsopt(fieldlines_tys, fieldlines_phi_hits)
    elif engine == "scipy-2d":
        out = poincare_ivp_2d(bs, RZstart, phis, **kwargs)
        pplane = PoincarePlanes.from_ivp(out)
    elif engine == "scipy":
        record = poincare_ivp(bs, RZstart, phis, **kwargs)
        pplane = PoincarePlanes.from_record(record)

    if plot:
        fig, ax = pplane.plot()
        return fieldlines_tys, fieldlines_phi_hits, fig, ax

    return pplane


def poincare_simsopt(bs, RZstart, phis, sc_fieldline, **kwargs):
    options = {"tmax": 40000, "tol": 1e-7, "comm": None}
    options.update(kwargs)

    fieldlines_tys, fieldlines_phi_hits = compute_fieldlines(
        bs,
        RZstart[:, 0],
        RZstart[:, 1],
        tmax=options["tmax"],
        tol=options["tol"],
        comm=options["comm"],
        phis=phis,
        stopping_criteria=[LevelsetStoppingCriterion(sc_fieldline.dist)],
    )

    return fieldlines_tys, fieldlines_phi_hits


def poincare_ivp(bs, RZstart, phis, **kwargs):
    options = {
        "rtol": 1e-10,
        "atol": 1e-10,
        "t_eval": None,
        "tmax": 100,
        "method": "DOP853",
        "eps": 1e-1,
    }
    options.update(kwargs)

    # Recording function for the crossing of the planes
    record = list()
    last_dist = []
    last_t = 0

    def record_crossing(t, xyz):
        current_phis = np.arctan2(xyz[1::3], xyz[::3])

        msh_Phi, msh_Plane = np.meshgrid(current_phis, phis)
        dist = msh_Phi - msh_Plane

        if len(last_dist) != 0:
            switch = np.logical_and(
                np.sign(last_dist) != np.sign(dist), np.abs(dist) < options["eps"]
            )
            for i, s in enumerate(switch):
                for j, ss in enumerate(s):
                    if ss:

                        def crossing(_, xyz):
                            return np.arctan2(xyz[1], xyz[0]) - phis[i]
                        
                        crossing.terminal = True

                        def minusbfield(_, xyz):
                            return -bs.B(xyz[::3], xyz[1::3], xyz[2::3]).flatten()

                        out = solve_ivp(
                            minusbfield,
                            [0, t - last_t],
                            [xyz[3 * j], xyz[3 * j + 1], xyz[3 * j + 2]],
                            events=crossing,
                            method=options["method"]
                        )
                        record.append(
                            [
                                j,
                                phis[i],
                                t - out.t_events[0][0],
                                out.y_events[0].flatten(),
                            ]
                        )

        last_dist = dist
        last_t = t

    # Define the Bfield function that uses a MagneticField from simsopt
    def Bfield(t, xyz, recording=True):
        if recording:
            record_crossing(t, xyz)
        bs.set_points(xyz.reshape((-1, 3)))
        return bs.B().flatten()

    # Putting (R0Z) coordinates to (xyz) for integration
    if RZstart.shape[1] != 3:
        RZstart = np.vstack(
            (RZstart[:, 0], np.zeros((RZstart.shape[0])), RZstart[:, 1])
        ).T

    # Integrate the field lines
    solve_ivp(
        Bfield,
        [0, options["tmax"]],
        RZstart.flatten(),
        t_eval=[],
        method=options["method"],
        rtol=options["rtol"],
        atol=options["atol"],
    )

    return record


def inv_Jacobian(R, phi, _):
    return np.array(
        [
            [np.cos(phi), np.sin(phi), 0],
            [-np.sin(phi) / R, np.cos(phi) / R, 0],
            [0, 0, 1],
        ]
    )


def poincare_ivp_2d(bs, RZstart, phis, **kwargs):
    options = {
        "rtol": 1e-7,
        "atol": 1e-8,
        "nintersect": 10,
        "method": "DOP853",
        "nfp": 1,
        "mpol": 1,
    }
    options.update(kwargs)

    def Bfield_2D(t, rzs):
        rzs = rzs.reshape((-1, 2))
        rphizs = np.ascontiguousarray(
            np.vstack(
                (rzs[:, 0], (t % (2 * np.pi)) * np.ones(rzs.shape[0]), rzs[:, 1])
            ).T
        )
        bs.set_points_cyl(rphizs)
        bs_Bs = bs.B()

        Bs = list()
        for position, B in zip(rphizs, bs_Bs):
            B = inv_Jacobian(*position) @ B.reshape(3, -1)
            Bs.append(np.array([B[0, 0] / B[1, 0], B[2, 0] / B[1, 0]]))

        return np.array(Bs).flatten()

    # setup the phis of the poincare sections
    phis = np.unique(np.mod(phis, 2 * np.pi / options["nfp"]))
    phis.sort()

    # setup the evaluation points for those sections
    phi_evals = np.array(
        [
            phis + options["mpol"] * 2 * np.pi * i / options["nfp"]
            for i in range(options["nintersect"] + 1)
        ]
    )

    # print(phi_evals[-1,-1])
    out = solve_ivp(
        Bfield_2D,
        [0, phi_evals[-1, -1]],
        RZstart.flatten(),
        t_eval=phi_evals.flatten(),
        method=options["method"],
        atol=options["atol"],
        rtol=options["rtol"],
    )

    return out


### Convergence domain for the X-O point finders ###


def join_convergence_domains(convdomA, convdomB, eps=1e-4):
    """Join two convergence domain results, returning a new tuple with the same format."""
    assignedB = convdomB[2].copy()
    fplistA = convdomA[3].copy()

    for i, fp in enumerate(convdomB[3]):
        fp_xyz = np.array([fp.x[0], fp.y[0], fp.z[0]])
        found_prev = False
        for j, fp_prev in enumerate(convdomA[3]):
            fp_prev_xyz = np.array([fp_prev.x[0], fp_prev.y[0], fp_prev.z[0]])
            if np.isclose(fp_xyz, fp_prev_xyz, atol=eps).all():
                assignedB[assignedB == j] = i
                found_prev = True
                break
        if not found_prev:
            assignedB[assignedB == i] = len(fplistA)
            fplistA.append(fp)

    return (
        np.concatenate((convdomA[0], convdomB[0])),
        np.concatenate((convdomA[1], convdomB[1])),
        np.concatenate((convdomA[2], assignedB)),
        fplistA,
    )


def convergence_domain(ps, Rw, Zw, **kwargs):
    """Compute where the FixedPoint solver converge to in the R-Z plane. Each point from the meshgrid given by the input Rw and Zw is tested for convergence.
    if a point converges, it is assigned a number, otherwise it is assigned -1. The number corresponds to the index of the fixed point in returned list of fixed points.

    Args:
        ps (pyoculus.problems.CartesianBfield): the problem to solve
        Rw (np.ndarray): the R values of the meshgrid
        Zw (np.ndarray): the Z values of the meshgrid

    Keyword Args:
        -- FixedPoint.compute --
        pp (int): the poloidal mode to use
        qq (int): the toroidal mode to use
        sbegin (float): the starting value of the R parameter
        send (float): the ending value of the R parameter
        tol (float): the tolerance of the fixed point finder
        checkonly (bool): whether to use checkonly theta for the Newton RZ\n
        -- Integrator --
        rtol (float): the relative tolerance of the integrator\n
        --- FixedPoint ---
        nrestart (int): the number of restarts for the fixed point finder
        niter (int): the number of iterations for the fixed point finder\n
        -- Comparison --
        eps (float): the tolerance for the comparison with the fixed points

    Returns:
        np.ndarray: the R values of the meshgrid
        np.ndarray: the Z values of the meshgrid
        np.ndarray: the assigned number for each point in the meshgrid
        list: the list of fixed points object (BaseSolver.OutputData)
    """
    options = {
        "pp": 3,
        "qq": 7,
        "sbegin": 1.2,
        "send": 1.9,
        "tol": 1e-4,
        "checkonly": True,
        "eps": 1e-4,
    }
    options.update(kwargs)

    # set up the integrator
    iparams = {"rtol": 1e-7}
    iparams.update(kwargs)

    # set up the point finder
    pparams = {"nrestart": 0, "niter": 30}
    pparams.update(kwargs)

    R, Z = np.meshgrid(Rw, Zw)

    assigned_to = list()
    fixed_points = list()
    all_fixed_points = list()

    for r, z in zip(R.flatten(), Z.flatten()):
        fp_result = FixedPoint(ps, pparams, integrator_params=iparams)
        fp_result.compute(
            guess=[r, z],
            pp=options["pp"],
            qq=options["qq"],
            sbegin=options["sbegin"],
            send=options["send"],
            tol=options["tol"],
            checkonly=options["checkonly"],
        )

        if fp_result.successful is True:
            fp_result_xyz = np.array([fp_result.x[0], fp_result.y[0], fp_result.z[0]])
            assigned = False
            for j, fpt in enumerate(fixed_points):
                fpt_xyz = np.array([fpt.x[0], fpt.y[0], fpt.z[0]])
                if np.isclose(fp_result_xyz, fpt_xyz, atol=options["eps"]).all():
                    assigned_to.append(j)
                    assigned = True
            if not assigned:
                assigned_to.append(len(fixed_points))
                fixed_points.append(fp_result)
            all_fixed_points.append(fp_result)
        else:
            assigned_to.append(-1)
            all_fixed_points.append(None)

    return R, Z, np.array(assigned_to), fixed_points, all_fixed_points

def plot_convergence_domain(convdom, ax=None, colors=None):
    return plot_convergence_domain(*convdom[0:4], ax=ax, colors=colors)

def plot_convergence_domain(R, Z, assigned_to, fixed_points, ax=None, colors=None):
    """Plot the convergence domain for FixedPoint solver in the R-Z plane. If ax is None, a new figure is created,
    otherwise the plot is added to the existing figure.

    Args:
        R (np.ndarray): the R values of the meshgrid
        Z (np.ndarray): the Z values of the meshgrid
        assigned_to (np.ndarray): the assigned number for each point in the meshgrid
        fixed_points (list): the list of fixed points object (BaseSolver.OutputData)\n
        -- Optional --
        ax (matplotlib.axes.Axes): the axes to plot on. Defaults to None.
        colors (np.ndarray): the colors to use. Defaults to COLORS. Should be of dimension (k, 3 or 4) for RGB/RGBA with k at least the number of fixed point plus one.

    Returns:
        tuple: (fig, ax)
    """

    assigned_to = assigned_to + 1

    if colors is None:
        colors = cm.rainbow(np.linspace(0, 1, len(fixed_points) + 1))
        colors[:, 3] = 0.8
        colors = np.vstack(([0.5, 0.5, 0.5, 0.5], colors))

    cmap = np.array([colors[j] for j in assigned_to])
    cmap = cmap.reshape(R.shape[0], R.shape[1], cmap.shape[1])

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    ax.pcolormesh(R, Z, cmap, shading="nearest")

    # for r,z in zip(R, Z):
    #     ax.scatter(r, z, color = 'blue', s = 1)

    for i, fpt in enumerate(fixed_points):
        if fpt.GreenesResidue < 0:
            marker = "X"
        elif fpt.GreenesResidue > 0:
            marker = "o"
        else:
            marker = "s"

        ax.scatter(
            fpt.x[0],
            fpt.z[0],
            color=colors[i + 1, :3],
            marker=marker,
            edgecolors="black",
            linewidths=1,
        )

    # # Plot arrows from the meshgrid points to the fixed points they converge to
    # for r, z, a in zip(R.flat, Z.flat, assigned_to.flat):
    #     if a > 0:
    #         fpt = fixed_points[a - 1]
    #         dr = np.array([fpt.x[0] - r, fpt.z[0] - z])
    #         dr = 0.1*dr
    #         ax.arrow(r, z, dr[0], dr[1], color='blue')

    ax.set_aspect("equal")

    return fig, ax
