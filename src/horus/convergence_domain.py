from pyoculus.solvers import FixedPoint
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

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