from pyoculus.solvers import FixedPoint # type: ignore
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

### Convergence domain for the X-O point finders ###


def join_convergence_domains(convdomlist, eps=1e-5):
    """Join together a list of convergence domain results, returning a new tuple with the same format."""
    convdomReturn = ([], [], [])

    for convdom in convdomlist:
        R_values, Z_values, _, _, all_fixed_points = convdom
        convdomReturn[0].append(R_values)
        convdomReturn[1].append(Z_values)
        convdomReturn[2].append(all_fixed_points)

    R_values = np.concatenate(convdomReturn[0])
    Z_values = np.concatenate(convdomReturn[1])
    all_fixed_points = np.concatenate(convdomReturn[2])

    # Get the indices that would sort R_values and Z_values
    R_sort_indices = np.argsort(R_values)
    Z_sort_indices = np.argsort(Z_values)

    # Use these indices to rearrange all_fixed_points
    all_fixed_points = all_fixed_points[R_sort_indices][Z_sort_indices]

    # # Reloop through the sorted arrays and assign the correct index to each point

    # for fp in all_fixed_points:
    #         fp_result = all_fixed_points[i]

    #         if fp_result.successful is True:
    #             fp_result_xyz = np.array([fp_result.x[0], fp_result.y[0], fp_result.z[0]])
    #             assigned = False
    #             for j, fpt in enumerate(fixed_points):
    #                 fpt_xyz = np.array([fpt.x[0], fpt.y[0], fpt.z[0]])
    #                 if np.isclose(fp_result_xyz, fpt_xyz, atol=options["eps"]).all():
    #                     assigned_to.append(j)
    #                     assigned = True
    #             if not assigned:
    #                 assigned_to.append(len(fixed_points))
    #                 fixed_points.append(fp_result)
    #             all_fixed_points.append(fp_result)
    #         else:
    #             assigned_to.append(-1)
    #             all_fixed_points.append(None)

    # return R_values, Z_values, assigned_to, all_fixed_points


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
    iparams = {"rtol": 1e-13}
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
            all_fixed_points.append(fp_result)

    return np.array([R, Z, np.array(assigned_to).reshape(R.shape), np.array(all_fixed_points).reshape(R.shape)]), np.array(fixed_points, dtype=object)


# def plot_convergence_domain(convdom, ax=None, colors=None):
#     return plot_convergence_domain(*convdom[0:4], ax=ax, colors=colors)


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

    assigned_to = assigned_to.flatten() + 1
    assigned_to = assigned_to.astype(int)

    if colors is None:
        colors = cm.rainbow(np.linspace(0, 1, len(fixed_points)))
        colors[:, 3] = 0.8
        colors = np.vstack(([0.3, 0.3, 0.3, 0.15], colors))
        
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
        if hasattr(fpt, "GreenesResidue"):
            if fpt.GreenesResidue < 0:
                marker = "X"
            elif fpt.GreenesResidue > 0:
                marker = "o"
            else:
                marker = "s"
        else:
            marker = "s"

        ax.scatter(
            fpt.x[0],
            fpt.z[0],
            color=colors[i + 1, :3],
            marker=marker,
            edgecolors="black",
            linewidths=1,
            label=f"[{fpt.x[0]:.2f},{ fpt.z[0]:.2f}]",
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