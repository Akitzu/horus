import numpy as np
import matplotlib.pyplot as plt
plt.style.use('lateky')
from scipy.integrate import solve_ivp
from simsopt.field import ( # type: ignore
    compute_fieldlines,
    LevelsetStoppingCriterion,
)
import pickle

### Drawing of a Poincare section ###


class PoincarePlanes():

    def plot(self, phis, **kwargs):
        return plot_poincare_data(self.phi_hits, phis, **kwargs)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump((self.tys, self.phi_hits), f)

    @classmethod
    def from_ivp(cls, out):
        instance = cls()
        instance.out = out
        return instance

    @classmethod    
    def from_simsopt(cls, fieldlines_tys, fieldlines_phi_hits):
        instance = cls()
        instance.tys = fieldlines_tys
        instance.phi_hits = fieldlines_phi_hits
        return instance

    @classmethod
    def from_record(cls, record):
        instance = cls()
        instance.record = record
        return instance

    # @property
    # def hits(self):
    #     if hasattr(self, "out"):
    #         return self.out
    #     elif hasattr(self, "phi_hits"):
    #         return self.phi_hits
    #     elif hasattr(self, "record"):
    #         return self.record

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

    # if plot:
    #     fig, ax = pplane.plot()
    #     return fieldlines_tys, fieldlines_phi_hits, fig, ax

    return pplane


def poincare_simsopt(bs, RZstart, phis, sc_fieldline, **kwargs):
    options = {"tmax": 40000, "tol": 1e-9, "comm": None}
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
                            method=options["method"],
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


### REAL ploting functions

def plot_poincare_pyoculus(xydata, ax, xlims = [3.5, 9.2], ylims = [-6, 2.5], **kwargs):
    options = {
        "color": "black",
        "s": 1,
        "linewidths": 1,
        "zorder": 10,
        "marker": ".",
    }
    options.update(kwargs)

    rdata, zdata = xydata
    for rs, zs in zip(rdata, zdata):
        ax.scatter(rs, zs, **options)

    if xlims is not None:
        ax.set_xlim(xlims)
    if ylims is not None:
        ax.set_ylim(ylims)

    ax.set_xlabel(r"R", fontsize=16)
    ax.set_ylabel(r"Z", fontsize=16)
    ax.set_aspect("equal")
    return ax.get_figure(), ax

def plot_poincare_simsopt(fieldlines_phi_hits, ax, idx=None, **kwargs):
    options = {
        "color": "black",
        "s": 1,
        "linewidths": 0,
        "zorder": 10,
        "marker": ".",
    }
    options.update(kwargs)

    for j in range(len(fieldlines_phi_hits)):
        if idx is None:
            where = np.where(fieldlines_phi_hits[j][:, 1] >= 0)[0]
        else:
            where = np.where(fieldlines_phi_hits[j][:, 1] == idx)[0]

        data_this_phi = fieldlines_phi_hits[j][
            where, :
        ]
        if data_this_phi.size == 0:
            continue
        r = np.sqrt(data_this_phi[:, 2] ** 2 + data_this_phi[:, 3] ** 2)
        ax.scatter(
            r, data_this_phi[:, 4], **options
        )
    
    ax.set_xlabel(r"R [m]")
    ax.set_ylabel(r"Z [m]")
    ax.set_aspect("equal")
    return ax.get_figure(), ax