from simsopt.configs import get_ncsx_data, get_w7x_data # type: ignore
from simsopt.field import ( # type: ignore
    BiotSavart,
    InterpolatedField,
    coils_via_symmetries,
    SurfaceClassifier,
)
from simsopt.geo import SurfaceRZFourier # type: ignore
import numpy as np

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
