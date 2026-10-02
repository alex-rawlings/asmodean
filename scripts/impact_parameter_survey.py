"""Survey of SMBH binary formation against impact parameter, from a snapshot.

Reads a Gadget-4 HDF5 snapshot, builds a static lanfear SCF potential from all
the field particles or from the stars alone, and takes the two black holes'
phase-space coordinates as the initial conditions of a black hole pair. The
pair is then integrated (in parallel) for a range of impact parameters, with
dynamical friction, until it becomes a hard binary; the eccentricity it has at
that moment is plotted against the two-body deflection angle
``2 arctan(b90 / b)``, with ``b90 = G (m_1 + m_2) / v^2`` for the pair's
relative speed ``v`` at the snapshot.

The impact parameter is that of the pair's relative orbit at the snapshot
time, ``b = |r x v| / |v|`` for the relative position ``r`` and velocity ``v``.
It is set by rotating ``v`` within the orbital plane (keeping ``|v|`` and an
approaching radial velocity), so the positions, the speeds, the energy and the
centre-of-mass motion are exactly those of the snapshot; ``b`` must be below
the separation ``|r|``. Only impact parameters whose deflection angle reaches
``--min-deflection`` (default 30 degrees), i.e. ``b <= b90 / tan(theta_min /
2)``, are integrated; the ``--n-b`` samples are spread over that part of
``--b-range``.

The binary is hard once its semimajor axis is ``a <= G mu / (4 sigma^2)`` (see
``asmodean.IntegrationSettings``). By default ``sigma`` is the 1D velocity
dispersion of the field particles (the same ones as the potential) within the
black holes' influence radius, which encloses twice their mass.

    python scripts/impact_parameter_survey.py snap_030.hdf5 -o survey --potential-from stars
    python scripts/impact_parameter_survey.py --plot-only survey/survey.npz

Writes, in the output directory: ``survey.npz`` (the impact parameters and the
outcome of every run), ``setup.npz`` (the potential, friction, settings and the
unperturbed black holes, for ``run_scattering.py``), ``runs/run_XXX.npz`` (each
:class:`asmodean.ScatteringResult`) and the figure
``impact_parameter_eccentricity.png``.
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import asmodean as am

_SURVEY_FORMAT = "asmodean-impact-survey-1"
# Reference categorical palette (see the plotting conventions in the docs).
_HARD_COLOUR = "#2a78d6"
_OTHER_COLOUR = "#eb6834"


def dispersion_arg(value):
    """Parse --hard-dispersion: a velocity, 'influence' or 'friction'."""
    return value if value in ("influence", "friction") else float(value)


def coulomb_arg(value):
    """Parse --coulomb-logarithm: a number or 'variable'."""
    return value if value == "variable" else float(value)


def load_snapshot(path, potential_from, centre):
    """Read and prepare a snapshot, and pick the particles the potential uses.

    Parameters
    ----------
    path : str
        Gadget-4 HDF5 snapshot.
    potential_from : {"all", "stars"}
        Build the potential from every field particle, or from the stars only.
    centre : str
        Centring passed to ``lanfear.ParticleSystem.prepare``.

    Returns
    -------
    particles : lanfear.ParticleSystem
        The prepared snapshot (all species).
    source : lanfear.ParticleSystem
        The particles the potential is built from, plus the black holes.
    """
    import lanfear as lf

    particles = lf.ParticleSystem.from_gadget_hdf5(path)
    # Centre and align on the whole snapshot; asmodean's figure never rotates.
    particles.prepare(centre=centre, pattern_speed="none")
    if potential_from == "all":
        return particles, particles
    mask = particles.species_mask("STAR", "BH")
    if not np.any(particles.species == "STAR"):
        raise ValueError(f"{path} holds no star particles")
    source = particles.select(mask)
    source.scale_radius = particles.scale_radius
    return particles, source


def pick_black_holes(particles, ids=None):
    """The two black holes forming the pair.

    Parameters
    ----------
    particles : lanfear.ParticleSystem
        The prepared snapshot.
    ids : sequence of int, optional
        Particle IDs of the two black holes; default the two most massive.

    Returns
    -------
    black_holes : lanfear.ParticleSystem
        The two black holes, more massive first.

    Raises
    ------
    ValueError
        If the snapshot does not hold the requested black holes.
    """
    bhs = particles.black_holes
    if ids is None:
        if bhs.n_particles < 2:
            raise ValueError(
                f"the snapshot holds {bhs.n_particles} black hole(s), need two"
            )
        chosen = np.argsort(bhs.mass)[::-1][:2]
    else:
        chosen = [int(np.flatnonzero(bhs.ids == i)[0]) for i in ids if i in bhs.ids]
        if len(chosen) != 2:
            raise ValueError(f"black hole IDs {ids} not all found in the snapshot")
        chosen = sorted(chosen, key=lambda k: -bhs.mass[k])
    mask = np.zeros(bhs.n_particles, dtype=bool)
    mask[chosen] = True
    pair = bhs.select(mask)
    return pair.select(np.argsort(pair.mass)[::-1])


def build_potential(source, n_max, l_max, G):
    """lanfear SCF potential of the field particles, black holes left out.

    Parameters
    ----------
    source : lanfear.ParticleSystem
        The particles (black holes are dropped from the potential, as they are
        integrated as bodies).
    n_max, l_max : int
        SCF truncation orders.
    G : float
        Gravitational constant.

    Returns
    -------
    potential : asmodean.ExternalPotential
        The potential.
    """
    import lanfear as lf

    pot = lf.Potential.from_particles(source, n_max=n_max, l_max=l_max, G=G)
    return am.ExternalPotential.from_lanfear(pot, include_black_holes=False)


def influence_dispersion(source, black_holes):
    """1D velocity dispersion of the field within the black holes' influence radius.

    Parameters
    ----------
    source : lanfear.ParticleSystem
        The particles the potential was built from.
    black_holes : lanfear.ParticleSystem
        The pair.

    Returns
    -------
    sigma : float
        Mass-weighted 1D dispersion (physical velocity).
    r_infl : float
        The influence radius (enclosing twice the pair's mass).
    """
    from lanfear.binary import influence_radius

    m_pair = float(np.sum(black_holes.mass))
    centre = np.average(black_holes.pos, weights=black_holes.mass, axis=0)
    r_infl = influence_radius(source, m_pair, centre=centre)
    field = source.field
    inside = np.linalg.norm(field.pos - centre, axis=1) <= r_infl
    w = field.mass[inside]
    v = field.vel[inside]
    dv = v - np.average(v, weights=w, axis=0)
    sigma = float(np.sqrt(np.average(np.sum(dv**2, axis=1), weights=w) / 3.0))
    return sigma, r_infl


def impact_parameter(positions, velocities):
    """Impact parameter of a pair's relative orbit, ``|r x v| / |v|``.

    Parameters
    ----------
    positions, velocities : numpy.ndarray
        (2, 3) positions and velocities of the pair.

    Returns
    -------
    b : float
        The impact parameter.
    """
    r = positions[0] - positions[1]
    v = velocities[0] - velocities[1]
    return float(np.linalg.norm(np.cross(r, v)) / np.linalg.norm(v))


def aim_pair(masses, positions, velocities, b):
    """Rotate a pair's relative velocity to give it impact parameter ``b``.

    The relative velocity keeps its magnitude and stays in the original
    orbital plane, with an approaching (negative) radial component; positions
    and the centre-of-mass velocity are unchanged.

    Parameters
    ----------
    masses : numpy.ndarray
        (2,) masses.
    positions, velocities : numpy.ndarray
        (2, 3) positions and velocities.
    b : float
        The impact parameter, ``0 <= b < |r|``.

    Returns
    -------
    velocities : numpy.ndarray
        (2, 3) new velocities.

    Raises
    ------
    ValueError
        If ``b`` is not below the separation.
    """
    r = positions[0] - positions[1]
    v = velocities[0] - velocities[1]
    sep, speed = np.linalg.norm(r), np.linalg.norm(v)
    if not (0 <= b < sep):
        raise ValueError(f"impact parameter {b:.4g} must be in [0, {sep:.4g})")
    r_hat = r / sep
    tangent = v - np.dot(v, r_hat) * r_hat
    if np.linalg.norm(tangent) < 1e-12 * speed:  # radial orbit: any perpendicular
        trial = np.eye(3)[np.argmin(np.abs(r_hat))]
        tangent = trial - np.dot(trial, r_hat) * r_hat
    t_hat = tangent / np.linalg.norm(tangent)
    sin_theta = b / sep
    v_new = speed * (-np.sqrt(1.0 - sin_theta**2) * r_hat + sin_theta * t_hat)
    m1, m2 = masses
    v_com = (m1 * velocities[0] + m2 * velocities[1]) / (m1 + m2)
    return np.stack([v_com + m2 / (m1 + m2) * v_new, v_com - m1 / (m1 + m2) * v_new])


def survey_arrays(results, impact_parameters):
    """Outcome of every run as arrays.

    Parameters
    ----------
    results : list of asmodean.ScatteringResult or None
        One result per impact parameter.
    impact_parameters : numpy.ndarray
        The impact parameters.

    Returns
    -------
    arrays : dict of str to numpy.ndarray
        ``stop_reason``, ``t_end``, ``hard_time``, ``hard_semimajor_axis`` and
        ``hard_eccentricity`` (NaN where the binary did not become hard).
    """
    n = len(impact_parameters)
    out = {
        "stop_reason": np.full(n, "integration_error", dtype="U32"),
        "t_end": np.full(n, np.nan),
        "hard_time": np.full(n, np.nan),
        "hard_semimajor_axis": np.full(n, np.nan),
        "hard_eccentricity": np.full(n, np.nan),
    }
    for k, res in enumerate(results):
        if res is None:
            continue
        out["stop_reason"][k] = res.stop_reason
        out["t_end"][k] = res.t_end
        ev = res.hard_binary
        if ev is not None:
            out["hard_time"][k] = ev.time
            out["hard_semimajor_axis"][k] = ev.value
            out["hard_eccentricity"][k] = ev.eccentricity
    return out


def deflection_angle(b, b90):
    """Two-body deflection angle ``2 arctan(b90 / b)``, in degrees.

    Parameters
    ----------
    b : float or array-like of float
        Impact parameter (180 degrees at ``b = 0``).
    b90 : float
        Impact parameter of a 90-degree deflection, ``G (m_1 + m_2) / v^2``.

    Returns
    -------
    theta : float or numpy.ndarray
        The deflection angle (degrees).
    """
    return np.degrees(2.0 * np.arctan2(b90, np.asarray(b, dtype=np.float64)))


def select_impact_parameters(sep, b90, b_range, n_b, min_deflection, explicit=None):
    """Impact parameters whose deflection angle reaches a threshold.

    ``2 arctan(b90 / b) >= min_deflection`` is ``b <= b90 / tan(min_deflection
    / 2)``. Explicit impact parameters beyond that are dropped; otherwise
    ``n_b`` values are spread uniformly over the part of ``b_range`` (in units
    of the separation) below it.

    Parameters
    ----------
    sep : float
        The pair's initial separation.
    b90 : float
        Impact parameter of a 90-degree deflection.
    b_range : sequence of float
        ``(min, max)`` impact parameter in units of ``sep``.
    n_b : int
        Number of impact parameters.
    min_deflection : float
        Smallest deflection angle (degrees, in (0, 180]).
    explicit : sequence of float, optional
        Explicit impact parameters (physical length) to filter instead.

    Returns
    -------
    impact_parameters : numpy.ndarray
        The selected impact parameters.

    Raises
    ------
    ValueError
        If the threshold is invalid or leaves no impact parameter.
    """
    if not (0 < min_deflection <= 180):
        raise ValueError(f"min_deflection must be in (0, 180], got {min_deflection}")
    b_limit = b90 / np.tan(np.radians(min_deflection) / 2)
    if explicit is not None:
        bs = np.asarray(explicit, dtype=np.float64)
        keep = bs <= b_limit
        if np.any(~keep):
            print(
                f"Dropping {np.sum(~keep)} impact parameter(s) beyond {b_limit:.4g} "
                f"(deflection below {min_deflection:g} deg)"
            )
        bs = bs[keep]
    else:
        lo, hi = sep * b_range[0], min(sep * b_range[1], b_limit)
        if lo > hi:
            raise ValueError(
                f"--b-range starts at b={lo:.4g}, beyond b={b_limit:.4g} where the "
                f"deflection falls below {min_deflection:g} deg"
            )
        bs = np.linspace(lo, hi, n_b)
    if bs.size == 0:
        raise ValueError(f"no impact parameter deflects by {min_deflection:g} deg")
    return bs


def survey_b90(survey):
    """``b90 = G (m_1 + m_2) / v^2`` of a survey, from the snapshot relative speed.

    Parameters
    ----------
    survey : mapping of str to numpy.ndarray
        Contents of ``survey.npz``.

    Returns
    -------
    b90 : float
        The 90-degree deflection impact parameter.
    """
    if "b90" in survey:
        return float(survey["b90"])
    G = float(survey["G"]) if "G" in survey else am.DEFAULT_G
    m = float(np.sum(survey["masses"]))
    return G * m / float(survey["relative_speed"]) ** 2


def plot_survey(survey, path):
    """Eccentricity at hardening against two-body deflection angle.

    The deflection angle is ``2 arctan(b90 / b)``, with
    ``b90 = G (m_1 + m_2) / v^2`` for the pair's relative speed ``v`` at the
    snapshot (the same in every run). The top axis gives the corresponding
    impact parameter in units of the initial separation. Runs whose binary did
    not become hard are marked along the bottom axis.

    Parameters
    ----------
    survey : mapping of str to numpy.ndarray
        Contents of ``survey.npz``.
    path : str
        Output figure path.
    """
    b90 = survey_b90(survey)
    sep = float(survey["separation"])
    theta = deflection_angle(survey["impact_parameter"], b90)
    ecc = np.asarray(survey["hard_eccentricity"])
    hard = np.isfinite(ecc)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(
        theta[hard],
        ecc[hard],
        ls="-",
        lw=1.0,
        color=_HARD_COLOUR,
        alpha=0.5,
        zorder=1,
    )
    ax.scatter(
        theta[hard],
        ecc[hard],
        s=36,
        color=_HARD_COLOUR,
        edgecolor="white",
        linewidth=1.0,
        zorder=2,
        label=f"became hard ({np.sum(hard)})",
    )
    if np.any(~hard):
        ax.scatter(
            theta[~hard],
            np.full(np.sum(~hard), -0.04),
            s=36,
            marker="x",
            color=_OTHER_COLOUR,
            zorder=2,
            label=f"did not become hard ({np.sum(~hard)})",
            clip_on=False,
        )
    ax.set_ylim(-0.06, 1.02)
    ax.set_xlabel(r"deflection angle $2\arctan(b_{90}/b)$ [deg]")
    ax.set_ylabel("eccentricity when hard")
    ax.grid(alpha=0.25, lw=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    def to_b(theta):
        with np.errstate(divide="ignore", invalid="ignore"):
            return b90 / (sep * np.tan(np.radians(np.clip(theta, 1e-6, 180)) / 2))

    def to_theta(x):
        return deflection_angle(np.asarray(x) * sep, b90)

    top = ax.secondary_xaxis("top", functions=(to_b, to_theta))
    top.set_xlabel("impact parameter / initial separation")
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Wrote {path}")


def parse_args():
    """Command-line arguments.

    Returns
    -------
    args : argparse.Namespace
        The arguments.
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(type=str, nargs="?", help="Gadget-4 HDF5 snapshot", dest="snapshot")
    ap.add_argument("-o", "--output", type=str, default="survey", help="output dir")
    ap.add_argument(
        "--plot-only",
        type=str,
        metavar="SURVEY",
        help="only re-plot an existing survey.npz",
    )
    g = ap.add_argument_group("potential")
    g.add_argument(
        "--potential-from",
        choices=["all", "stars"],
        default="all",
        help="build the potential from every field particle or the stars only",
    )
    g.add_argument("--n-max", type=int, default=18, help="SCF radial order")
    g.add_argument("--l-max", type=int, default=6, help="SCF angular order")
    g.add_argument("--centre", type=str, default="bh", help="lanfear centring")
    g.add_argument(
        "-G", type=float, default=am.DEFAULT_G, help="gravitational constant"
    )
    g = ap.add_argument_group("black holes")
    g.add_argument(
        "--bh-ids", type=int, nargs=2, help="IDs of the pair (default: most massive)"
    )
    g.add_argument(
        "--bh-softening",
        type=float,
        default=1e-5,
        help="Plummer-equivalent softening of the black holes",
    )
    g.add_argument(
        "--collision-distance", type=float, help="merge the pair closer than this"
    )
    g = ap.add_argument_group("impact parameters")
    g.add_argument(
        "--impact-parameters",
        type=float,
        nargs="+",
        help="explicit impact parameters (physical length)",
    )
    g.add_argument(
        "--b-range",
        type=float,
        nargs=2,
        default=[0.0, 0.9],
        metavar=("MIN", "MAX"),
        help="range in units of the initial separation (default 0 0.9)",
    )
    g.add_argument(
        "--n-b",
        type=int,
        default=16,
        help="number of impact parameters, spread over the part of --b-range "
        "allowed by --min-deflection",
    )
    g.add_argument(
        "--min-deflection",
        type=float,
        default=30.0,
        help="only impact parameters whose deflection angle 2 arctan(b90 / b) is "
        "at least this many degrees (default 30)",
    )
    g = ap.add_argument_group("dynamical friction and hardening")
    g.add_argument(
        "--friction-mode",
        choices=["individual", "system"],
        default="individual",
        help="'individual' drags each black hole, so the pair can sink and bind",
    )
    g.add_argument(
        "--coulomb-logarithm",
        type=coulomb_arg,
        default=3.0,
        help="constant Coulomb logarithm, or 'variable'",
    )
    g.add_argument(
        "--friction-range",
        type=float,
        nargs=2,
        metavar=("R_MIN", "R_MAX"),
        help="radii of the friction profile (default 1e-4 and 100 scale radii)",
    )
    g.add_argument(
        "--hard-dispersion",
        type=dispersion_arg,
        default="influence",
        help="sigma in a_h = G mu / (4 sigma^2): a velocity, 'influence' "
        "(field within the influence radius; default) or 'friction' (local, "
        "from the friction profile)",
    )
    g = ap.add_argument_group("integration")
    g.add_argument(
        "--n-periods",
        type=float,
        default=1000.0,
        help="time limit in circular periods at the pair's initial radius",
    )
    g.add_argument("--n-samples", type=int, default=2000, help="output samples")
    g.add_argument(
        "--no-trajectories", action="store_true", help="keep only first/last states"
    )
    g.add_argument("--rel-tol", type=float, default=1e-10, help="relative tolerance")
    g.add_argument("--progress", action="store_true", help="print progress")
    ap.add_argument("-v", "--verbose", action="store_true", help="INFO logging")
    args = ap.parse_args()
    if args.plot_only is None and args.snapshot is None:
        ap.error("a snapshot is needed unless --plot-only is given")
    return args


def main():
    args = parse_args()
    am.print_package_info()
    if args.verbose:
        am.set_verbosity("INFO")
    if args.plot_only is not None:
        with np.load(args.plot_only, allow_pickle=False) as survey:
            out = os.path.join(
                os.path.dirname(args.plot_only), "impact_parameter_eccentricity.png"
            )
            plot_survey(survey, out)
        return

    os.makedirs(os.path.join(args.output, "runs"), exist_ok=True)
    particles, source = load_snapshot(args.snapshot, args.potential_from, args.centre)
    pair = pick_black_holes(particles, args.bh_ids)
    potential = build_potential(source, args.n_max, args.l_max, args.G)

    labels = [f"BH{i}" for i in pair.ids]
    base = am.BodySystem.from_particles(
        pair, softening=args.bh_softening, labels=labels, G=args.G
    )
    masses, positions = base.masses, base.positions
    sep = float(np.linalg.norm(positions[0] - positions[1]))
    b_snapshot = impact_parameter(positions, base.velocities)
    v_rel = float(np.linalg.norm(base.velocities[0] - base.velocities[1]))
    print(
        f"pair {labels[0]}-{labels[1]}: masses {masses[0]:.4g}, {masses[1]:.4g}; "
        f"separation {sep:.4g}, relative speed {v_rel:.4g}, snapshot impact "
        f"parameter {b_snapshot:.4g}"
    )

    r_min, r_max = args.friction_range or (
        1e-4 * particles.scale_radius,
        100.0 * particles.scale_radius,
    )
    friction = am.DynamicalFriction.from_potential(
        potential,
        r_min=r_min,
        r_max=r_max,
        central_mass=float(np.sum(masses)),
        coulomb_logarithm=args.coulomb_logarithm,
        mode=args.friction_mode,
    )

    sigma = args.hard_dispersion
    if sigma == "influence":
        sigma, r_infl = influence_dispersion(source, pair)
        print(f"influence radius {r_infl:.4g}, field dispersion within it {sigma:.4g}")
    if sigma != "friction":
        a_h = am.hard_binary_semimajor_axis(masses[0], masses[1], sigma, args.G)
        print(f"hard-binary semimajor axis a_h = {float(a_h):.4g}")

    b90 = args.G * float(np.sum(masses)) / v_rel**2
    print(
        f"b90 = G (m1 + m2) / v^2 = {b90:.4g}; deflection >= "
        f"{args.min_deflection:g} deg needs b <= "
        f"{b90 / np.tan(np.radians(args.min_deflection) / 2):.4g}"
    )
    bs = select_impact_parameters(
        sep,
        b90,
        args.b_range,
        args.n_b,
        args.min_deflection,
        args.impact_parameters,
    )
    systems = []
    for b in bs:
        velocities = aim_pair(masses, positions, base.velocities, b)
        systems.append(
            am.BodySystem.from_arrays(
                masses, positions, velocities, args.bh_softening, labels, args.G
            )
        )

    settings = am.IntegrationSettings(
        n_periods=args.n_periods,
        period="circular",
        n_samples=args.n_samples,
        collision_distance=args.collision_distance,
        rel_tol=args.rel_tol,
        record_trajectory=not args.no_trajectories,
        hard_binary_dispersion=sigma,
    )
    am.ScatteringSetup(base, potential, friction, settings).save(
        os.path.join(args.output, "setup.npz")
    )
    print(
        f"Integrating {len(bs)} impact parameters in [{bs.min():.4g}, {bs.max():.4g}]"
    )
    results = am.integrate_ensemble(
        systems, potential, settings, friction, progress=args.progress
    )
    for k, res in enumerate(results):
        if res is not None:
            res.save(os.path.join(args.output, "runs", f"run_{k:03d}.npz"))

    survey = {
        "_format": np.asarray(_SURVEY_FORMAT),
        "impact_parameter": bs,
        "separation": np.asarray(sep),
        "relative_speed": np.asarray(v_rel),
        "b90": np.asarray(b90),
        "min_deflection": np.asarray(args.min_deflection),
        "G": np.asarray(args.G),
        "snapshot_impact_parameter": np.asarray(b_snapshot),
        "masses": masses,
        "hard_dispersion": np.asarray(np.nan if sigma == "friction" else sigma),
        "snapshot": np.asarray(os.path.abspath(args.snapshot)),
        "potential_from": np.asarray(args.potential_from),
        "friction_mode": np.asarray(args.friction_mode),
    }
    survey.update(survey_arrays(results, bs))
    path = os.path.join(args.output, "survey.npz")
    np.savez(path, **survey)
    print(f"Wrote {path}")
    for b, reason, e in zip(bs, survey["stop_reason"], survey["hard_eccentricity"]):
        print(
            f"  b={b:.4g} (deflection {deflection_angle(b, b90):.1f} deg): {reason}"
            + (f", e={e:.3f}" if np.isfinite(e) else "")
        )
    plot_survey(survey, os.path.join(args.output, "impact_parameter_eccentricity.png"))


if __name__ == "__main__":
    main()
