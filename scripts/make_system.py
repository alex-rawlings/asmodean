"""Construct a scattering experiment from a TOML configuration and save it.

Builds the external potential (with lanfear, from a Gadget snapshot or a
synthetic Hernquist sphere, or loads a saved one), the bodies, the dynamical
friction model and the integration settings, and writes them to one ``.npz``
setup file for ``run_scattering.py``. See ``example_config.toml`` for every
option.

    python scripts/make_system.py scripts/example_config.toml -o setup.npz
"""

import argparse
import tomllib

import numpy as np
import asmodean as am


def hernquist_potential(cfg, G):
    """lanfear SCF fit to a random sample of a Hernquist sphere.

    Parameters
    ----------
    cfg : dict
        The ``[potential]`` table (``mass``, ``scale_radius``, ``n_particles``,
        ``n_max``, ``l_max``, ``seed``).
    G : float
        Gravitational constant.

    Returns
    -------
    potential : asmodean.ExternalPotential
        The potential.
    """
    from lanfear import _core as lanfear_core

    n = int(cfg.get("n_particles", 200_000))
    rng = np.random.default_rng(cfg.get("seed", 0))
    # Invert M(<r)/M = r^2 / (1 + r)^2 in scale-radius units.
    su = np.sqrt(rng.uniform(0, 1, n))
    r = su / (1 - su)
    mu = rng.uniform(-1, 1, n)
    az = rng.uniform(0, 2 * np.pi, n)
    st = np.sqrt(1 - mu**2)
    pos = np.stack([r * st * np.cos(az), r * st * np.sin(az), r * mu], axis=1)
    core = lanfear_core.SCFPotential(
        int(cfg.get("n_max", 12)), int(cfg.get("l_max", 0)), pos, np.full(n, 1.0 / n)
    )
    return am.ExternalPotential.from_lanfear_core(
        core,
        scale_radius=float(cfg["scale_radius"]),
        field_mass=float(cfg["mass"]),
        G=G,
    )


def lanfear_potential(cfg, G):
    """Build a lanfear potential from a Gadget snapshot.

    Parameters
    ----------
    cfg : dict
        The ``[potential]`` table (``snapshot``, ``basis``, basis options,
        ``centre``, ``bh_softening``, ``include_black_holes``, ``validate``).
    G : float
        Gravitational constant.

    Returns
    -------
    potential : asmodean.ExternalPotential
        The potential.
    particles : lanfear.ParticleSystem
        The prepared snapshot (for bodies/friction built from it).
    """
    import lanfear as lf

    ps = lf.ParticleSystem.from_gadget_hdf5(cfg["snapshot"])
    # asmodean's potential is always static: no figure rotation.
    ps.prepare(centre=cfg.get("centre", "bh"), pattern_speed="none")
    basis = cfg.get("basis", "scf")
    common = {"bh_softening": float(cfg.get("bh_softening", 1e-3)), "G": G}
    if basis == "scf":
        pot = lf.Potential.from_particles(
            ps, n_max=int(cfg["n_max"]), l_max=int(cfg["l_max"]), **common
        )
    elif basis == "disc":
        pot = lf.DiscPotential.from_particles(
            ps,
            n_radial=int(cfg.get("n_radial", 8)),
            n_vert=int(cfg.get("n_vert", 3)),
            **common,
        )
    elif basis == "multi":
        components = {}
        for species, spec in cfg["components"].items():
            spec = dict(spec)
            kind = spec.pop("basis")
            maker = {"scf": lf.scf_component, "disc": lf.disc_component}[kind]
            components[species] = maker(**spec)
        pot = lf.MultiComponentPotential.from_particles(ps, components, **common)
    else:
        raise ValueError(f"unknown basis '{basis}' (expected scf, disc or multi)")
    if cfg.get("validate", False):
        print(f"lanfear potential validation: {pot.validate()}")
    potential = am.ExternalPotential.from_lanfear(
        pot, include_black_holes=cfg.get("include_black_holes", True)
    )
    return potential, ps


def build_potential(cfg, G):
    """The external potential described by the ``[potential]`` table.

    Parameters
    ----------
    cfg : dict
        The ``[potential]`` table.
    G : float
        Gravitational constant.

    Returns
    -------
    potential : asmodean.ExternalPotential or None
        The potential (None for ``type = "none"``).
    particles : lanfear.ParticleSystem or None
        The prepared snapshot, for ``type = "lanfear"``.
    """
    kind = cfg.get("type", "none")
    if kind == "none":
        return None, None
    if kind == "file":
        return am.ExternalPotential.load(cfg["file"]), None
    if kind == "hernquist":
        return hernquist_potential(cfg, G), None
    if kind == "lanfear":
        return lanfear_potential(cfg, G)
    raise ValueError(f"unknown potential type '{kind}'")


def orientation(entry):
    """Rotation matrix from ``orientation_deg = [inclination, node, argument]``.

    Parameters
    ----------
    entry : dict
        A body table.

    Returns
    -------
    rotation : numpy.ndarray or None
        (3, 3) rotation matrix, or None if no orientation is given.
    """
    if "orientation_deg" not in entry:
        return None
    inc, node, arg = np.radians(entry["orientation_deg"])
    return am.rotation_matrix(inc, node, arg)


def build_bodies(cfg, G, particles):
    """The bodies described by the configuration.

    Parameters
    ----------
    cfg : dict
        The whole configuration.
    G : float
        Gravitational constant.
    particles : lanfear.ParticleSystem or None
        The prepared snapshot (for ``bodies_from_black_holes``).

    Returns
    -------
    system : asmodean.BodySystem
        The bodies.
    """
    pcfg = cfg.get("potential", {})
    if pcfg.get("bodies_from_black_holes", False):
        if particles is None:
            raise ValueError("bodies_from_black_holes needs a lanfear snapshot")
        system = am.BodySystem.from_particles(
            particles.black_holes,
            softening=float(pcfg.get("bh_body_softening", 0.0)),
            G=G,
        )
    else:
        system = am.BodySystem(G=G)

    for b in cfg.get("body", []):
        system.add_body(
            b["mass"],
            b["position"],
            b["velocity"],
            softening=b.get("softening", 0.0),
            label=b.get("label"),
        )
    for b in cfg.get("binary", []):
        system.add_binary(
            b["masses"],
            b["semimajor_axis"],
            eccentricity=b.get("eccentricity", 0.0),
            inclination=np.radians(b.get("inclination_deg", 0.0)),
            longitude_of_ascending_node=np.radians(b.get("node_deg", 0.0)),
            argument_of_pericentre=np.radians(b.get("argument_deg", 0.0)),
            true_anomaly=np.radians(b.get("true_anomaly_deg", 0.0)),
            centre_position=b.get("centre_position", (0.0, 0.0, 0.0)),
            centre_velocity=b.get("centre_velocity", (0.0, 0.0, 0.0)),
            softening=b.get("softening", 0.0),
            labels=b.get("labels"),
        )
    for b in cfg.get("incoming", []):
        system.add_incoming_body(
            b["mass"],
            b["v_infinity"],
            b["impact_parameter"],
            b["distance"],
            target=b.get("target"),
            rotation=orientation(b),
            softening=b.get("softening", 0.0),
            label=b.get("label"),
        )
    if system.n_bodies == 0:
        raise ValueError("the configuration defines no bodies")
    return system


def build_friction(cfg, potential, particles, system):
    """The dynamical friction described by the ``[friction]`` table.

    Parameters
    ----------
    cfg : dict
        The ``[friction]`` table.
    potential : asmodean.ExternalPotential or None
        The potential (for ``source = "potential"``).
    particles : lanfear.ParticleSystem or None
        The prepared snapshot (for ``source = "particles"``).
    system : asmodean.BodySystem
        The bodies (for ``central_mass = "bodies"``).

    Returns
    -------
    friction : asmodean.DynamicalFriction or None
        The friction model, or None if disabled.
    """
    if not cfg.get("enabled", False):
        return None
    options = {
        "coulomb_logarithm": cfg.get("coulomb_logarithm", 3.0),
        "mode": cfg.get("mode", "system"),
    }
    source = cfg.get("source", "potential")
    if source == "potential":
        if potential is None:
            raise ValueError("friction source 'potential' needs a potential")
        central_mass = cfg.get("central_mass", 0.0)
        if central_mass == "bodies":
            central_mass = system.total_mass
        return am.DynamicalFriction.from_potential(
            potential,
            r_min=float(cfg["r_min"]),
            r_max=float(cfg["r_max"]),
            n_bins=int(cfg.get("n_bins", 64)),
            central_mass=float(central_mass),
            **options,
        )
    if source == "particles":
        if particles is None:
            raise ValueError("friction source 'particles' needs a lanfear snapshot")
        return am.DynamicalFriction.from_particle_system(
            particles,
            n_bins=int(cfg.get("n_bins", 50)),
            r_min=cfg.get("r_min"),
            r_max=cfg.get("r_max"),
            **options,
        )
    if source == "constant":
        return am.DynamicalFriction.constant(
            cfg["density"], cfg["dispersion"], **options
        )
    raise ValueError(f"unknown friction source '{source}'")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(type=str, help="TOML configuration file", dest="config")
    ap.add_argument("-o", "--output", type=str, default="setup.npz", help="setup file")
    ap.add_argument("-v", "--verbose", action="store_true", help="INFO logging")
    args = ap.parse_args()
    if args.verbose:
        am.set_verbosity("INFO")

    with open(args.config, "rb") as f:
        cfg = tomllib.load(f)
    G = float(cfg.get("units", {}).get("G", am.DEFAULT_G))

    potential, particles = build_potential(cfg.get("potential", {}), G)
    system = build_bodies(cfg, G, particles)
    friction = build_friction(cfg.get("friction", {}), potential, particles, system)
    settings = am.IntegrationSettings(**cfg.get("integration", {}))

    setup = am.ScatteringSetup(system, potential, friction, settings)
    path = setup.save(args.output)
    print(f"{system.n_bodies} bodies: {', '.join(system.labels)}")
    print(f"potential: {'none' if potential is None else potential.kind}")
    print(f"friction: {'off' if friction is None else friction.mode}")
    print(
        f"reference period: {am.estimate_period(system, potential, settings.period):.6g}"
    )
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
