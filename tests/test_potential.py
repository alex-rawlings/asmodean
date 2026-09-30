"""Taking potentials from lanfear: fidelity, black holes, units, persistence.

The external potential is rebuilt inside asmodean from a lanfear potential's
pickled state, so asmodean must evaluate exactly what lanfear evaluates.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import asmodean as am  # noqa: E402

lf = pytest.importorskip("lanfear")
lfc = lf._core


def hernquist_sample(n=50_000, seed=3):
    """Positions of a unit Hernquist sphere (G = M = a = 1)."""
    rng = np.random.default_rng(seed)
    su = np.sqrt(rng.uniform(0, 1, n))
    r = su / (1 - su)
    mu = rng.uniform(-1, 1, n)
    az = rng.uniform(0, 2 * np.pi, n)
    st = np.sqrt(1 - mu**2)
    return np.stack([r * st * np.cos(az), r * st * np.sin(az), r * mu], axis=1)


@pytest.fixture(scope="module")
def scf_core():
    pos = hernquist_sample()
    core = lfc.SCFPotential(8, 4, pos, np.full(len(pos), 1.0 / len(pos)))
    core.add_black_hole(0.01, 0.1, -0.05, 0.02, 1e-3)
    return core


@pytest.fixture(scope="module")
def disc_core():
    core = lfc.DiscPotential(np.array([0.5, 1.0, 2.0]), np.array([0.1, 0.2, 0.3]))
    core.set_coefficients([0.3, 0.5, 0.2])
    return core


def probe_points(n=500, seed=1):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, 3)) * np.exp(rng.uniform(-3, 2, size=(n, 1)))


def assert_same_field(lanfear_core, pot):
    pts = probe_points()
    np.testing.assert_allclose(
        pot.core.potential_batch(pts), lanfear_core.potential_batch(pts), rtol=1e-13
    )
    np.testing.assert_allclose(
        pot.core.acceleration_batch(pts),
        lanfear_core.acceleration_batch(pts),
        rtol=1e-12,
        atol=1e-14,
    )


def test_scf_matches_lanfear(scf_core):
    pot = am.ExternalPotential.from_lanfear_core(scf_core, 1.0, 1.0, G=1.0)
    assert pot.kind == "scf"
    assert pot.n_black_holes == 1
    assert_same_field(scf_core, pot)


def test_disc_matches_lanfear(disc_core):
    pot = am.ExternalPotential.from_lanfear_core(disc_core, 1.0, 1.0, G=1.0)
    assert pot.kind == "disc"
    assert_same_field(disc_core, pot)


def test_composite_matches_lanfear(scf_core, disc_core):
    comp = lfc.CompositePotential(
        [(scf_core, 0.8, 0.5, 0.4, "DM"), (disc_core, 1.5, 0.75, 1.125, "STAR")]
    )
    comp.add_black_hole(0.02, 0.0, 0.0, 0.0, 1e-3)
    pot = am.ExternalPotential.from_lanfear_core(comp, 1.0, 1.0, G=1.0)
    assert pot.kind == "composite"
    assert_same_field(comp, pot)


def test_drop_black_holes(scf_core):
    """include_black_holes=False removes exactly the point-mass term."""
    with_bh = am.ExternalPotential.from_lanfear_core(scf_core, 1.0, 1.0, G=1.0)
    without = am.ExternalPotential.from_lanfear_core(
        scf_core, 1.0, 1.0, G=1.0, include_black_holes=False
    )
    assert without.n_black_holes == 0
    far = np.array([[3.0, 1.0, -2.0]])
    r = np.linalg.norm(far - np.array([0.1, -0.05, 0.02]))
    # Outside its softening the black hole is a Newtonian point mass.
    assert with_bh.potential(far)[0] - without.potential(far)[0] == pytest.approx(
        -0.01 / r, rel=1e-10
    )


def test_physical_units(scf_core):
    """Evaluation converts physical <-> internal units."""
    a, M, G = 2.0, 50.0, am.DEFAULT_G
    pot = am.ExternalPotential.from_lanfear_core(scf_core, a, M, G=G)
    x = np.array([[1.0, 0.5, -0.3]])
    assert pot.potential(x * a)[0] == pytest.approx(
        scf_core.potential(*x[0]) * G * M / a, rel=1e-12
    )
    np.testing.assert_allclose(
        pot.acceleration(x * a)[0],
        np.array(scf_core.acceleration(*x[0])) * G * M / a**2,
        rtol=1e-12,
    )


def test_circular_velocity_hernquist():
    """A monopole SCF fit reproduces the Hernquist circular velocity."""
    pos = hernquist_sample(200_000)
    core = lfc.SCFPotential(10, 0, pos, np.full(len(pos), 1.0 / len(pos)))
    a, M, G = 1.5, 10.0, am.DEFAULT_G
    pot = am.ExternalPotential.from_lanfear_core(core, a, M, G=G)
    r = np.array([0.1, 0.5, 1.0, 3.0]) * a
    v_exact = np.sqrt(G * M * r) / (r + a)
    np.testing.assert_allclose(pot.circular_velocity(r), v_exact, rtol=0.02)
    np.testing.assert_allclose(pot.enclosed_mass(r), M * r**2 / (r + a) ** 2, rtol=0.03)


def test_from_lanfear_potential_object():
    """A full lanfear Potential (with a BH and a pattern speed) is taken as is."""
    rng = np.random.default_rng(0)
    pos = hernquist_sample(20_000) * 2.0
    n = len(pos)
    ps = lf.ParticleSystem(
        pos=np.vstack([pos, [[0.0, 0.0, 0.0]]]),
        vel=np.vstack([rng.normal(size=(n, 3)) * 50, [[0.0, 0.0, 0.0]]]),
        mass=np.concatenate([np.full(n, 1e-3), [0.05]]),
        ids=np.arange(n + 1),
        species=np.array(["STAR"] * n + ["BH"]),
    )
    ps.prepare(centre="field", pattern_speed="none", check_figure_rotation=False)
    lpot = lf.Potential.from_particles(ps, n_max=6, l_max=2)
    lpot.pattern_speed = 5.0  # asmodean ignores it (with a warning)
    pot = am.ExternalPotential.from_lanfear(lpot)
    assert pot.units.length == pytest.approx(lpot.scale_radius)
    assert pot.units.mass == pytest.approx(lpot.field_mass)
    assert pot.n_black_holes == 1
    pts = probe_points(50) * lpot.scale_radius
    np.testing.assert_allclose(
        pot.potential(pts),
        lpot.potential(pts) * pot.units.specific_energy,
        rtol=1e-12,
    )
    assert am.ExternalPotential.from_lanfear(lpot, False).n_black_holes == 0


def test_save_load_roundtrip(tmp_path, scf_core, disc_core):
    comp = lfc.CompositePotential(
        [(scf_core, 0.8, 0.5, 0.4, "DM"), (disc_core, 1.5, 0.75, 1.125, "STAR")]
    )
    for core in (scf_core, disc_core, comp):
        pot = am.ExternalPotential.from_lanfear_core(core, 1.3, 7.0)
        path = pot.save(tmp_path / f"{pot.kind}.npz")
        back = am.ExternalPotential.load(path)
        assert back.kind == pot.kind
        assert back.units == pot.units
        pts = probe_points(100)
        np.testing.assert_array_equal(
            back.core.acceleration_batch(pts), pot.core.acceleration_batch(pts)
        )
    none = am.ExternalPotential.none()
    assert am.ExternalPotential.load(none.save(tmp_path / "none.npz")).is_null


def test_orbit_in_potential_conserves_energy(scf_core):
    """A single body orbiting in the (aspherical) SCF field conserves energy."""
    pot = am.ExternalPotential.from_lanfear_core(
        scf_core, 1.0, 10.0, include_black_holes=False
    )
    system = am.BodySystem()
    v_c = pot.circular_velocity(1.0)[0]
    system.add_body(0.01, [1.0, 0.0, 0.0], [0.0, 0.8 * v_c, 0.3 * v_c])
    res = am.integrate(system, pot, n_periods=20, period="circular")
    assert res.stop_reason == "time_limit"
    assert np.max(np.abs(res.energy_error())) < 1e-9
    assert res.period == pytest.approx(2 * np.pi * 1.0 / v_c)
