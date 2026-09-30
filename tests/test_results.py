"""Set-up helpers, persistence, ensembles and outcome analysis."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import asmodean as am  # noqa: E402


# ------------------------------------------------------------------ set-up
def test_kepler_roundtrip():
    """Orbital elements -> Cartesian -> orbital elements."""
    rng = np.random.default_rng(4)
    for _ in range(20):
        a, e = rng.uniform(0.1, 10), rng.uniform(0, 0.95)
        inc, node, arg, nu = rng.uniform(0, np.pi), *rng.uniform(0, 2 * np.pi, 3)
        mu = rng.uniform(0.5, 5)
        pos, vel = am.kepler_to_cartesian(mu, a, e, inc, node, arg, nu)
        el = am.orbital_elements(mu, 0.0, np.zeros(3), pos, np.zeros(3), vel, G=1.0)
        assert el.bound
        assert el.semimajor_axis == pytest.approx(a, rel=1e-10)
        assert el.eccentricity == pytest.approx(e, abs=1e-10)
        assert el.inclination == pytest.approx(inc, abs=1e-8)


def test_add_binary_centre_of_mass():
    system = am.BodySystem(G=1.0)
    system.add_binary(
        [2.0, 1.0],
        semimajor_axis=1.0,
        eccentricity=0.3,
        centre_position=[5.0, 0.0, 1.0],
        centre_velocity=[0.0, 1.0, 0.0],
    )
    x, v = system.centre_of_mass()
    np.testing.assert_allclose(x, [5.0, 0.0, 1.0], atol=1e-14)
    np.testing.assert_allclose(v, [0.0, 1.0, 0.0], atol=1e-14)


def test_add_incoming_body():
    """The incoming body is on the requested hyperbola; the CoM is unchanged."""
    system = am.BodySystem(G=1.0)
    system.add_binary([1.0, 1.0], semimajor_axis=0.1, labels=["A", "B"])
    x0, v0 = system.centre_of_mass()
    rot = am.rotation_matrix(0.3, 1.0, 2.0)
    i = system.add_incoming_body(
        0.5,
        v_infinity=2.0,
        impact_parameter=0.4,
        distance=20.0,
        rotation=rot,
        label="C",
    )
    x_t, v_t = system.centre_of_mass(["A", "B"])
    el = am.orbital_elements(
        2.0, 0.5, x_t, system.positions[i], v_t, system.velocities[i], G=1.0
    )
    assert not el.bound
    assert el.energy == pytest.approx(0.5 * 2.0**2, rel=1e-12)
    rel_x, rel_v = system.positions[i] - x_t, system.velocities[i] - v_t
    assert np.linalg.norm(np.cross(rel_x, rel_v)) == pytest.approx(0.4 * 2.0, rel=1e-12)
    assert np.dot(rel_x, rel_v) < 0  # approaching
    x1, v1 = system.centre_of_mass()
    np.testing.assert_allclose(x1, x0, atol=1e-13)
    np.testing.assert_allclose(v1, v0, atol=1e-13)


def test_period_estimates():
    system = am.BodySystem(G=2.0)
    system.add_binary([1.0, 1.0], semimajor_axis=2.0)
    assert am.estimate_period(system, method="binary") == pytest.approx(
        2 * np.pi * np.sqrt(8.0 / 4.0)
    )
    R = system.rms_radius()
    assert am.estimate_period(system, method="internal") == pytest.approx(
        2 * np.pi * np.sqrt(R**3 / 4.0)
    )
    assert am.estimate_period(system, method=3.5) == 3.5
    with pytest.raises(ValueError):
        am.estimate_period(system, method="circular")


# ------------------------------------------------------------------ persistence
def three_body():
    system = am.BodySystem(G=1.0)
    system.add_binary(
        [1.0, 1.0], semimajor_axis=1.0, eccentricity=0.5, labels=["A", "B"]
    )
    system.add_incoming_body(
        0.5, v_infinity=0.3, impact_parameter=0.5, distance=15.0, label="C"
    )
    return system


def test_result_roundtrip(tmp_path):
    res = am.integrate(
        three_body(),
        n_periods=300,
        period="binary",
        n_samples=500,
        ejection_radius=20.0,
        collision_distance=1e-3,
    )
    assert res.events  # something happened
    back = am.ScatteringResult.load(res.save(tmp_path / "res.npz"))
    for name in ("times", "positions", "velocities", "masses", "energy", "merged_into"):
        np.testing.assert_array_equal(getattr(back, name), getattr(res, name))
    assert back.events[0].kind == res.events[0].kind
    np.testing.assert_array_equal(back.events[0].position, res.events[0].position)
    assert back.status == res.status
    assert back.stop_reason == res.stop_reason
    assert back.units == res.units
    assert back.metadata["settings.ejection_radius"] == 20.0
    assert "remaining" in back.summary()


def test_setup_roundtrip(tmp_path):
    settings = am.IntegrationSettings(n_periods=7, period=2.5, ejection_radius=3.0)
    friction = am.DynamicalFriction.constant(1.0, 2.0, coulomb_logarithm="variable")
    setup = am.ScatteringSetup(three_body(), None, friction, settings)
    back = am.ScatteringSetup.load(setup.save(tmp_path / "setup.npz"))
    assert back.settings == settings
    assert back.potential is None
    assert back.friction.coulomb_logarithm == "variable"
    np.testing.assert_array_equal(back.system.positions, setup.system.positions)
    assert back.system.labels == ["A", "B", "C"]
    settings = am.IntegrationSettings(period="binary")
    assert am.IntegrationSettings.from_arrays(settings.to_arrays()) == settings


def test_ensemble_matches_single():
    systems = []
    for b in (0.3, 0.6, 0.9):
        system = am.BodySystem(G=1.0)
        system.add_binary([1.0, 1.0], semimajor_axis=1.0, labels=["A", "B"])
        system.add_incoming_body(0.5, 0.5, b, 15.0, label="C")
        systems.append(system)
    kw = dict(n_periods=50, period="binary", ejection_radius=20.0, n_samples=100)
    batch = am.integrate_ensemble(systems, **kw)
    for system, res in zip(systems, batch):
        single = am.integrate(system, **kw)
        np.testing.assert_array_equal(res.positions, single.positions)
        assert res.stop_reason == single.stop_reason
    # Without a trajectory, steps are no longer shortened to hit output times,
    # so compare with a single run made the same way (a chaotic three-body
    # system amplifies any change in the step sequence).
    light = am.integrate_ensemble(systems, record_trajectory=False, **kw)
    assert all(r.n_samples == 2 for r in light)
    single = am.integrate(systems[0], record_trajectory=False, **kw)
    np.testing.assert_array_equal(light[0].positions, single.positions)


# ------------------------------------------------------------------ analysis
def test_classify_flyby():
    """A distant, fast flyby leaves the binary intact."""
    system = am.BodySystem(G=1.0)
    system.add_binary([1.0, 1.0], semimajor_axis=0.1, labels=["A", "B"])
    system.add_incoming_body(0.1, 5.0, 3.0, 30.0, label="C")
    res = am.integrate(system, n_periods=1, period=40.0, ejection_radius=35.0)
    out = am.classify_outcome(res)
    assert out.kind == "flyby"
    assert out.resolved
    assert out.ejected == [2]
    assert out.final_pair.bodies == (0, 1)
    assert "flyby" in out.describe(res.labels)


def test_classify_merger_and_unbound():
    system = am.BodySystem(G=1.0)
    system.add_body(1.0, [0, 0, 0], [0, 0, 0])
    system.add_body(1.0, [1, 0, 0], [0, 0, 0])
    res = am.integrate(system, period=10.0, collision_distance=0.01)
    assert am.classify_outcome(res).kind == "merger"

    system = am.BodySystem(G=1.0)
    system.add_body(1.0, [0, 0, 0], [0, 0, 0])
    system.add_body(1.0, [1, 0, 0], [10, 0, 0])
    res = am.integrate(system, n_periods=1, period=1.0)
    assert am.classify_outcome(res).kind == "unbound"


def test_bound_pairs_order():
    masses = np.array([1.0, 1.0, 1.0])
    pos = np.array([[0, 0, 0], [0.1, 0, 0], [5, 0, 0]], dtype=float)
    vel = np.zeros((3, 3))
    vel[1, 1] = np.sqrt(2 / 0.1)  # circular A-B
    pairs = am.bound_pairs_from_state(masses, pos, vel, G=1.0)
    assert pairs[0].bodies == (0, 1)
    assert pairs[0].semimajor_axis == pytest.approx(0.1)
    assert all(p.binding_energy <= pairs[0].binding_energy for p in pairs)


def test_plots(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    res = am.integrate(three_body(), n_periods=20, period="binary", n_samples=200)
    for ax in (
        res.plot_trajectories(),
        res.plot_trajectories(plane="xz", frame="centre_of_mass"),
        res.plot_trajectories(plane="yz", colour_by_time=False),
        res.plot_separations(),
        res.plot_energy(),
    ):
        ax.figure.savefig(tmp_path / "fig.png")


def test_trajectory_panels_share_time_scale(tmp_path):
    """plane='all' draws x-y, x-z, y-z with one shared time colour scale."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.collections import LineCollection

    res = am.integrate(
        three_body(),
        n_periods=300,
        period="binary",
        n_samples=500,
        ejection_radius=20.0,
    )
    axes = res.plot_trajectories(plane="all")
    assert len(axes) == 3
    assert [(a.get_xlabel(), a.get_ylabel()) for a in axes] == [
        ("$x$", "$y$"),
        ("$x$", "$z$"),
        ("$y$", "$z$"),
    ]
    collections = [
        c for a in axes for c in a.collections if isinstance(c, LineCollection)
    ]
    assert len(collections) == 3 * res.n_bodies
    limits = {(c.norm.vmin, c.norm.vmax) for c in collections}
    assert limits == {(res.times[0], res.times[-1])}
    # Inactive (NaN) samples of an ejected body produce no segments.
    for c in collections:
        assert np.all(np.isfinite(np.concatenate(c.get_segments())))
    axes[0].figure.savefig(tmp_path / "all.png")
    two = res.plot_trajectories(plane=["xy", "yz"])
    assert len(two) == 2
    with pytest.raises(ValueError):
        res.plot_trajectories(plane="xw")
