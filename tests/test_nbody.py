"""Few-body physics of the C++ core: softening, Kepler orbits, mergers, ejections.

All tests work in G = 1 units with no background potential, where the answers
are analytic.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import asmodean as am  # noqa: E402
from asmodean import _core  # noqa: E402

G1 = am.UnitSystem(length=1.0, mass=1.0, G=1.0)


def no_potential():
    return am.ExternalPotential.none(G1)


# ------------------------------------------------------------------ softening
def test_spline_kernel_is_gadget():
    """The spline kernel is Newtonian beyond h = 2.8 eps and Plummer-like at 0."""
    eps = 0.1
    h = float(_core.SPLINE_TO_PLUMMER) * eps
    assert h == pytest.approx(2.8 * eps)
    # -1/eps at the centre: the Plummer-equivalent depth.
    assert _core.spline_potential(0.0, h) == pytest.approx(-1.0 / eps)
    for r in (h, 1.5 * h, 10 * h):
        assert _core.spline_potential(r, h) == pytest.approx(-1.0 / r)
        assert _core.spline_force_factor(r, h) == pytest.approx(1.0 / r**3)
    # The force is minus the gradient of the potential everywhere.
    for r in np.linspace(0.05, 1.5, 25) * h:
        dr = 1e-6 * h
        dphi = (
            _core.spline_potential(r + dr, h) - _core.spline_potential(r - dr, h)
        ) / (2 * dr)
        assert _core.spline_force_factor(r, h) * r == pytest.approx(dphi, rel=1e-6)


def test_pair_softening_uses_larger_kernel():
    """Two bodies of different softening interact with max(h_i, h_j)."""
    h_small, h_large = 0.01, 0.3
    r = 0.1  # inside the larger kernel only
    masses = np.array([1.0, 2.0])
    states = np.zeros((2, 6))
    states[1, 0] = r
    acc = _core.accelerations(
        no_potential().core,
        masses,
        states,
        np.array([h_small, h_large]),
        _core.DynamicalFriction(),
    )
    g = _core.spline_force_factor(r, h_large)
    assert acc[0, 0] == pytest.approx(masses[1] * g * r)
    assert acc[1, 0] == pytest.approx(-masses[0] * g * r)
    # Momentum conservation: sum m a = 0.
    np.testing.assert_allclose(masses @ acc, 0.0, atol=1e-14)


# ------------------------------------------------------------------ Kepler
@pytest.mark.parametrize("stepper", ["rkf78", "dopri5"])
def test_kepler_binary(stepper):
    """A Keplerian binary returns to its initial state and conserves energy."""
    system = am.BodySystem(G=1.0)
    system.add_binary([1.0, 0.3], semimajor_axis=1.0, eccentricity=0.6, inclination=0.4)
    n_orbits = 20
    res = am.integrate(
        system,
        n_periods=n_orbits,
        period="binary",
        n_samples=n_orbits + 1,
        stepper=stepper,
        rel_tol=1e-12,
    )
    assert res.stop_reason == "time_limit"
    assert res.period == pytest.approx(2 * np.pi / np.sqrt(1.3))
    np.testing.assert_allclose(res.positions[-1], res.positions[0], atol=1e-6)
    assert np.max(np.abs(res.energy_error())) < 1e-9
    el = am.pair_elements(res, 0, 1)
    assert el[-1].semimajor_axis == pytest.approx(1.0, rel=1e-8)
    assert el[-1].eccentricity == pytest.approx(0.6, rel=1e-8)


def test_encounter_is_resolved_with_step_cap():
    """A near-radial passage through the softening kernel conserves energy.

    Without the kernel step cap the high-order error estimate is fooled by the
    piecewise-polynomial spline force and ~1e-4 of the energy is lost. The
    remaining error is the tolerance amplified by KE_peri / |E_0| ~ 1e4.
    """
    system = am.BodySystem(G=1.0)
    system.add_body(1.0, [-1.0, 1e-3, 0.0], [0.5, 0.0, 0.0], softening=1e-4)
    system.add_body(1.0, [1.0, -1e-3, 0.0], [-0.5, 0.0, 0.0], softening=1e-4)
    el = am.orbital_elements(1.0, 1.0, *system.positions, *system.velocities, G=1.0)
    assert el.pericentre < 1e-5  # passes deep inside the softening kernel
    res = am.integrate(system, n_periods=1, period=10.0, n_samples=100)
    assert res.stop_reason == "time_limit"
    assert np.max(np.abs(res.energy_error())) < 1e-7
    loose = am.integrate(system, n_periods=1, period=10.0, kernel_step_factor=0.1)
    assert np.max(np.abs(loose.energy_error())) > 1e-5  # the failure being guarded


# ------------------------------------------------------------------ mergers
def radial_infall_time(d0, d_c, m_total):
    x = d_c / d0
    return np.sqrt(d0**3 / (2 * m_total)) * (
        np.sqrt(x * (1 - x)) + np.arccos(np.sqrt(x))
    )


def test_radial_infall_merger():
    """Two bodies released from rest merge at the analytic contact time."""
    d0, d_c = 1.0, 0.02
    m1, m2 = 1.0, 0.25
    system = am.BodySystem(G=1.0)
    system.add_body(m1, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], label="A")
    system.add_body(m2, [d0, 0.0, 0.0], [0.0, 0.0, 0.0], label="B")
    p0 = m1 * system.velocities[0] + m2 * system.velocities[1]
    res = am.integrate(
        system, n_periods=10, period=1.0, collision_distance=d_c, n_samples=1000
    )
    assert res.stop_reason == "all_merged"
    assert len(res.mergers) == 1
    ev = res.mergers[0]
    assert ev.time == pytest.approx(radial_infall_time(d0, d_c, m1 + m2), rel=1e-9)
    # The more massive body survives with the total mass.
    assert (ev.body, ev.partner) == (0, 1)
    assert ev.mass == pytest.approx(m1 + m2)
    assert res.status == ["active", "merged"]
    assert res.merged_into[1] == 0
    # Momentum is conserved and the merged body sits at the centre of mass.
    np.testing.assert_allclose((m1 + m2) * ev.velocity, p0, atol=1e-12)
    np.testing.assert_allclose(ev.position, [m2 * d0 / (m1 + m2), 0, 0], atol=1e-12)
    # The contact speed is the free-fall speed at d_c.
    v_c = np.sqrt(2 * (m1 + m2) * (1 / d_c - 1 / d0))
    assert ev.value == pytest.approx(v_c, rel=1e-8)
    # The merger removes the pair's (bound) relative energy; the budget closes
    # (|E_0| is small next to the kinetic energy at contact, hence 1e-8).
    assert np.max(np.abs(res.energy_error())) < 1e-8


def test_merger_then_continued_evolution():
    """Regression: after a merger the remaining bodies are integrated correctly.

    Two bodies fall together and merge while a third orbits far away; the third
    must keep orbiting the merged body smoothly, with the energy budget intact.
    """
    system = am.BodySystem(G=1.0)
    system.add_body(1.0, [-0.5, 0.0, 0.0], [0.0, 0.0, 0.0], label="A")
    system.add_body(1.0, [0.5, 0.0, 0.0], [0.0, 0.0, 0.0], label="B")
    r_c = 20.0
    system.add_body(0.1, [0.0, r_c, 0.0], [np.sqrt(2.1 / r_c), 0.0, 0.0], label="C")
    res = am.integrate(
        system,
        n_periods=1,
        period=40.0,
        collision_distance=0.01,
        ejection_radius=1e3,
        n_samples=400,
    )
    assert [e.kind for e in res.events] == ["merger"]
    assert res.stop_reason == "time_limit"
    assert np.max(np.abs(res.energy_error())) < 1e-8
    # C stays on a near-circular orbit about the merged body.
    sep = res.separation("A", "C")
    after = res.times > res.mergers[0].time + 1
    assert np.nanmax(np.abs(sep[after] / r_c - 1)) < 0.05
    # Trajectories are continuous across the merger (no scrambled states).
    steps = np.linalg.norm(np.diff(res.positions[:, 2], axis=0), axis=1)
    assert np.max(steps) < 5 * np.median(steps)


def test_merger_at_start():
    """Bodies already in contact at t = 0 merge immediately."""
    system = am.BodySystem(G=1.0)
    system.add_body(1.0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    system.add_body(1.0, [0.001, 0.0, 0.0], [0.0, 0.1, 0.0])
    res = am.integrate(system, period=1.0, collision_distance=0.01)
    assert res.stop_reason == "all_merged"
    assert res.mergers[0].time == 0.0
    assert res.t_end == 0.0


# ------------------------------------------------------------------ ejections
def test_hyperbolic_flyby_ejection():
    """An unbound flyby: the lighter body is ejected once beyond the radius."""
    system = am.BodySystem(G=1.0)
    system.add_body(1.0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], label="heavy")
    system.add_incoming_body(
        0.1, v_infinity=1.0, impact_parameter=1.0, distance=10.0, label="light"
    )
    r_ej = 20.0
    res = am.integrate(system, n_periods=1, period=200.0, ejection_radius=r_ej)
    assert res.stop_reason == "bodies_ejected"
    assert [e.kind for e in res.ejections] == ["ejection"]
    ev = res.ejections[0]
    assert res.labels[ev.body] == "light"  # tie in distance -> lighter body
    assert res.separation(0, 1)[-2] < r_ej  # ejected on first exceeding r_ej
    # Escape energy of the relative orbit is v_inf^2 / 2.
    assert ev.value == pytest.approx(0.5, rel=1e-6)
    assert np.max(np.abs(res.energy_error())) < 1e-9


def test_binary_member_is_not_ejected():
    """A single escaping a binary is ejected; the bound binary is kept."""
    system = am.BodySystem(G=1.0)
    system.add_binary(
        [1.0, 1.0], semimajor_axis=0.1, eccentricity=0.2, labels=["A", "B"]
    )
    system.add_body(0.5, [3.0, 0.0, 0.0], [5.0, 0.0, 0.0], label="C")
    res = am.integrate(system, n_periods=100, period="binary", ejection_radius=5.0)
    assert [res.labels[e.body] for e in res.ejections] == ["C"]
    assert res.stop_reason == "time_limit"
    assert res.remaining == [0, 1]
    pairs = am.bound_pairs(res)
    assert pairs[0].bodies == (0, 1)
    assert pairs[0].semimajor_axis == pytest.approx(0.1, rel=1e-3)


def test_ejection_from_centre():
    """With the centre reference a lone unbound body is ejected past the radius."""
    system = am.BodySystem(G=1.0)
    system.add_body(1.0, [1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    res = am.integrate(
        system,
        n_periods=1,
        period=100.0,
        ejection_radius=10.0,
        ejection_reference="centre",
    )
    assert res.stop_reason == "bodies_ejected"
    assert res.ejections[0].time == pytest.approx(9.0, abs=0.05)
    assert res.status == ["ejected"]


def test_single_body_runs_to_time_limit():
    """One body alone is not an 'all merged' state: it runs to the time limit."""
    system = am.BodySystem(G=1.0)
    system.add_body(1.0, [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    res = am.integrate(system, n_periods=1, period=3.0, n_samples=4)
    assert res.stop_reason == "time_limit"
    np.testing.assert_allclose(res.positions[-1, 0], [1.0, 3.0, 0.0], atol=1e-12)


# ------------------------------------------------------------------ limits
def test_python_constants_match_core():
    assert am.MAX_BODIES == _core.MAX_BODIES
    assert am.STOP_REASONS == list(_core.STOP_REASONS)
    assert am.EVENT_TYPES == list(_core.EVENT_TYPES)
    assert am.BODY_STATUSES == list(_core.BODY_STATUSES)


def test_body_limit():
    system = am.BodySystem(G=1.0)
    for i in range(am.MAX_BODIES):
        system.add_body(1.0, [i, 0, 0], [0, 0, 0])
    with pytest.raises(ValueError):
        system.add_body(1.0, [99, 0, 0], [0, 0, 0])


def test_step_limit():
    system = am.BodySystem(G=1.0)
    system.add_binary([1.0, 1.0], semimajor_axis=1.0)
    res = am.integrate(system, n_periods=100, period="binary", max_steps=50)
    assert res.stop_reason == "step_limit"
    assert res.n_steps == 50
