"""Chandrasekhar dynamical friction: the formula, its two modes, the energy
budget, and the background profiles."""

import os
import sys

import numpy as np
import pytest
from math import erf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import asmodean as am  # noqa: E402
from asmodean import _core  # noqa: E402

G1 = am.UnitSystem(length=1.0, mass=1.0, G=1.0)


def chandrasekhar(mass, rho, sigma, lnl, vel):
    """Reference Chandrasekhar deceleration (G = 1)."""
    vel = np.asarray(vel, dtype=float)
    v = np.linalg.norm(vel)
    X = v / (np.sqrt(2) * sigma)
    bracket = erf(X) - 2 * X / np.sqrt(np.pi) * np.exp(-X * X)
    return -4 * np.pi * mass * rho * lnl * bracket / v**3 * vel


@pytest.mark.parametrize("speed", [1e-4, 5e-3, 0.3, 1.0, 5.0])
def test_formula(speed):
    """The core reproduces the formula, including the small-X series branch."""
    df = am.DynamicalFriction.constant(2.0, 0.7, coulomb_logarithm=4.0)
    vel = speed * np.array([0.6, -0.8, 0.0])
    a = df.acceleration(0.3, [1.0, 2.0, 3.0], vel, G1)
    # The reference loses ~eps / X^2 to cancellation at small X (which is why
    # the core switches to a series there), so compare more loosely.
    rtol = 1e-9 if speed > 1e-2 else 1e-6
    np.testing.assert_allclose(a, chandrasekhar(0.3, 2.0, 0.7, 4.0, vel), rtol=rtol)


def test_small_speed_is_linear_drag():
    """At v << sigma the drag is linear in v, continuous across the series switch."""
    df = am.DynamicalFriction.constant(1.0, 1.0)
    X_switch = 1e-3
    v_switch = X_switch * np.sqrt(2)
    below = df.acceleration(1.0, [0, 0, 0], [v_switch * (1 - 1e-9), 0, 0], G1)[0]
    above = df.acceleration(1.0, [0, 0, 0], [v_switch * (1 + 1e-9), 0, 0], G1)[0]
    assert below / v_switch == pytest.approx(above / v_switch, rel=1e-6)
    k = 4 * np.pi * 3.0 * 4 / (3 * np.sqrt(np.pi)) / (2 * np.sqrt(2))
    assert below == pytest.approx(-k * v_switch, rel=1e-6)


def test_variable_coulomb_logarithm():
    df = am.DynamicalFriction.constant(1.0, 1.0, coulomb_logarithm="variable")
    core = df.to_core(G1)
    mass, pos, vel = 0.01, [2.0, 0, 0], [3.0, 0, 0]
    lam = 2.0 * (9.0 + 1.0) / mass
    lnl = 0.5 * np.log1p(lam**2)
    np.testing.assert_allclose(
        core.acceleration(mass, pos, vel), chandrasekhar(mass, 1.0, 1.0, lnl, vel)
    )


def test_profile_interpolation():
    r = np.array([1.0, 10.0, 100.0])
    rho = np.array([100.0, 1.0, 1e-2])
    sigma = np.array([2.0, 1.0, 0.5])
    core = am.DynamicalFriction(r, rho, sigma).to_core(G1)
    # Power-law between nodes (log-log linear) ...
    assert float(core.density(np.sqrt(10.0))) == pytest.approx(10.0)
    assert float(core.dispersion(np.sqrt(10.0))) == pytest.approx(np.sqrt(2.0))
    # ... and clamped outside the table.
    assert float(core.density(0.01)) == pytest.approx(100.0)
    assert float(core.density(1e4)) == pytest.approx(1e-2)
    assert float(core.dispersion(0.0)) == pytest.approx(2.0)


def binary_moving(com_velocity):
    system = am.BodySystem(G=1.0)
    system.add_binary(
        [1.0, 0.25],
        semimajor_axis=0.1,
        eccentricity=0.4,
        centre_velocity=com_velocity,
        labels=["A", "B"],
    )
    return system


def test_system_mode_shares_force_by_mass():
    """'system' mode: one drag on the total mass at the CoM velocity, shared in
    proportion to mass -> the same deceleration for every body."""
    system = binary_moving([2.0, 0.0, 0.0])
    df = am.DynamicalFriction.constant(0.5, 1.0)
    masses, states, soft = system.to_internal(G1)
    pot = am.ExternalPotential.none(G1).core
    no_df = _core.accelerations(pot, masses, states, soft, _core.DynamicalFriction())
    with_df = _core.accelerations(pot, masses, states, soft, df.to_core(G1))
    drag = with_df - no_df
    x_com, v_com = system.centre_of_mass()
    expected = chandrasekhar(system.total_mass, 0.5, 1.0, 3.0, v_com)
    np.testing.assert_allclose(drag, np.tile(expected, (2, 1)), rtol=1e-12)
    # The force on each body is proportional to its mass.
    forces = masses[:, None] * drag
    np.testing.assert_allclose(forces[0, 0] / forces[1, 0], masses[0] / masses[1])


def test_individual_mode():
    system = binary_moving([2.0, 0.0, 0.0])
    df = am.DynamicalFriction.constant(0.5, 1.0, mode="individual")
    masses, states, soft = system.to_internal(G1)
    pot = am.ExternalPotential.none(G1).core
    drag = _core.accelerations(pot, masses, states, soft, df.to_core(G1)) - (
        _core.accelerations(pot, masses, states, soft, _core.DynamicalFriction())
    )
    for k in range(2):
        np.testing.assert_allclose(
            drag[k], chandrasekhar(masses[k], 0.5, 1.0, 3.0, states[k, 3:]), rtol=1e-12
        )


def test_system_mode_leaves_binary_intact():
    """Friction in 'system' mode slows the CoM but not the internal orbit."""
    system = binary_moving([2.0, 0.0, 0.0])
    df = am.DynamicalFriction.constant(0.5, 1.0)
    res = am.integrate(system, friction=df, n_periods=50, period="binary", n_samples=51)
    _, v_com = res.centre_of_mass()
    assert np.linalg.norm(v_com[-1]) < 0.5 * np.linalg.norm(v_com[0])
    el = am.pair_elements(res, "A", "B")
    assert el[-1].semimajor_axis == pytest.approx(0.1, rel=1e-8)
    assert el[-1].eccentricity == pytest.approx(0.4, rel=1e-7)
    # Energy budget: the lost energy is the friction work.
    assert res.friction_work[-1] < 0
    assert np.max(np.abs(res.energy_error())) < 1e-9


def test_deceleration_matches_ode():
    """A lone body in a uniform background slows as dv/dt = |a_df(v)|."""
    scipy_integrate = pytest.importorskip("scipy.integrate")
    rho, sigma, lnl, m = 0.2, 1.0, 3.0, 0.5
    df = am.DynamicalFriction.constant(rho, sigma, coulomb_logarithm=lnl)
    system = am.BodySystem(G=1.0)
    system.add_body(m, [0.0, 0.0, 0.0], [3.0, 0.0, 0.0])
    T = 5.0
    res = am.integrate(system, friction=df, n_periods=1, period=T, n_samples=11)

    def rhs(_, y):
        return [chandrasekhar(m, rho, sigma, lnl, [y[0], 0, 0])[0]]

    ref = scipy_integrate.solve_ivp(
        rhs, (0, T), [3.0], t_eval=res.times, rtol=1e-12, atol=1e-14
    )
    np.testing.assert_allclose(res.velocities[:, 0, 0], ref.y[0], rtol=1e-8)
    assert np.max(np.abs(res.energy_error())) < 1e-10


def hernquist_dispersion(r):
    """Isotropic Hernquist (1990, eq. 10) dispersion, G = M = a = 1."""
    return np.sqrt(
        r * (1 + r) ** 3 * np.log((1 + r) / r)
        - r / (12 * (1 + r)) * (25 + 52 * r + 42 * r**2 + 12 * r**3)
    )


def test_profiles_from_potential_hernquist():
    """Gauss + isotropic Jeans on a Hernquist SCF fit reproduce the analytic model."""
    lf = pytest.importorskip("lanfear")
    rng = np.random.default_rng(5)
    n = 400_000
    su = np.sqrt(rng.uniform(0, 1, n))
    r = su / (1 - su)
    mu = rng.uniform(-1, 1, n)
    az = rng.uniform(0, 2 * np.pi, n)
    st = np.sqrt(1 - mu**2)
    pos = np.stack([r * st * np.cos(az), r * st * np.sin(az), r * mu], axis=1)
    core = lf._core.SCFPotential(12, 0, pos, np.full(n, 1.0 / n))
    pot = am.ExternalPotential.from_lanfear_core(core, 1.0, 1.0, G=1.0)
    df = am.DynamicalFriction.from_potential(pot, r_min=0.05, r_max=10.0, n_bins=40)
    rho_exact = 1 / (2 * np.pi * df.radius * (1 + df.radius) ** 3)
    inner = (df.radius > 0.1) & (df.radius < 5)
    np.testing.assert_allclose(df.density[inner], rho_exact[inner], rtol=0.05)
    np.testing.assert_allclose(
        df.dispersion[inner], hernquist_dispersion(df.radius[inner]), rtol=0.03
    )
    # A central point mass raises the dispersion near the centre only.
    df_bh = am.DynamicalFriction.from_potential(
        pot, r_min=0.05, r_max=10.0, n_bins=40, central_mass=0.01
    )
    assert df_bh.dispersion[0] > df.dispersion[0]
    np.testing.assert_allclose(df_bh.density, df.density)
    assert df_bh.dispersion[-1] == pytest.approx(df.dispersion[-1], rel=0.01)


def test_profiles_from_particles():
    """Binned density/dispersion of an isotropic Gaussian-velocity sample."""
    rng = np.random.default_rng(2)
    n = 200_000
    radius = rng.uniform(0, 1, n) ** (1 / 3)  # uniform sphere of radius 1
    direction = rng.normal(size=(n, 3))
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    pos = radius[:, None] * direction
    vel = rng.normal(scale=2.0, size=(n, 3))
    mass = np.full(n, 3.0 / n)
    df = am.DynamicalFriction.from_particles(
        pos, vel, mass, n_bins=10, r_min=0.2, r_max=0.9
    )
    np.testing.assert_allclose(df.density, 3.0 / (4 / 3 * np.pi), rtol=0.05)
    np.testing.assert_allclose(df.dispersion, 2.0, rtol=0.02)


def test_stiffness_warning(caplog):
    """A dense, cold background at the bodies' position triggers a warning."""
    system = binary_moving([0.0, 0.0, 0.0])
    df = am.DynamicalFriction.constant(1e6, 1e-3)
    with caplog.at_level("WARNING", logger="asmodean"):
        logger = am.get_logger()
        logger.propagate = True
        try:
            am.integrate(system, friction=df, n_periods=1e-6, period="binary")
        finally:
            logger.propagate = False
    assert "stiff" in caplog.text


def test_validation():
    with pytest.raises(ValueError):
        am.DynamicalFriction([1.0, 0.5], [1.0, 1.0], [1.0, 1.0])
    with pytest.raises(ValueError):
        am.DynamicalFriction([1.0], [-1.0], [1.0])
    with pytest.raises(ValueError):
        am.DynamicalFriction.constant(1.0, 1.0, mode="bodies")
    with pytest.raises(ValueError):
        am.DynamicalFriction.constant(1.0, 1.0, coulomb_logarithm="auto")
