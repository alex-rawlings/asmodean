"""Stopping the integration once a (massive black hole) binary becomes hard."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import asmodean as am  # noqa: E402

G1 = am.UnitSystem(length=1.0, mass=1.0, G=1.0)


def binary(semimajor_axis=1.0, masses=(1.0, 1.0), eccentricity=0.0):
    system = am.BodySystem(G=1.0)
    system.add_binary(
        list(masses), semimajor_axis=semimajor_axis, eccentricity=eccentricity
    )
    return system


def dispersion_for(a_hard, masses=(1.0, 1.0)):
    """The 1D dispersion for which the hard-binary semimajor axis is a_hard."""
    mu = masses[0] * masses[1] / sum(masses)
    return float(np.sqrt(mu / (4.0 * a_hard)))


def test_hard_binary_semimajor_axis():
    assert am.hard_binary_semimajor_axis(1.0, 1.0, 0.5, G=1.0) == pytest.approx(0.5)
    assert am.hard_binary_semimajor_axis(3.0, 1.0, 1.0, G=2.0) == pytest.approx(
        2.0 * 0.75 / 4.0
    )


def test_wide_binary_runs_to_time_limit():
    """Without hardening, a binary wider than a_h never triggers the stop."""
    res = am.integrate(
        binary(1.0),
        n_periods=3,
        period="binary",
        hard_binary_dispersion=dispersion_for(0.5),
    )
    assert res.stop_reason == "time_limit"
    assert res.hard_binary is None


def test_already_hard_stops_at_start():
    res = am.integrate(
        binary(0.5, masses=(2.0, 1.0), eccentricity=0.4),
        n_periods=3,
        period="binary",
        hard_binary_dispersion=dispersion_for(1.0, masses=(2.0, 1.0)),
    )
    assert res.stop_reason == "hard_binary"
    assert res.t_end == 0.0
    ev = res.hard_binary
    assert (ev.body, ev.partner) == (0, 1)  # the more massive member first
    assert ev.mass == pytest.approx(3.0)
    assert ev.value == pytest.approx(0.5)
    assert ev.eccentricity == pytest.approx(0.4)
    np.testing.assert_allclose(ev.position, 0.0, atol=1e-12)


@pytest.mark.parametrize("source", ["fixed", "friction"])
def test_friction_hardens_binary_until_hard(source):
    """Individual-mode friction shrinks the orbit until a <= a_h."""
    a_hard = 0.6
    sigma = dispersion_for(a_hard)
    friction = am.DynamicalFriction.constant(2e-3, sigma, mode="individual")
    res = am.integrate(
        binary(1.0),
        friction=friction,
        n_periods=500,
        period="binary",
        hard_binary_dispersion=sigma if source == "fixed" else "friction",
    )
    assert res.stop_reason == "hard_binary"
    assert 0 < res.t_end < 500 * res.period
    ev = res.hard_binary
    assert ev.time == pytest.approx(res.t_end)
    assert {ev.body, ev.partner} == {0, 1}
    # Stopped within one step of crossing a_h.
    assert ev.value <= a_hard
    assert ev.value == pytest.approx(a_hard, rel=1e-2)
    final = am.bound_pairs(res, -1)[0]
    assert final.semimajor_axis == pytest.approx(ev.value, rel=1e-6)
    assert final.eccentricity == pytest.approx(ev.eccentricity, abs=1e-6)
    assert "hard binary" in res.summary()


def test_hardest_pair_is_reported():
    """With two hard pairs, the stop names the one with the smallest a / a_h."""
    system = am.BodySystem(G=1.0)
    system.add_binary([1.0, 1.0], semimajor_axis=0.2, labels=["A1", "A2"])
    system.add_body(1.0, [50.0, 0.0, 0.0], [0.0, 0.0, 0.0], label="B1")
    system.add_body(1.0, [50.0, 0.5, 0.0], [0.0, 0.0, 0.0], label="B2")
    # a_h = 1 for both pairs: A1-A2 (a = 0.2) is harder than B1-B2 (a = 0.25).
    res = am.integrate(
        system, n_periods=1, period=1.0, hard_binary_dispersion=dispersion_for(1.0)
    )
    assert res.stop_reason == "hard_binary"
    assert {res.hard_binary.body, res.hard_binary.partner} == {0, 1}


def test_settings_validation():
    with pytest.raises(ValueError):
        am.IntegrationSettings(hard_binary_dispersion=-1.0)
    with pytest.raises(ValueError):
        am.IntegrationSettings(hard_binary_dispersion="potential")
    with pytest.raises(ValueError, match="friction"):
        am.integrate(binary(), n_periods=1, hard_binary_dispersion="friction")


@pytest.mark.parametrize("sigma", [None, 0.3, "friction"])
def test_settings_round_trip(sigma):
    settings = am.IntegrationSettings(hard_binary_dispersion=sigma)
    back = am.IntegrationSettings.from_arrays(settings.to_arrays())
    assert back.hard_binary_dispersion == sigma


def test_result_round_trip(tmp_path):
    res = am.integrate(
        binary(0.5),
        n_periods=1,
        period="binary",
        hard_binary_dispersion=dispersion_for(1.0),
    )
    path = res.save(tmp_path / "result.npz")
    back = am.ScatteringResult.load(path)
    assert back.stop_reason == "hard_binary"
    assert back.hard_binary.kind == "hard_binary"
    assert back.hard_binary.value == pytest.approx(res.hard_binary.value)
    assert back.hard_binary.partner == res.hard_binary.partner
    assert back.hard_binary.eccentricity == pytest.approx(res.hard_binary.eccentricity)
    assert all(e.eccentricity is None for e in back.events if e.kind != "hard_binary")


def test_ensemble():
    systems = [binary(0.5), binary(2.0)]
    results = am.integrate_ensemble(
        systems,
        n_periods=2,
        period="binary",
        hard_binary_dispersion=dispersion_for(1.0),
    )
    assert [r.stop_reason for r in results] == ["hard_binary", "time_limit"]
