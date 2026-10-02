"""Run scattering experiments: settings, the time limit, and integration.

:func:`integrate` integrates one :class:`~asmodean.BodySystem` in an
:class:`~asmodean.ExternalPotential` (optionally with
:class:`~asmodean.DynamicalFriction`) and returns a
:class:`~asmodean.ScatteringResult`. :func:`integrate_ensemble` integrates many
independent experiments in parallel (OpenMP in the C++ core).
:class:`ScatteringSetup` bundles everything one experiment needs, and saves to
/ loads from a ``.npz`` file (see ``scripts/make_system.py`` and
``scripts/run_scattering.py``).

The integration stops at the first of (see :class:`IntegrationSettings`):

* the time limit, ``n_periods`` reference periods;
* every body merged into one (pairs closer than ``collision_distance`` are
  lumped together);
* at most one body left after an ejection (a body beyond ``ejection_radius``
  that has escaped is removed and no longer integrated);
* optionally, a pair of bodies (a massive black hole binary) becoming hard
  (see ``hard_binary_dispersion``);
* the step limit, or an integration error.
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass, fields, replace
from typing import List, Optional, Sequence, Union

import numpy as np

from . import _core
from ._logging import get_logger
from .analysis import bound_pairs_from_state
from .bodies import BodySystem
from .friction import DynamicalFriction
from .potential import ExternalPotential
from .results import ScatteringResult
from .units import UnitSystem

logger = get_logger(__name__)

_SETUP_FORMAT = "asmodean-setup-1"
_EJECTION_REFERENCES = {"system": 0, "centre": 1}
_STEPPERS = {"rkf78": 0, "dopri5": 1}
_HARD_BINARY_FIXED = 1
_HARD_BINARY_FRICTION = 2
PERIOD_CHOICES = ("internal", "binary", "circular")


@dataclass
class IntegrationSettings:
    """How to integrate a scattering experiment (physical units).

    Parameters
    ----------
    n_periods : float, optional
        Time limit, in reference periods (see ``period``).
    period : float or str, optional
        The reference period: a physical time, or how to estimate it from the
        initial conditions (see :func:`estimate_period`): ``"internal"``
        (default) -- the Keplerian period ``2 pi sqrt(R^3 / (G M))`` of the
        bodies' total mass ``M`` at their mass-weighted RMS radius ``R`` about
        their centre of mass; ``"binary"`` -- the Keplerian period of the most
        tightly bound pair; ``"circular"`` -- the circular period of the
        external potential at the bodies' mass-weighted mean distance from its
        centre.
    n_samples : int, optional
        Number of uniformly spaced output samples over the full time limit
        (fewer are recorded if the integration stops early; the final state is
        always recorded).
    collision_distance : float, optional
        Two bodies closer than this are merged (momentum-conserving). None
        disables mergers.
    ejection_radius : float, optional
        Bodies beyond this distance that have escaped are removed. None
        disables ejections.
    ejection_reference : {"system", "centre"}, optional
        ``"system"`` (default): distance and escape are measured relative to
        the centre of mass of the other bodies, which the body must be
        unbound from (and not bound to any single one of them).
        ``"centre"``: relative to the potential centre, with positive energy
        in the potential plus the other bodies' field.
    rel_tol : float, optional
        Relative error tolerance of the adaptive integrator.
    abs_tol : float, optional
        Absolute error tolerance, in the potential's internal units.
    step_factor : float, optional
        Caps each step at this fraction of the shortest pairwise crossing or
        free-fall time, so encounters and collisions are never stepped over.
    kernel_step_factor : float, optional
        While two bodies are inside (1.2 times) their softening kernel, caps
        each step so they move at most this fraction of the kernel length
        ``h = 2.8 epsilon`` relative to each other. The spline force is only
        piecewise smooth, which fools the error estimate of a high-order
        integrator; without this cap a passage through the kernel can lose
        energy far beyond the tolerance.
    max_steps : int, optional
        Stop after this many accepted steps.
    stepper : {"rkf78", "dopri5"}, optional
        Runge-Kutta-Fehlberg 7(8) (default) or Dormand-Prince 5(4).
    record_trajectory : bool, optional
        Record the ``n_samples`` samples (default); if False only the initial
        and final states are kept (e.g. for large ensembles).
    hard_binary_dispersion : float or "friction", optional
        Stop once a pair of bodies -- assumed to be a massive black hole
        binary -- becomes hard: bound, with semimajor axis
        ``a <= a_h = G mu / (4 sigma^2)``, where ``mu`` is the pair's reduced
        mass and ``sigma`` the background 1D velocity dispersion (Merritt
        2013, ch. 8). A float is a fixed ``sigma`` (physical velocity);
        ``"friction"`` takes ``sigma`` from the dynamical-friction profile at
        the pair's centre-of-mass distance from the potential centre (needs
        dynamical friction). None (default) disables the criterion. It is
        tested after every step, and at the start: a binary that is already
        hard stops the integration at once. The stop is recorded as a
        ``"hard_binary"`` event, with the pair's semimajor axis and
        eccentricity.
    """

    n_periods: float = 100.0
    period: Union[float, str] = "internal"
    n_samples: int = 2000
    collision_distance: Optional[float] = None
    ejection_radius: Optional[float] = None
    ejection_reference: str = "system"
    rel_tol: float = 1e-11
    abs_tol: float = 1e-14
    step_factor: float = 0.1
    kernel_step_factor: float = 0.01
    max_steps: int = 100_000_000
    stepper: str = "rkf78"
    record_trajectory: bool = True
    hard_binary_dispersion: Optional[Union[float, str]] = None

    def __post_init__(self) -> None:
        """Validate the settings.

        Raises
        ------
        ValueError
            If a setting is invalid.
        """
        if not (self.n_periods > 0):
            raise ValueError(f"n_periods must be positive, got {self.n_periods}")
        if isinstance(self.period, str):
            if self.period not in PERIOD_CHOICES:
                raise ValueError(
                    f"period must be a time or one of {PERIOD_CHOICES}, got "
                    f"'{self.period}'"
                )
        elif not (self.period > 0):
            raise ValueError(f"period must be positive, got {self.period}")
        if self.ejection_reference not in _EJECTION_REFERENCES:
            raise ValueError(
                f"ejection_reference must be one of {sorted(_EJECTION_REFERENCES)}"
            )
        if self.stepper not in _STEPPERS:
            raise ValueError(f"stepper must be one of {sorted(_STEPPERS)}")
        for name in ("collision_distance", "ejection_radius"):
            value = getattr(self, name)
            if value is not None and not (value > 0):
                raise ValueError(f"{name} must be positive or None, got {value}")
        sigma = self.hard_binary_dispersion
        if isinstance(sigma, str):
            if sigma != "friction":
                raise ValueError(
                    f"hard_binary_dispersion must be a velocity, 'friction' or "
                    f"None, got '{sigma}'"
                )
        elif sigma is not None and not (sigma > 0 and np.isfinite(sigma)):
            raise ValueError(
                f"hard_binary_dispersion must be positive or None, got {sigma}"
            )

    def to_core(self, units: UnitSystem, t_max: float):
        """The C++ settings, in internal units.

        Parameters
        ----------
        units : UnitSystem
            The internal unit system.
        t_max : float
            Time limit (physical time).

        Returns
        -------
        settings : asmodean._core.IntegrationSettings
            The C++ settings object.
        """
        s = _core.IntegrationSettings()
        s.t_max = t_max / units.time
        s.n_samples = int(self.n_samples)
        s.abs_tol = float(self.abs_tol)
        s.rel_tol = float(self.rel_tol)
        s.collision_distance = (
            0.0
            if self.collision_distance is None
            else self.collision_distance / units.length
        )
        s.ejection_radius = (
            0.0 if self.ejection_radius is None else self.ejection_radius / units.length
        )
        s.ejection_reference = _EJECTION_REFERENCES[self.ejection_reference]
        s.step_factor = float(self.step_factor)
        s.kernel_step_factor = float(self.kernel_step_factor)
        s.max_steps = int(self.max_steps)
        s.stepper = _STEPPERS[self.stepper]
        s.record_trajectory = bool(self.record_trajectory)
        sigma = self.hard_binary_dispersion
        if sigma == "friction":
            s.hard_binary = _HARD_BINARY_FRICTION
        elif sigma is not None:
            s.hard_binary = _HARD_BINARY_FIXED
            s.hard_binary_dispersion = float(sigma) / units.velocity
        return s

    def to_arrays(self, prefix: str = "settings/") -> dict:
        """Serialise to a dict of NumPy arrays (``None`` stored as ``"None"``).

        Parameters
        ----------
        prefix : str, optional
            Prefix for every key.

        Returns
        -------
        arrays : dict of str to numpy.ndarray
            The settings.
        """
        return {
            f"{prefix}{k}": np.asarray("None" if v is None else v)
            for k, v in asdict(self).items()
        }

    @classmethod
    def from_arrays(cls, arrays, prefix: str = "settings/") -> "IntegrationSettings":
        """Rebuild from :meth:`to_arrays` output.

        Parameters
        ----------
        arrays : mapping of str to numpy.ndarray
            The stored arrays.
        prefix : str, optional
            Prefix used when storing.

        Returns
        -------
        settings : IntegrationSettings
            The settings.
        """
        kwargs = {}
        for f in fields(cls):
            key = f"{prefix}{f.name}"
            if key not in arrays:
                continue
            value = np.asarray(arrays[key]).item()
            if value == "None":
                value = None
            elif f.name in ("period", "hard_binary_dispersion") and isinstance(
                value, str
            ):
                try:
                    value = float(value)
                except ValueError:
                    pass
            kwargs[f.name] = value
        return cls(**kwargs)


def estimate_period(
    system: BodySystem,
    potential: Optional[ExternalPotential] = None,
    method: Union[float, str] = "internal",
) -> float:
    """The reference period that sets the time limit.

    Parameters
    ----------
    system : BodySystem
        The initial bodies.
    potential : ExternalPotential, optional
        The background potential (needed for ``"circular"``).
    method : float or str, optional
        A physical time (returned as is), or ``"internal"``, ``"binary"`` or
        ``"circular"`` (see :class:`IntegrationSettings`).

    Returns
    -------
    period : float
        The period (physical time).

    Raises
    ------
    ValueError
        If the period cannot be estimated with the chosen method.
    """
    if not isinstance(method, str):
        return float(method)
    G = system.G
    if method == "internal":
        if system.n_bodies < 2:
            raise ValueError("period='internal' needs at least two bodies")
        R = system.rms_radius()
        return float(2.0 * np.pi * np.sqrt(R**3 / (G * system.total_mass)))
    if method == "binary":
        pairs = bound_pairs_from_state(
            system.masses, system.positions, system.velocities, G
        )
        if not pairs:
            raise ValueError("period='binary' needs a bound pair of bodies")
        return float(pairs[0].period)
    if method == "circular":
        if potential is None or potential.is_null:
            raise ValueError("period='circular' needs an external potential")
        r = np.linalg.norm(system.positions, axis=1)
        r_mean = float(np.sum(system.masses * r) / system.total_mass)
        if r_mean <= 0:
            raise ValueError("period='circular' needs bodies away from the centre")
        period = float(potential.circular_period(r_mean)[0])
        if not np.isfinite(period):
            raise ValueError(f"no circular orbit at r={r_mean:.4g}")
        return period
    raise ValueError(f"unknown period method '{method}'")


def _units_for(system: BodySystem, potential: Optional[ExternalPotential]):
    """The potential to integrate in and its internal units.

    With no potential, internal units are matched to the bodies (their RMS
    radius and total mass), so the tolerances are meaningful.

    Parameters
    ----------
    system : BodySystem
        The bodies.
    potential : ExternalPotential, optional
        The background potential.

    Returns
    -------
    potential : ExternalPotential
        The potential (a null one if none was given).

    Raises
    ------
    ValueError
        If the bodies' G differs from the potential's.
    """
    if potential is None:
        length = system.rms_radius() if system.n_bodies > 1 else 0.0
        if not length > 0:
            length = float(np.max(np.linalg.norm(system.positions, axis=1))) or 1.0
        return ExternalPotential.none(
            UnitSystem(length=length, mass=system.total_mass, G=system.G)
        )
    if not np.isclose(potential.units.G, system.G, rtol=1e-10):
        raise ValueError(
            f"the bodies use G={system.G} but the potential G={potential.units.G}"
        )
    return potential


def _metadata(settings, potential, friction, period) -> dict:
    """Provenance stored with a result.

    Parameters
    ----------
    settings : IntegrationSettings
        The settings used.
    potential : ExternalPotential
        The potential used.
    friction : DynamicalFriction or None
        The friction model used.
    period : float
        The reference period.

    Returns
    -------
    metadata : dict
        Flat provenance dict.
    """
    meta = {
        f"settings.{k}": ("None" if v is None else v)
        for k, v in asdict(settings).items()
    }
    meta["potential_kind"] = potential.kind
    meta["potential_black_holes"] = potential.n_black_holes
    meta["friction"] = friction is not None
    if friction is not None:
        meta["friction.mode"] = friction.mode
        meta["friction.coulomb_logarithm"] = str(friction.coulomb_logarithm)
    return meta


def _check_hard_binary_settings(settings, friction) -> None:
    """Check that the hard-binary criterion has a dispersion to work with.

    Parameters
    ----------
    settings : IntegrationSettings
        The settings.
    friction : DynamicalFriction or None
        The friction model.

    Raises
    ------
    ValueError
        If ``hard_binary_dispersion="friction"`` without dynamical friction.
    """
    if settings.hard_binary_dispersion == "friction" and friction is None:
        raise ValueError(
            "hard_binary_dispersion='friction' needs dynamical friction (or give "
            "the dispersion as a velocity)"
        )


def _check_friction_stiffness(system, units, friction_core, period) -> None:
    """Warn if dynamical friction would make the equations of motion stiff.

    At low speed the Chandrasekhar drag is linear, ``a = -k v`` with
    ``k ~ G^2 M rho / sigma^3``; when ``1 / k`` is far shorter than the
    reference period the explicit integrator must take correspondingly tiny
    steps. This usually signals a background dispersion that is unrealistically
    small where the bodies are (see ``central_mass`` in
    :meth:`DynamicalFriction.from_potential`).

    Parameters
    ----------
    system : BodySystem
        The initial bodies.
    units : UnitSystem
        The internal unit system.
    friction_core : asmodean._core.DynamicalFriction
        The friction model in internal units.
    period : float
        The reference period (physical time).
    """
    masses, states, _ = system.to_internal(units)
    if friction_core.mode == 0:  # system: one drag on the centre of mass
        m_tot = float(np.sum(masses))
        targets = [(m_tot, np.sum(masses[:, None] * states[:, :3], axis=0) / m_tot)]
    else:
        targets = [(float(m), x[:3]) for m, x in zip(masses, states)]
    rate = 0.0
    for m, x in targets:
        v_test = 1e-3 * float(friction_core.dispersion(np.linalg.norm(x)))
        a = friction_core.acceleration(m, list(x), [v_test, 0.0, 0.0])
        rate = max(rate, float(np.linalg.norm(a)) / v_test)
    if rate <= 0:
        return
    damping_time = units.time / rate
    if damping_time < 1e-3 * period:
        logger.warning(
            f"Dynamical friction is stiff at the initial positions: the low-speed "
            f"damping time {damping_time:.3g} is {period / damping_time:.3g} times "
            f"shorter than the reference period, so the integration will be very "
            f"slow. Is the background dispersion near the bodies realistic (e.g. "
            f"DynamicalFriction.from_potential(..., central_mass=...))?"
        )


def integrate(
    system: BodySystem,
    potential: Optional[ExternalPotential] = None,
    settings: Optional[IntegrationSettings] = None,
    friction: Optional[DynamicalFriction] = None,
    **overrides,
) -> ScatteringResult:
    """Integrate one scattering experiment.

    Parameters
    ----------
    system : BodySystem
        The initial bodies (at most 10).
    potential : ExternalPotential, optional
        The static background potential; none by default.
    settings : IntegrationSettings, optional
        Integration settings; defaults if omitted.
    friction : DynamicalFriction, optional
        Dynamical friction; off if omitted.
    **overrides
        Individual :class:`IntegrationSettings` fields overriding ``settings``.

    Returns
    -------
    result : ScatteringResult
        Trajectories, events and final states, in physical units.
    """
    settings = replace(settings or IntegrationSettings(), **overrides)
    _check_hard_binary_settings(settings, friction)
    potential = _units_for(system, potential)
    units = potential.units
    period = estimate_period(system, potential, settings.period)
    t_max = settings.n_periods * period
    masses, states, soft = system.to_internal(units)
    friction_core = (
        friction.to_core(units) if friction is not None else _core.DynamicalFriction()
    )
    if friction is not None:
        _check_friction_stiffness(system, units, friction_core, period)

    logger.info(
        f"Integrating {system.n_bodies} bodies for {settings.n_periods:g} periods "
        f"of {period:.4g} (t_max={t_max:.4g}) in a '{potential.kind}' potential"
        + (f" with {friction.mode} dynamical friction" if friction is not None else "")
    )
    t0 = time.perf_counter()
    raw = _core.integrate(
        potential.core,
        masses,
        states,
        soft,
        settings.to_core(units, t_max),
        friction_core,
    )
    elapsed = time.perf_counter() - t0
    result = ScatteringResult.from_core(
        raw,
        system.labels,
        units,
        period,
        settings.n_periods,
        _metadata(settings, potential, friction, period),
    )
    logger.info(
        f"Finished in {elapsed:.2f} s ({result.n_steps} steps): {result.stop_reason} "
        f"at t={result.t_end:.4g}, {len(result.mergers)} merger(s), "
        f"{len(result.ejections)} ejection(s)"
    )
    if result.stop_reason in ("step_limit", "integration_error"):
        logger.warning(f"Integration ended early: {result.stop_reason}")
    if result.stop_reason == "hard_binary" and result.t_end == 0:
        logger.warning("A binary was already hard at the start: nothing integrated")
    return result


def integrate_ensemble(
    systems: Sequence[BodySystem],
    potential: Optional[ExternalPotential] = None,
    settings: Optional[IntegrationSettings] = None,
    friction: Optional[DynamicalFriction] = None,
    progress: bool = False,
    **overrides,
) -> List[ScatteringResult]:
    """Integrate many independent experiments in parallel (OpenMP).

    Every system must have the same number of bodies. With no potential the
    internal units are matched to the first system. Set
    ``record_trajectory=False`` to keep only initial and final states when the
    ensemble is large.

    Parameters
    ----------
    systems : sequence of BodySystem
        The initial conditions of each experiment.
    potential : ExternalPotential, optional
        The shared background potential.
    settings : IntegrationSettings, optional
        Shared integration settings (the period is estimated per system).
    friction : DynamicalFriction, optional
        Shared dynamical friction.
    progress : bool, optional
        Print progress every 10% of experiments.
    **overrides
        Individual :class:`IntegrationSettings` fields overriding ``settings``.

    Returns
    -------
    results : list of ScatteringResult or None
        One result per system, in order; None for an experiment whose
        integration raised (e.g. invalid initial conditions).

    Raises
    ------
    ValueError
        If the systems differ in their number of bodies.
    """
    systems = list(systems)
    if not systems:
        return []
    n = systems[0].n_bodies
    if any(s.n_bodies != n for s in systems):
        raise ValueError(
            "every system in an ensemble must have the same number of bodies"
        )
    settings = replace(settings or IntegrationSettings(), **overrides)
    _check_hard_binary_settings(settings, friction)
    potential = _units_for(systems[0], potential)
    units = potential.units
    periods = np.array(
        [estimate_period(s, potential, settings.period) for s in systems]
    )
    internal = [s.to_internal(units) for s in systems]
    masses = np.stack([m for m, _, _ in internal])
    states = np.stack([x for _, x, _ in internal])
    soft = np.stack([h for _, _, h in internal])
    t_max = settings.n_periods * periods / units.time
    friction_core = (
        friction.to_core(units) if friction is not None else _core.DynamicalFriction()
    )
    if friction is not None:
        _check_friction_stiffness(systems[0], units, friction_core, periods[0])

    t0 = time.perf_counter()
    raws = _core.integrate_batch(
        potential.core,
        masses,
        states,
        soft,
        t_max,
        settings.to_core(units, 1.0),
        friction_core,
        progress,
    )
    logger.info(
        f"Integrated {len(systems)} experiments in {time.perf_counter() - t0:.2f} s"
    )
    meta = [_metadata(settings, potential, friction, p) for p in periods]
    return [
        ScatteringResult.from_core(r, s.labels, units, p, settings.n_periods, m)
        if len(r["times"])
        else None
        for r, s, p, m in zip(raws, systems, periods, meta)
    ]


@dataclass
class ScatteringSetup:
    """Everything one experiment needs, savable to a ``.npz`` file.

    Parameters
    ----------
    system : BodySystem
        The initial bodies.
    potential : ExternalPotential, optional
        The background potential (None: no potential).
    friction : DynamicalFriction, optional
        Dynamical friction (None: off).
    settings : IntegrationSettings, optional
        Integration settings.
    """

    system: BodySystem
    potential: Optional[ExternalPotential] = None
    friction: Optional[DynamicalFriction] = None
    settings: IntegrationSettings = None

    def __post_init__(self) -> None:
        """Default the settings."""
        if self.settings is None:
            self.settings = IntegrationSettings()

    def run(self, **overrides) -> ScatteringResult:
        """Integrate this experiment.

        Parameters
        ----------
        **overrides
            Individual :class:`IntegrationSettings` fields to override.

        Returns
        -------
        result : ScatteringResult
            The result.
        """
        return integrate(
            self.system, self.potential, self.settings, self.friction, **overrides
        )

    def save(self, path: Union[str, os.PathLike]) -> str:
        """Write the setup to a ``.npz`` archive.

        Parameters
        ----------
        path : str or os.PathLike
            Destination (``.npz`` is appended by NumPy if absent).

        Returns
        -------
        path : str
            The path written.
        """
        arrays = {"_format": np.asarray(_SETUP_FORMAT)}
        arrays.update(self.system.to_arrays())
        arrays.update(self.settings.to_arrays())
        if self.potential is not None:
            arrays.update(self.potential.to_arrays())
        if self.friction is not None:
            arrays.update(self.friction.to_arrays())
        np.savez(path, **arrays)
        out = os.fspath(path)
        out = out if out.endswith(".npz") else out + ".npz"
        logger.info(f"Wrote scattering setup to {out}")
        return out

    @classmethod
    def load(cls, path: Union[str, os.PathLike]) -> "ScatteringSetup":
        """Read a setup written by :meth:`save`.

        Parameters
        ----------
        path : str or os.PathLike
            A ``.npz`` file written by :meth:`save`.

        Returns
        -------
        setup : ScatteringSetup
            The setup.

        Raises
        ------
        ValueError
            If the file is not a asmodean setup.
        """
        with np.load(path, allow_pickle=False) as npz:
            if "_format" not in npz or str(npz["_format"]) != _SETUP_FORMAT:
                raise ValueError(f"{os.fspath(path)!r} is not a asmodean setup file")
            potential = (
                ExternalPotential.from_arrays(npz)
                if "potential/_format" in npz
                else None
            )
            friction = (
                DynamicalFriction.from_arrays(npz) if "friction/radius" in npz else None
            )
            return cls(
                system=BodySystem.from_saved_arrays(npz),
                potential=potential,
                friction=friction,
                settings=IntegrationSettings.from_arrays(npz),
            )
