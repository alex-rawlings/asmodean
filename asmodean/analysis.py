"""Analysis of scattering results: bound pairs, orbital elements, outcomes.

The two-body quantities here are Keplerian (unsoftened, and ignoring the
external potential and the other bodies), which is the standard description of
a binary that is compact compared to its surroundings.

:func:`classify_outcome` labels an experiment by comparing the most tightly
bound pair at the start with the one at the end, in the language of
binary-single scattering:

* ``"merger"`` -- at least one merger happened;
* ``"flyby"`` -- the same pair is the most bound one at the end (the binary
  survived, perhaps hardened or softened);
* ``"exchange"`` -- a different pair is the most bound one at the end;
* ``"ionisation"`` -- no pair is bound at the end;
* ``"unbound"`` -- no pair was bound at the start.

:func:`hard_binary_semimajor_axis` gives the semimajor axis ``a_h`` below
which a massive black hole binary is hard (the criterion behind
``IntegrationSettings.hard_binary_dispersion``).

:attr:`Outcome.resolved` records whether the interaction was over when the
integration stopped (an ejection or merger ended it) or merely ran out of time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .results import ScatteringResult


@dataclass(frozen=True)
class OrbitalElements:
    """Keplerian elements of a two-body orbit.

    Parameters
    ----------
    semimajor_axis : float
        ``a = -G M / (2 E)``; negative for an unbound (hyperbolic) orbit.
    eccentricity : float
        Eccentricity.
    energy : float
        Specific orbital energy ``v^2 / 2 - G M / r``.
    period : float
        Orbital period (infinite if unbound).
    pericentre : float
        Pericentre distance ``a (1 - e)`` (``|a| (e - 1)`` if unbound).
    apocentre : float
        Apocentre distance ``a (1 + e)`` (infinite if unbound).
    inclination : float
        Inclination of the orbital plane to the x-y plane (radians).
    separation : float
        Current separation.
    bound : bool
        Whether the orbit is bound (``energy < 0``).
    """

    semimajor_axis: float
    eccentricity: float
    energy: float
    period: float
    pericentre: float
    apocentre: float
    inclination: float
    separation: float
    bound: bool


def orbital_elements(
    m_1: float, m_2: float, x_1, x_2, v_1, v_2, G: float
) -> OrbitalElements:
    """Keplerian elements of the relative orbit of two bodies.

    Parameters
    ----------
    m_1, m_2 : float
        Masses.
    x_1, x_2 : array-like of float
        (3,) positions.
    v_1, v_2 : array-like of float
        (3,) velocities.
    G : float
        Gravitational constant.

    Returns
    -------
    elements : OrbitalElements
        The elements of the relative orbit.
    """
    mu = G * (m_1 + m_2)
    r_vec = np.asarray(x_2, dtype=np.float64) - np.asarray(x_1, dtype=np.float64)
    v_vec = np.asarray(v_2, dtype=np.float64) - np.asarray(v_1, dtype=np.float64)
    r = float(np.linalg.norm(r_vec))
    energy = 0.5 * float(np.dot(v_vec, v_vec)) - mu / r
    h_vec = np.cross(r_vec, v_vec)
    h = float(np.linalg.norm(h_vec))
    ecc = float(np.sqrt(max(0.0, 1.0 + 2.0 * energy * h * h / (mu * mu))))
    inclination = float(np.arccos(np.clip(h_vec[2] / h, -1.0, 1.0))) if h > 0 else 0.0
    if energy < 0:
        a = -mu / (2.0 * energy)
        return OrbitalElements(
            a,
            ecc,
            energy,
            float(2.0 * np.pi * np.sqrt(a**3 / mu)),
            a * (1.0 - ecc),
            a * (1.0 + ecc),
            inclination,
            r,
            True,
        )
    a = -mu / (2.0 * energy) if energy != 0 else -np.inf
    peri = abs(a) * (ecc - 1.0) if np.isfinite(a) else h * h / (2.0 * mu)
    return OrbitalElements(a, ecc, energy, np.inf, peri, np.inf, inclination, r, False)


@dataclass(frozen=True)
class BoundPair:
    """A gravitationally bound pair of bodies.

    Parameters
    ----------
    body_1, body_2 : int
        The bodies' indices (``body_1 < body_2``).
    elements : OrbitalElements
        The pair's relative orbit.
    binding_energy : float
        ``-E_orb`` of the pair: ``G m_1 m_2 / (2 a)`` (positive).
    """

    body_1: int
    body_2: int
    elements: OrbitalElements
    binding_energy: float

    @property
    def bodies(self) -> Tuple[int, int]:
        """The two body indices.

        Returns
        -------
        bodies : tuple of int
            ``(body_1, body_2)``.
        """
        return (self.body_1, self.body_2)

    @property
    def semimajor_axis(self) -> float:
        """Semimajor axis of the pair.

        Returns
        -------
        a : float
            The semimajor axis.
        """
        return self.elements.semimajor_axis

    @property
    def eccentricity(self) -> float:
        """Eccentricity of the pair.

        Returns
        -------
        e : float
            The eccentricity.
        """
        return self.elements.eccentricity

    @property
    def period(self) -> float:
        """Orbital period of the pair.

        Returns
        -------
        period : float
            The period.
        """
        return self.elements.period


def bound_pairs_from_state(masses, positions, velocities, G: float) -> List[BoundPair]:
    """Every bound pair among a set of bodies, most tightly bound first.

    Bodies with non-finite mass (inactive in a result) are skipped.

    Parameters
    ----------
    masses : array-like of float
        (n,) masses.
    positions : array-like of float
        (n, 3) positions.
    velocities : array-like of float
        (n, 3) velocities.
    G : float
        Gravitational constant.

    Returns
    -------
    pairs : list of BoundPair
        Bound pairs sorted by decreasing binding energy ``G m_1 m_2 / (2a)``.
    """
    masses = np.asarray(masses, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    velocities = np.asarray(velocities, dtype=np.float64)
    live = [i for i in range(len(masses)) if np.isfinite(masses[i])]
    pairs = []
    for a_idx, i in enumerate(live):
        for j in live[a_idx + 1 :]:
            el = orbital_elements(
                masses[i],
                masses[j],
                positions[i],
                positions[j],
                velocities[i],
                velocities[j],
                G,
            )
            if el.bound:
                binding = G * masses[i] * masses[j] / (2.0 * el.semimajor_axis)
                pairs.append(BoundPair(i, j, el, binding))
    return sorted(pairs, key=lambda p: -p.binding_energy)


def bound_pairs(result: ScatteringResult, sample: int = -1) -> List[BoundPair]:
    """Bound pairs of the active bodies at one sample of a result.

    Parameters
    ----------
    result : ScatteringResult
        The result.
    sample : int, optional
        Sample index (default the final state).

    Returns
    -------
    pairs : list of BoundPair
        Bound pairs, most tightly bound first.
    """
    return bound_pairs_from_state(
        result.masses[sample],
        result.positions[sample],
        result.velocities[sample],
        result.units.G,
    )


def pair_elements(
    result: ScatteringResult, body_1, body_2
) -> List[Optional[OrbitalElements]]:
    """Orbital elements of one pair at every sample.

    Parameters
    ----------
    result : ScatteringResult
        The result.
    body_1, body_2 : int or str
        The bodies (index or label).

    Returns
    -------
    elements : list of OrbitalElements or None
        One entry per sample; None where either body is inactive.
    """
    i, j = result.index(body_1), result.index(body_2)
    out = []
    for k in range(result.n_samples):
        m = result.masses[k]
        if not (np.isfinite(m[i]) and np.isfinite(m[j])):
            out.append(None)
            continue
        out.append(
            orbital_elements(
                m[i],
                m[j],
                result.positions[k, i],
                result.positions[k, j],
                result.velocities[k, i],
                result.velocities[k, j],
                result.units.G,
            )
        )
    return out


def hard_binary_semimajor_axis(mass_1, mass_2, dispersion, G: float):
    """Semimajor axis below which a massive black hole binary is hard.

    ``a_h = G mu / (4 sigma^2)``, with ``mu = m_1 m_2 / (m_1 + m_2)`` the
    reduced mass and ``sigma`` the 1D velocity dispersion of the surrounding
    stars (Merritt 2013, *Dynamics and Evolution of Galactic Nuclei*,
    ch. 8).

    Parameters
    ----------
    mass_1, mass_2 : float or array-like of float
        The binary's masses.
    dispersion : float or array-like of float
        Background 1D velocity dispersion.
    G : float
        Gravitational constant.

    Returns
    -------
    a_h : float or numpy.ndarray
        The hard-binary semimajor axis.
    """
    mass_1 = np.asarray(mass_1, dtype=np.float64)
    mass_2 = np.asarray(mass_2, dtype=np.float64)
    mu = mass_1 * mass_2 / (mass_1 + mass_2)
    return G * mu / (4.0 * np.asarray(dispersion, dtype=np.float64) ** 2)


@dataclass(frozen=True)
class Outcome:
    """Classification of a scattering experiment.

    Parameters
    ----------
    kind : str
        ``"merger"``, ``"flyby"``, ``"exchange"``, ``"ionisation"`` or
        ``"unbound"`` (see the module docstring).
    resolved : bool
        Whether the interaction had ended when the integration stopped (every
        body merged, or only non-interacting bodies remain after an ejection).
    stop_reason : str
        Why the integration stopped.
    initial_pair : BoundPair or None
        The most tightly bound pair at the start.
    final_pair : BoundPair or None
        The most tightly bound pair at the end.
    ejected : list of int
        Ejected bodies.
    mergers : list of tuple
        ``(survivor, absorbed)`` of each merger.
    """

    kind: str
    resolved: bool
    stop_reason: str
    initial_pair: Optional[BoundPair]
    final_pair: Optional[BoundPair]
    ejected: List[int]
    mergers: List[Tuple[int, int]]

    def describe(self, labels: Optional[List[str]] = None) -> str:
        """One-paragraph description.

        Parameters
        ----------
        labels : list of str, optional
            Body names (default indices).

        Returns
        -------
        text : str
            The description.
        """
        name = (lambda i: labels[i]) if labels else str

        def pair_text(p: Optional[BoundPair]) -> str:
            if p is None:
                return "none"
            return (
                f"{name(p.body_1)}-{name(p.body_2)} (a={p.semimajor_axis:.4g}, "
                f"e={p.eccentricity:.3f})"
            )

        lines = [
            f"Outcome: {self.kind} ({'resolved' if self.resolved else 'unresolved'}; "
            f"stopped: {self.stop_reason})",
            f"  initial binary: {pair_text(self.initial_pair)}",
            f"  final binary:   {pair_text(self.final_pair)}",
        ]
        if self.mergers:
            lines.append(
                "  mergers: "
                + ", ".join(f"{name(b)}->{name(a)}" for a, b in self.mergers)
            )
        if self.ejected:
            lines.append("  ejected: " + ", ".join(name(i) for i in self.ejected))
        return "\n".join(lines)


def classify_outcome(result: ScatteringResult) -> Outcome:
    """Classify a scattering experiment (see the module docstring).

    Parameters
    ----------
    result : ScatteringResult
        The result.

    Returns
    -------
    outcome : Outcome
        The classification.
    """
    initial = bound_pairs(result, 0)
    final = bound_pairs(result, -1)
    p0 = initial[0] if initial else None
    p1 = final[0] if final else None
    mergers = [(e.body, e.partner) for e in result.mergers]
    ejected = [e.body for e in result.ejections]
    if mergers:
        kind = "merger"
    elif p0 is None:
        kind = "unbound"
    elif p1 is None:
        kind = "ionisation"
    elif set(p1.bodies) == set(p0.bodies):
        kind = "flyby"
    else:
        kind = "exchange"
    resolved = result.stop_reason in ("all_merged", "bodies_ejected") or bool(ejected)
    return Outcome(kind, resolved, result.stop_reason, p0, p1, ejected, mergers)
