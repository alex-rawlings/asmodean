"""The bodies of a scattering experiment: masses, phase-space states, softening.

A :class:`BodySystem` holds at most :data:`MAX_BODIES` (10) point masses in
physical units, each with its own mass and Gadget softening length, and
provides helpers for the usual scattering set-ups: a Keplerian binary from its
orbital elements (:meth:`BodySystem.add_binary`), a body arriving on a
hyperbolic orbit with a given speed at infinity and impact parameter
(:meth:`BodySystem.add_incoming_body`), or the black holes of a lanfear
snapshot (:meth:`BodySystem.from_particles`).

The orbital helpers are Keplerian: they ignore the external potential and any
bodies outside the pair/target, which is appropriate when the set-up is
compact compared to the scale on which the background varies.

Softening follows Gadget: each body's ``softening`` is the Plummer-equivalent
length epsilon, and the force uses the cubic-spline kernel with compact
support ``h = 2.8 epsilon`` (Newtonian beyond ``h``); two bodies interact with
the larger of their two kernel lengths. A softening of zero is exactly
Newtonian.
"""

from __future__ import annotations

import copy
from typing import Optional, Sequence, Tuple, Union

import numpy as np

from . import _core
from ._logging import get_logger
from .units import DEFAULT_G

logger = get_logger(__name__)

# Maximum number of bodies (kMaxBodies in the C++ core).
MAX_BODIES = 10


def rotation_matrix(inclination: float, node: float, argument: float) -> np.ndarray:
    """Perifocal-to-reference rotation ``Rz(node) Rx(inclination) Rz(argument)``.

    Parameters
    ----------
    inclination : float
        Inclination (radians).
    node : float
        Longitude of the ascending node (radians).
    argument : float
        Argument of pericentre (radians).

    Returns
    -------
    rotation : numpy.ndarray
        (3, 3) rotation matrix.
    """
    cO, sO = np.cos(node), np.sin(node)
    ci, si = np.cos(inclination), np.sin(inclination)
    cw, sw = np.cos(argument), np.sin(argument)
    rz_node = np.array([[cO, -sO, 0.0], [sO, cO, 0.0], [0.0, 0.0, 1.0]])
    rx_inc = np.array([[1.0, 0.0, 0.0], [0.0, ci, -si], [0.0, si, ci]])
    rz_arg = np.array([[cw, -sw, 0.0], [sw, cw, 0.0], [0.0, 0.0, 1.0]])
    return rz_node @ rx_inc @ rz_arg


def kepler_to_cartesian(
    mu: float,
    semimajor_axis: float,
    eccentricity: float,
    inclination: float = 0.0,
    longitude_of_ascending_node: float = 0.0,
    argument_of_pericentre: float = 0.0,
    true_anomaly: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Relative position and velocity of a bound Keplerian orbit.

    Parameters
    ----------
    mu : float
        ``G (m_1 + m_2)`` in physical units.
    semimajor_axis : float
        Semimajor axis (physical length).
    eccentricity : float
        Eccentricity, ``0 <= e < 1``.
    inclination, longitude_of_ascending_node, argument_of_pericentre : float, optional
        Orientation angles (radians) relative to the x-y plane and x axis.
    true_anomaly : float, optional
        Orbital phase (radians); zero is pericentre.

    Returns
    -------
    position : numpy.ndarray
        (3,) separation vector ``x_2 - x_1``.
    velocity : numpy.ndarray
        (3,) relative velocity ``v_2 - v_1``.

    Raises
    ------
    ValueError
        If the elements do not describe a bound orbit.
    """
    if not (semimajor_axis > 0):
        raise ValueError(f"semimajor axis must be positive, got {semimajor_axis}")
    if not (0.0 <= eccentricity < 1.0):
        raise ValueError(f"eccentricity must be in [0, 1), got {eccentricity}")
    p = semimajor_axis * (1.0 - eccentricity**2)
    r = p / (1.0 + eccentricity * np.cos(true_anomaly))
    pos_pf = r * np.array([np.cos(true_anomaly), np.sin(true_anomaly), 0.0])
    vel_pf = np.sqrt(mu / p) * np.array(
        [-np.sin(true_anomaly), eccentricity + np.cos(true_anomaly), 0.0]
    )
    rot = rotation_matrix(
        inclination, longitude_of_ascending_node, argument_of_pericentre
    )
    return rot @ pos_pf, rot @ vel_pf


class BodySystem:
    """The bodies of one scattering experiment, in physical units.

    Parameters
    ----------
    G : float, optional
        Gravitational constant in the physical unit system (default Gadget: kpc,
        1e10 Msun, km/s). Used by the orbital helpers; it must match the
        external potential's.
    """

    def __init__(self, G: float = DEFAULT_G) -> None:
        self.G = float(G)
        self._masses: list = []
        self._positions: list = []
        self._velocities: list = []
        self._softening: list = []
        self._labels: list = []

    # ------------------------------------------------------------- builders
    @classmethod
    def from_arrays(
        cls,
        masses,
        positions,
        velocities,
        softening=0.0,
        labels: Optional[Sequence[str]] = None,
        G: float = DEFAULT_G,
    ) -> "BodySystem":
        """Build from arrays of masses and phase-space coordinates.

        Parameters
        ----------
        masses : array-like of float
            (n,) masses.
        positions : array-like of float
            (n, 3) positions.
        velocities : array-like of float
            (n, 3) velocities.
        softening : float or array-like of float, optional
            Plummer-equivalent softening per body (or one value for all).
        labels : sequence of str, optional
            Body names; default ``body0``, ``body1``, ...
        G : float, optional
            Gravitational constant in the physical unit system.

        Returns
        -------
        system : BodySystem
            The bodies.
        """
        masses = np.atleast_1d(np.asarray(masses, dtype=np.float64))
        positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
        velocities = np.asarray(velocities, dtype=np.float64).reshape(-1, 3)
        softening = np.broadcast_to(
            np.asarray(softening, dtype=np.float64), masses.shape
        )
        system = cls(G=G)
        for i in range(len(masses)):
            system.add_body(
                masses[i],
                positions[i],
                velocities[i],
                softening=float(softening[i]),
                label=None if labels is None else labels[i],
            )
        return system

    @classmethod
    def from_particles(
        cls,
        particles,
        softening=0.0,
        labels: Optional[Sequence[str]] = None,
        G: float = DEFAULT_G,
    ) -> "BodySystem":
        """Use particles of a lanfear snapshot (typically its black holes) as bodies.

        Parameters
        ----------
        particles : lanfear.ParticleSystem
            The particles to turn into bodies, e.g. ``ps.black_holes`` of a
            prepared system (so they share the potential's frame).
        softening : float or array-like of float, optional
            Plummer-equivalent softening per body.
        labels : sequence of str, optional
            Body names; default ``"<species><particle ID>"``.
        G : float, optional
            Gravitational constant in the physical unit system.

        Returns
        -------
        system : BodySystem
            The bodies.
        """
        if labels is None:
            labels = [f"{s}{i}" for s, i in zip(particles.species, particles.ids)]
        return cls.from_arrays(
            particles.mass, particles.pos, particles.vel, softening, labels, G
        )

    def add_body(
        self,
        mass: float,
        position,
        velocity,
        softening: float = 0.0,
        label: Optional[str] = None,
    ) -> int:
        """Add one body.

        Parameters
        ----------
        mass : float
            Mass (physical units).
        position : array-like of float
            (3,) position.
        velocity : array-like of float
            (3,) velocity.
        softening : float, optional
            Plummer-equivalent softening length (zero: Newtonian).
        label : str, optional
            Name; default ``body<index>``.

        Returns
        -------
        index : int
            The new body's index.

        Raises
        ------
        ValueError
            If the system is full, or the inputs are invalid.
        """
        if self.n_bodies >= MAX_BODIES:
            raise ValueError(f"at most {MAX_BODIES} bodies are supported")
        if not (np.isfinite(mass) and mass > 0):
            raise ValueError(f"mass must be positive and finite, got {mass}")
        if not (np.isfinite(softening) and softening >= 0):
            raise ValueError(f"softening must be non-negative, got {softening}")
        pos = np.asarray(position, dtype=np.float64).reshape(3)
        vel = np.asarray(velocity, dtype=np.float64).reshape(3)
        if not (np.all(np.isfinite(pos)) and np.all(np.isfinite(vel))):
            raise ValueError("position and velocity must be finite")
        label = f"body{self.n_bodies}" if label is None else str(label)
        if label in self._labels:
            raise ValueError(f"duplicate body label '{label}'")
        self._masses.append(float(mass))
        self._positions.append(pos)
        self._velocities.append(vel)
        self._softening.append(float(softening))
        self._labels.append(label)
        return self.n_bodies - 1

    def add_binary(
        self,
        masses: Sequence[float],
        semimajor_axis: float,
        eccentricity: float = 0.0,
        inclination: float = 0.0,
        longitude_of_ascending_node: float = 0.0,
        argument_of_pericentre: float = 0.0,
        true_anomaly: float = 0.0,
        centre_position=(0.0, 0.0, 0.0),
        centre_velocity=(0.0, 0.0, 0.0),
        softening: Union[float, Sequence[float]] = 0.0,
        labels: Optional[Sequence[str]] = None,
    ) -> Tuple[int, int]:
        """Add a Keplerian binary from its orbital elements.

        Parameters
        ----------
        masses : sequence of float
            ``(m_1, m_2)``.
        semimajor_axis : float
            Semimajor axis of the relative orbit.
        eccentricity : float, optional
            Eccentricity (``0 <= e < 1``).
        inclination, longitude_of_ascending_node, argument_of_pericentre : float, optional
            Orientation angles (radians).
        true_anomaly : float, optional
            Orbital phase (radians); zero is pericentre.
        centre_position, centre_velocity : array-like of float, optional
            (3,) centre-of-mass position and velocity of the pair.
        softening : float or sequence of float, optional
            Plummer-equivalent softening of each member (or one for both).
        labels : sequence of str, optional
            The two members' names.

        Returns
        -------
        indices : tuple of int
            The two new bodies' indices.
        """
        m1, m2 = (float(m) for m in masses)
        m = m1 + m2
        rel_pos, rel_vel = kepler_to_cartesian(
            self.G * m,
            semimajor_axis,
            eccentricity,
            inclination,
            longitude_of_ascending_node,
            argument_of_pericentre,
            true_anomaly,
        )
        com_pos = np.asarray(centre_position, dtype=np.float64)
        com_vel = np.asarray(centre_velocity, dtype=np.float64)
        soft = np.broadcast_to(np.asarray(softening, dtype=np.float64), (2,))
        labels = labels or (None, None)
        i = self.add_body(
            m1,
            com_pos - m2 / m * rel_pos,
            com_vel - m2 / m * rel_vel,
            float(soft[0]),
            labels[0],
        )
        j = self.add_body(
            m2,
            com_pos + m1 / m * rel_pos,
            com_vel + m1 / m * rel_vel,
            float(soft[1]),
            labels[1],
        )
        return i, j

    def add_incoming_body(
        self,
        mass: float,
        v_infinity: float,
        impact_parameter: float,
        distance: float,
        target: Optional[Sequence[Union[int, str]]] = None,
        rotation=None,
        softening: float = 0.0,
        label: Optional[str] = None,
    ) -> int:
        """Add a body approaching a group of bodies on a hyperbolic orbit.

        The target group is treated as a point mass at its centre of mass. The
        new body is placed at ``distance`` from it on the incoming branch of the
        hyperbola with speed at infinity ``v_infinity`` and impact parameter
        ``impact_parameter``, then the target is recoiled so that the combined
        centre of mass and momentum of target + newcomer equal the target's
        original ones.

        Before ``rotation`` is applied, the body starts on the negative x axis
        relative to the target, moving in +x, with the orbit in the x-y plane
        (angular momentum along -z).

        Parameters
        ----------
        mass : float
            Mass of the incoming body.
        v_infinity : float
            Relative speed at infinity (> 0).
        impact_parameter : float
            Impact parameter (>= 0) of the relative orbit.
        distance : float
            Initial separation from the target's centre of mass; should be
            large compared to the target's size.
        target : sequence of int or str, optional
            Indices or labels of the target bodies; default every existing body.
        rotation : array-like of float, optional
            (3, 3) rotation matrix orienting the encounter (e.g. from
            ``scipy.spatial.transform.Rotation.random().as_matrix()``).
        softening : float, optional
            Plummer-equivalent softening of the new body.
        label : str, optional
            The new body's name.

        Returns
        -------
        index : int
            The new body's index.

        Raises
        ------
        ValueError
            If there is no target or the geometry is impossible
            (``distance`` inside the orbit's reach for this impact parameter).
        """
        idx = self._resolve(target)
        if not idx:
            raise ValueError("add_incoming_body needs at least one target body")
        if not (v_infinity > 0):
            raise ValueError(f"v_infinity must be positive, got {v_infinity}")
        m_t = float(np.sum(self.masses[idx]))
        x_t = np.sum(self.masses[idx, None] * self.positions[idx], axis=0) / m_t
        v_t = np.sum(self.masses[idx, None] * self.velocities[idx], axis=0) / m_t
        mu = self.G * (m_t + mass)
        speed = np.sqrt(v_infinity**2 + 2.0 * mu / distance)
        v_tan = impact_parameter * v_infinity / distance
        if v_tan > speed:
            raise ValueError(
                f"distance {distance} is too small for impact parameter "
                f"{impact_parameter}: the body would not be on the incoming branch"
            )
        v_rad = np.sqrt(speed**2 - v_tan**2)
        rel_pos = np.array([-distance, 0.0, 0.0])
        rel_vel = np.array([v_rad, v_tan, 0.0])
        if rotation is not None:
            rot = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
            rel_pos, rel_vel = rot @ rel_pos, rot @ rel_vel

        m = m_t + mass
        for i in idx:
            self._positions[i] = self._positions[i] - mass / m * rel_pos
            self._velocities[i] = self._velocities[i] - mass / m * rel_vel
        return self.add_body(
            mass,
            x_t + m_t / m * rel_pos,
            v_t + m_t / m * rel_vel,
            softening,
            label,
        )

    # ----------------------------------------------------------- accessors
    @property
    def n_bodies(self) -> int:
        """Number of bodies.

        Returns
        -------
        n : int
            The body count.
        """
        return len(self._masses)

    @property
    def masses(self) -> np.ndarray:
        """Masses, (n,).

        Returns
        -------
        masses : numpy.ndarray
            A copy of the masses.
        """
        return np.asarray(self._masses, dtype=np.float64)

    @property
    def positions(self) -> np.ndarray:
        """Positions, (n, 3).

        Returns
        -------
        positions : numpy.ndarray
            A copy of the positions.
        """
        return np.asarray(self._positions, dtype=np.float64).reshape(-1, 3)

    @property
    def velocities(self) -> np.ndarray:
        """Velocities, (n, 3).

        Returns
        -------
        velocities : numpy.ndarray
            A copy of the velocities.
        """
        return np.asarray(self._velocities, dtype=np.float64).reshape(-1, 3)

    @property
    def softening(self) -> np.ndarray:
        """Plummer-equivalent softening lengths, (n,).

        Returns
        -------
        softening : numpy.ndarray
            A copy of the softening lengths.
        """
        return np.asarray(self._softening, dtype=np.float64)

    @property
    def labels(self) -> list:
        """Body names.

        Returns
        -------
        labels : list of str
            A copy of the labels.
        """
        return list(self._labels)

    @property
    def total_mass(self) -> float:
        """Total mass of the bodies.

        Returns
        -------
        mass : float
            Sum of the masses.
        """
        return float(np.sum(self.masses))

    def index(self, label: str) -> int:
        """Index of the body called ``label``.

        Parameters
        ----------
        label : str
            A body name.

        Returns
        -------
        index : int
            Its index.
        """
        return self._labels.index(label)

    def _resolve(self, bodies) -> list:
        """Indices for a selection of bodies given by index or label.

        Parameters
        ----------
        bodies : sequence of int or str, optional
            The selection; None selects every body.

        Returns
        -------
        indices : list of int
            The selected indices.
        """
        if bodies is None:
            return list(range(self.n_bodies))
        return [
            b if isinstance(b, (int, np.integer)) else self.index(b) for b in bodies
        ]

    def centre_of_mass(self, bodies=None) -> Tuple[np.ndarray, np.ndarray]:
        """Centre-of-mass position and velocity.

        Parameters
        ----------
        bodies : sequence of int or str, optional
            Subset of bodies (default all).

        Returns
        -------
        position : numpy.ndarray
            (3,) centre-of-mass position.
        velocity : numpy.ndarray
            (3,) centre-of-mass velocity.
        """
        idx = self._resolve(bodies)
        m = self.masses[idx]
        return (
            np.sum(m[:, None] * self.positions[idx], axis=0) / np.sum(m),
            np.sum(m[:, None] * self.velocities[idx], axis=0) / np.sum(m),
        )

    def shift(self, position=(0.0, 0.0, 0.0), velocity=(0.0, 0.0, 0.0)) -> None:
        """Translate every body in position and velocity.

        Parameters
        ----------
        position : array-like of float, optional
            (3,) offset added to every position.
        velocity : array-like of float, optional
            (3,) offset added to every velocity.
        """
        dx = np.asarray(position, dtype=np.float64)
        dv = np.asarray(velocity, dtype=np.float64)
        self._positions = [p + dx for p in self._positions]
        self._velocities = [v + dv for v in self._velocities]

    def rms_radius(self) -> float:
        """Mass-weighted RMS distance of the bodies from their centre of mass.

        Returns
        -------
        radius : float
            ``sqrt(sum m |x - X|^2 / M)``.
        """
        x_com, _ = self.centre_of_mass()
        d2 = np.sum((self.positions - x_com) ** 2, axis=1)
        return float(np.sqrt(np.sum(self.masses * d2) / self.total_mass))

    def energy(self, potential=None) -> float:
        """Total energy: kinetic + external potential + softened pairwise.

        Parameters
        ----------
        potential : ExternalPotential, optional
            Background potential (default none).

        Returns
        -------
        energy : float
            Total energy in physical units (mass velocity^2).
        """
        from .potential import ExternalPotential
        from .units import UnitSystem

        if potential is None:
            potential = ExternalPotential.none(UnitSystem(1.0, 1.0, self.G))
        u = potential.units
        masses, states, soft = self.to_internal(u)
        return (
            float(_core.total_energy(potential.core, masses, states, soft)) * u.energy
        )

    def copy(self) -> "BodySystem":
        """An independent copy.

        Returns
        -------
        system : BodySystem
            The copy.
        """
        return copy.deepcopy(self)

    # ------------------------------------------------------------ internal
    def to_internal(self, units) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Masses, states and spline softening lengths in internal units.

        Parameters
        ----------
        units : UnitSystem
            The internal unit system.

        Returns
        -------
        masses : numpy.ndarray
            (n,) masses.
        states : numpy.ndarray
            (n, 6) positions and velocities.
        softening : numpy.ndarray
            (n,) spline-kernel lengths ``h = 2.8 epsilon``.
        """
        states = np.empty((self.n_bodies, 6))
        states[:, :3] = self.positions / units.length
        states[:, 3:] = self.velocities / units.velocity
        return (
            self.masses / units.mass,
            states,
            float(_core.SPLINE_TO_PLUMMER) * self.softening / units.length,
        )

    # ----------------------------------------------------------------- io
    def to_arrays(self, prefix: str = "bodies/") -> dict:
        """Serialise to a dict of NumPy arrays (for ``.npz`` files).

        Parameters
        ----------
        prefix : str, optional
            Prefix for every key.

        Returns
        -------
        arrays : dict of str to numpy.ndarray
            The bodies.
        """
        return {
            f"{prefix}masses": self.masses,
            f"{prefix}positions": self.positions,
            f"{prefix}velocities": self.velocities,
            f"{prefix}softening": self.softening,
            f"{prefix}labels": np.asarray(self.labels),
            f"{prefix}G": np.asarray(self.G),
        }

    @classmethod
    def from_saved_arrays(cls, arrays, prefix: str = "bodies/") -> "BodySystem":
        """Rebuild from :meth:`to_arrays` output.

        Parameters
        ----------
        arrays : mapping of str to numpy.ndarray
            The stored arrays.
        prefix : str, optional
            Prefix used when storing.

        Returns
        -------
        system : BodySystem
            The bodies.
        """
        return cls.from_arrays(
            arrays[f"{prefix}masses"],
            arrays[f"{prefix}positions"],
            arrays[f"{prefix}velocities"],
            arrays[f"{prefix}softening"],
            [str(s) for s in arrays[f"{prefix}labels"]],
            float(arrays[f"{prefix}G"]),
        )

    def __repr__(self) -> str:
        """Short description.

        Returns
        -------
        text : str
            The representation.
        """
        return f"BodySystem(n_bodies={self.n_bodies}, labels={self.labels})"
