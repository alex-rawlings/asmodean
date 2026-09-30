"""Unit bookkeeping between physical units and the C++ core's internal units.

Everything the user passes in or gets back is in *physical* units -- by
default the Gadget system of kpc, 1e10 Msun and km/s, with
``G = 43009.1 kpc (km/s)^2 / (1e10 Msun)``. The C++ core integrates in the
internal (Hernquist-Ostriker, "HO") units of the external potential, in which
``G = 1`` and the length and mass units are the potential's scale radius and
field mass (lanfear's convention). A :class:`UnitSystem` holds that length and
mass unit and derives the rest.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Gravitational constant in the default Gadget unit system
# (kpc, 1e10 Msun, km/s): G = 4.30091e-6 kpc (km/s)^2 / Msun * 1e10.
DEFAULT_G = 43009.1


@dataclass(frozen=True)
class UnitSystem:
    """Internal (G = 1) unit system, expressed in physical units.

    Parameters
    ----------
    length : float
        Physical length of one internal length unit.
    mass : float
        Physical mass of one internal mass unit.
    G : float, optional
        Gravitational constant in the physical unit system (default Gadget).
    """

    length: float
    mass: float
    G: float = DEFAULT_G

    def __post_init__(self) -> None:
        """Check the units are positive and finite.

        Raises
        ------
        ValueError
            If any unit is not a positive, finite number.
        """
        for name in ("length", "mass", "G"):
            value = getattr(self, name)
            if not (np.isfinite(value) and value > 0):
                raise ValueError(
                    f"unit {name} must be positive and finite, got {value}"
                )

    @property
    def velocity(self) -> float:
        """Physical velocity of one internal velocity unit, ``sqrt(G M / L)``.

        Returns
        -------
        velocity : float
            The velocity unit.
        """
        return float(np.sqrt(self.G * self.mass / self.length))

    @property
    def time(self) -> float:
        """Physical time of one internal time unit, ``L / V``.

        Returns
        -------
        time : float
            The time unit.
        """
        return self.length / self.velocity

    @property
    def energy(self) -> float:
        """Physical energy of one internal energy unit, ``M V^2``.

        Returns
        -------
        energy : float
            The energy unit.
        """
        return self.mass * self.velocity**2

    @property
    def specific_energy(self) -> float:
        """Physical specific energy (potential) unit, ``V^2``.

        Returns
        -------
        specific_energy : float
            The specific-energy unit.
        """
        return self.velocity**2

    @property
    def acceleration(self) -> float:
        """Physical acceleration unit, ``V^2 / L``.

        Returns
        -------
        acceleration : float
            The acceleration unit.
        """
        return self.velocity**2 / self.length

    @property
    def density(self) -> float:
        """Physical density unit, ``M / L^3``.

        Returns
        -------
        density : float
            The density unit.
        """
        return self.mass / self.length**3

    def to_arrays(self, prefix: str = "") -> dict:
        """Serialise to a dict of NumPy arrays (for ``.npz`` files).

        Parameters
        ----------
        prefix : str, optional
            Prefix for every key.

        Returns
        -------
        arrays : dict of str to numpy.ndarray
            The unit values.
        """
        return {
            f"{prefix}length": np.asarray(self.length, dtype=np.float64),
            f"{prefix}mass": np.asarray(self.mass, dtype=np.float64),
            f"{prefix}G": np.asarray(self.G, dtype=np.float64),
        }

    @classmethod
    def from_arrays(cls, arrays, prefix: str = "") -> "UnitSystem":
        """Rebuild from :meth:`to_arrays` output.

        Parameters
        ----------
        arrays : mapping of str to numpy.ndarray
            The stored arrays (e.g. an open ``.npz`` file).
        prefix : str, optional
            Prefix used when storing.

        Returns
        -------
        units : UnitSystem
            The reconstructed unit system.
        """
        return cls(
            length=float(arrays[f"{prefix}length"]),
            mass=float(arrays[f"{prefix}mass"]),
            G=float(arrays[f"{prefix}G"]),
        )
