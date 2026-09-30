"""Chandrasekhar dynamical friction on the bodies.

The drag is Chandrasekhar's (1943) formula for a mass ``M`` moving at
velocity ``v`` through a static, isotropic Maxwellian background of density
``rho`` and one-dimensional dispersion ``sigma`` (Binney & Tremaine 2008,
eq. 8.7)::

    a_df = -4 pi G^2 M rho lnL / v^3 [erf(X) - 2X/sqrt(pi) exp(-X^2)] v,
    X = v / (sqrt(2) sigma).

``rho(r)`` and ``sigma(r)`` are radial profiles about the potential centre,
built here in one of three ways:

* :meth:`DynamicalFriction.from_potential` -- from the analytical potential
  alone: the spherically averaged enclosed mass (Gauss's theorem) gives
  ``rho``, and the isotropic Jeans equation gives ``sigma``;
* :meth:`DynamicalFriction.from_particles` -- binned from simulation particles
  (e.g. a lanfear snapshot's field particles);
* :meth:`DynamicalFriction.constant` -- a uniform background.

``mode`` sets how the drag acts on several bodies:

* ``"system"`` (default): the bodies are treated as one object. The drag force
  on their combined mass, moving with their centre-of-mass velocity at their
  centre of mass, is distributed among them in proportion to their masses --
  each body feels the same deceleration, which slows the group as a whole
  without disturbing its internal motion;
* ``"individual"``: each body feels the drag of its own mass and velocity.

The Coulomb logarithm is a constant, or ``"variable"`` for
``lnL = 0.5 ln(1 + Lambda^2)`` with ``Lambda = r (v^2 + sigma^2) / (G M)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from . import _core
from ._logging import get_logger
from .units import UnitSystem

logger = get_logger(__name__)

_MODES = {"system": 0, "individual": 1}


@dataclass
class DynamicalFriction:
    """Chandrasekhar dynamical friction from tabulated background profiles.

    Profiles are interpolated linearly in log-log and held at their end values
    outside the tabulated range.

    Parameters
    ----------
    radius : numpy.ndarray
        (k,) strictly increasing, positive radii (physical length).
    density : numpy.ndarray
        (k,) background density at ``radius`` (physical mass / length^3).
    dispersion : numpy.ndarray
        (k,) background 1D velocity dispersion at ``radius`` (physical velocity).
    coulomb_logarithm : float or str, optional
        Constant Coulomb logarithm (default 3), or ``"variable"``.
    mode : {"system", "individual"}, optional
        How the drag is applied to several bodies (see the module docstring).
    """

    radius: np.ndarray
    density: np.ndarray
    dispersion: np.ndarray
    coulomb_logarithm: Union[float, str] = 3.0
    mode: str = "system"

    def __post_init__(self) -> None:
        """Validate and normalise the profile arrays.

        Raises
        ------
        ValueError
            If the profiles or options are invalid.
        """
        self.radius = np.atleast_1d(np.asarray(self.radius, dtype=np.float64))
        self.density = np.atleast_1d(np.asarray(self.density, dtype=np.float64))
        self.dispersion = np.atleast_1d(np.asarray(self.dispersion, dtype=np.float64))
        if not (self.radius.shape == self.density.shape == self.dispersion.shape):
            raise ValueError("radius, density and dispersion must have equal shapes")
        if len(self.radius) == 0:
            raise ValueError("friction profiles must be non-empty")
        if np.any(np.diff(self.radius) <= 0) or np.any(self.radius <= 0):
            raise ValueError("radius must be positive and strictly increasing")
        if np.any(self.density <= 0) or np.any(self.dispersion <= 0):
            raise ValueError("density and dispersion must be positive")
        if self.mode not in _MODES:
            raise ValueError(f"mode must be one of {sorted(_MODES)}, got '{self.mode}'")
        if isinstance(self.coulomb_logarithm, str):
            if self.coulomb_logarithm != "variable":
                raise ValueError(
                    f"coulomb_logarithm must be a number or 'variable', got "
                    f"'{self.coulomb_logarithm}'"
                )
        elif not (self.coulomb_logarithm >= 0):
            raise ValueError("coulomb_logarithm must be non-negative")

    # ------------------------------------------------------------- builders
    @classmethod
    def constant(
        cls,
        density: float,
        dispersion: float,
        coulomb_logarithm: Union[float, str] = 3.0,
        mode: str = "system",
    ) -> "DynamicalFriction":
        """A uniform background.

        Parameters
        ----------
        density : float
            Background density (physical units).
        dispersion : float
            Background 1D velocity dispersion (physical units).
        coulomb_logarithm : float or str, optional
            Constant Coulomb logarithm, or ``"variable"``.
        mode : {"system", "individual"}, optional
            How the drag is applied to several bodies.

        Returns
        -------
        friction : DynamicalFriction
            The friction model.
        """
        return cls(
            radius=np.array([1.0]),
            density=np.array([float(density)]),
            dispersion=np.array([float(dispersion)]),
            coulomb_logarithm=coulomb_logarithm,
            mode=mode,
        )

    @classmethod
    def from_particles(
        cls,
        positions,
        velocities,
        masses,
        n_bins: int = 50,
        r_min: Optional[float] = None,
        r_max: Optional[float] = None,
        centre=(0.0, 0.0, 0.0),
        min_particles: int = 20,
        coulomb_logarithm: Union[float, str] = 3.0,
        mode: str = "system",
    ) -> "DynamicalFriction":
        """Spherically binned density and dispersion of simulation particles.

        Parameters
        ----------
        positions : array-like of float
            (N, 3) particle positions (physical units, same frame as the
            potential).
        velocities : array-like of float
            (N, 3) particle velocities.
        masses : array-like of float
            (N,) particle masses.
        n_bins : int, optional
            Number of logarithmic radial bins.
        r_min, r_max : float, optional
            Radial range; default the 0.1 and 99.9 percentiles of the particle
            radii.
        centre : array-like of float, optional
            (3,) centre of the profiles (the potential centre, default origin).
        min_particles : int, optional
            Bins with fewer particles are dropped.
        coulomb_logarithm : float or str, optional
            Constant Coulomb logarithm, or ``"variable"``.
        mode : {"system", "individual"}, optional
            How the drag is applied to several bodies.

        Returns
        -------
        friction : DynamicalFriction
            The friction model.

        Raises
        ------
        ValueError
            If no bin has enough particles.
        """
        pos = np.asarray(positions, dtype=np.float64) - np.asarray(centre)
        vel = np.asarray(velocities, dtype=np.float64)
        mass = np.asarray(masses, dtype=np.float64)
        r = np.linalg.norm(pos, axis=1)
        r_min = float(np.percentile(r, 0.1)) if r_min is None else float(r_min)
        r_max = float(np.percentile(r, 99.9)) if r_max is None else float(r_max)
        r_min = max(r_min, 1e-12 * r_max)
        edges = np.geomspace(r_min, r_max, n_bins + 1)
        which = np.digitize(r, edges) - 1
        radius, density, dispersion = [], [], []
        for k in range(n_bins):
            sel = which == k
            if np.count_nonzero(sel) < min_particles:
                continue
            m = mass[sel]
            m_sum = np.sum(m)
            v_mean = np.sum(m[:, None] * vel[sel], axis=0) / m_sum
            v2 = np.sum(m * np.sum((vel[sel] - v_mean) ** 2, axis=1)) / m_sum
            volume = 4.0 / 3.0 * np.pi * (edges[k + 1] ** 3 - edges[k] ** 3)
            radius.append(np.sqrt(edges[k] * edges[k + 1]))
            density.append(m_sum / volume)
            dispersion.append(np.sqrt(v2 / 3.0))
        if not radius:
            raise ValueError("no radial bin holds enough particles for a profile")
        logger.info(
            f"Dynamical-friction profiles from {len(mass)} particles: "
            f"{len(radius)} bins in [{r_min:.3g}, {r_max:.3g}]"
        )
        return cls(
            np.array(radius),
            np.array(density),
            np.array(dispersion),
            coulomb_logarithm=coulomb_logarithm,
            mode=mode,
        )

    @classmethod
    def from_particle_system(cls, particles, **kwargs) -> "DynamicalFriction":
        """Profiles from the field particles of a lanfear ``ParticleSystem``.

        Parameters
        ----------
        particles : lanfear.ParticleSystem
            A prepared system; its ``field`` (everything except black holes) is
            binned.
        **kwargs
            Forwarded to :meth:`from_particles`.

        Returns
        -------
        friction : DynamicalFriction
            The friction model.
        """
        field_particles = particles.field
        return cls.from_particles(
            field_particles.pos, field_particles.vel, field_particles.mass, **kwargs
        )

    @classmethod
    def from_potential(
        cls,
        potential,
        r_min: float,
        r_max: float,
        n_bins: int = 64,
        n_directions: int = 128,
        outer_factor: float = 1000.0,
        central_mass: float = 0.0,
        coulomb_logarithm: Union[float, str] = 3.0,
        mode: str = "system",
    ) -> "DynamicalFriction":
        """Profiles implied by the analytical potential (isotropic Jeans model).

        The spherically averaged enclosed mass follows from Gauss's theorem,
        ``M(<r) = -r^2 <a_r> / G``, and gives ``rho = (dM/dr) / (4 pi r^2)``.
        The isotropic Jeans equation then gives the dispersion::

            rho sigma^2 (r) = int_r^inf rho(r') (-<a_r>(r')) dr',

        integrated out to ``outer_factor * r_max``, beyond which the
        background is assumed negligible.

        Point masses attached to the potential enter ``M(<r)`` (and so
        ``sigma``) but add density only inside their softening length.

        When massive bodies (e.g. black holes integrated as bodies, and so
        absent from the potential) sit at the centre, pass their mass as
        ``central_mass``: it deepens the Jeans equation's potential well, so
        ``sigma`` near them rises as ``sqrt(G M / r)`` as it does around a real
        black hole. Without it the dispersion of a cuspy background falls to
        zero at the centre, and the low-velocity drag rate (``~ rho /
        sigma^3``) grows so large that the equations of motion become stiff.

        Parameters
        ----------
        potential : ExternalPotential
            The background potential (not :meth:`~ExternalPotential.none`).
        r_min, r_max : float
            Radial range of the tabulated profiles (physical length).
        n_bins : int, optional
            Number of logarithmic radii in ``[r_min, r_max]``.
        n_directions : int, optional
            Directions averaged over at each radius.
        outer_factor : float, optional
            The Jeans integral runs to ``outer_factor * r_max``.
        central_mass : float, optional
            Extra point mass at the centre (physical units) included in the
            Jeans equation (not in the density).
        coulomb_logarithm : float or str, optional
            Constant Coulomb logarithm, or ``"variable"``.
        mode : {"system", "individual"}, optional
            How the drag is applied to several bodies.

        Returns
        -------
        friction : DynamicalFriction
            The friction model.

        Raises
        ------
        ValueError
            If the potential is the null potential.
        """
        if potential.is_null:
            raise ValueError("cannot derive friction profiles without a potential")
        # One log-spaced grid: the tabulated range, then on out to
        # outer_factor * r_max for the Jeans integral.
        dlog = np.log(r_max / r_min) / (n_bins - 1)
        n_total = n_bins + int(np.ceil(np.log(outer_factor) / dlog))
        r = r_min * np.exp(dlog * np.arange(n_total))
        a_r = potential.radial_acceleration(r, n_directions)
        mass = -(r**2) * a_r / potential.units.G
        dm_dlnr = np.gradient(mass, np.log(r))
        rho = dm_dlnr / (4.0 * np.pi * r**3)
        n_bad = int(np.count_nonzero(rho[:n_bins] <= 0))
        if n_bad:
            logger.warning(
                f"{n_bad} non-positive density values from the potential were "
                f"clipped (noisy or truncated expansion?)"
            )
        floor = np.min(rho[rho > 0]) if np.any(rho > 0) else 1e-300
        rho = np.clip(rho, floor, None)
        # rho sigma^2 = int_r^inf rho (-a_r) dr = int rho (-a_r) r dlnr, cumulative
        # trapezoid from the outermost radius inwards.
        g_r = -a_r + potential.units.G * central_mass / r**2  # inward gravity
        integrand = rho * g_r * r
        seg = 0.5 * (integrand[1:] + integrand[:-1]) * np.diff(np.log(r))
        pressure = np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]])
        sigma = np.sqrt(np.clip(pressure[:n_bins] / rho[:n_bins], 0.0, None))
        sigma = np.clip(sigma, np.max(sigma) * 1e-6 if np.any(sigma > 0) else 1.0, None)
        logger.info(
            f"Dynamical-friction profiles from the potential (isotropic Jeans): "
            f"{n_bins} radii in [{r_min:.3g}, {r_max:.3g}]"
        )
        return cls(
            r[:n_bins],
            rho[:n_bins],
            sigma,
            coulomb_logarithm=coulomb_logarithm,
            mode=mode,
        )

    # ------------------------------------------------------------ core
    def to_core(self, units: UnitSystem):
        """The C++ friction model in internal units.

        Parameters
        ----------
        units : UnitSystem
            The internal unit system.

        Returns
        -------
        friction : asmodean._core.DynamicalFriction
            The C++ object.
        """
        variable = isinstance(self.coulomb_logarithm, str)
        return _core.DynamicalFriction(
            self.radius / units.length,
            self.density / units.density,
            self.dispersion / units.velocity,
            0.0 if variable else float(self.coulomb_logarithm),
            variable,
            _MODES[self.mode],
        )

    def acceleration(self, mass: float, position, velocity, units: UnitSystem):
        """Chandrasekhar deceleration of one mass (physical units).

        Parameters
        ----------
        mass : float
            The mass.
        position : array-like of float
            (3,) position.
        velocity : array-like of float
            (3,) velocity.
        units : UnitSystem
            Unit system supplying G (any length/mass scale gives the same
            answer).

        Returns
        -------
        acc : numpy.ndarray
            (3,) acceleration.
        """
        core = self.to_core(units)
        a = core.acceleration(
            mass / units.mass,
            list(np.asarray(position, dtype=np.float64) / units.length),
            list(np.asarray(velocity, dtype=np.float64) / units.velocity),
        )
        return np.asarray(a) * units.acceleration

    # ----------------------------------------------------------------- io
    def to_arrays(self, prefix: str = "friction/") -> dict:
        """Serialise to a dict of NumPy arrays (for ``.npz`` files).

        Parameters
        ----------
        prefix : str, optional
            Prefix for every key.

        Returns
        -------
        arrays : dict of str to numpy.ndarray
            The profiles and options.
        """
        return {
            f"{prefix}radius": self.radius,
            f"{prefix}density": self.density,
            f"{prefix}dispersion": self.dispersion,
            f"{prefix}coulomb_logarithm": np.asarray(str(self.coulomb_logarithm)),
            f"{prefix}mode": np.asarray(self.mode),
        }

    @classmethod
    def from_arrays(cls, arrays, prefix: str = "friction/") -> "DynamicalFriction":
        """Rebuild from :meth:`to_arrays` output.

        Parameters
        ----------
        arrays : mapping of str to numpy.ndarray
            The stored arrays.
        prefix : str, optional
            Prefix used when storing.

        Returns
        -------
        friction : DynamicalFriction
            The friction model.
        """
        lnl = str(arrays[f"{prefix}coulomb_logarithm"])
        return cls(
            np.asarray(arrays[f"{prefix}radius"]),
            np.asarray(arrays[f"{prefix}density"]),
            np.asarray(arrays[f"{prefix}dispersion"]),
            coulomb_logarithm=lnl if lnl == "variable" else float(lnl),
            mode=str(arrays[f"{prefix}mode"]),
        )
