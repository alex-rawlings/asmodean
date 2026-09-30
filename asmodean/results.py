"""The outcome of one scattering experiment: trajectories, events, energies.

A :class:`ScatteringResult` holds, in physical units, the uniformly sampled
trajectory of every body (NaN once a body has merged away or been ejected), the
list of :class:`Event` s (mergers and ejections), each body's final state, the
energy bookkeeping and why the integration stopped. It saves to and loads from
a ``.npz`` archive, and draws the basic diagnostic plots.

Energy bookkeeping: :attr:`ScatteringResult.energy` is the total energy of the
bodies still active (kinetic + external potential + softened pairwise),
:attr:`~ScatteringResult.energy_removed` the cumulative energy carried off by
mergers and ejections, and :attr:`~ScatteringResult.friction_work` the
cumulative work done by dynamical friction (negative). ``energy +
energy_removed - friction_work`` is conserved to integration accuracy; see
:meth:`ScatteringResult.energy_error`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt

from ._logging import get_logger
from .units import UnitSystem

logger = get_logger(__name__)

_RESULT_FORMAT = "asmodean-result-1"
# Mirrors of the C++ core's enumerations (index = code); defined here rather
# than read from asmodean._core so the docs build without the compiled module.
STOP_REASONS = [
    "time_limit",
    "all_merged",
    "bodies_ejected",
    "step_limit",
    "integration_error",
]
EVENT_TYPES = ["merger", "ejection"]
BODY_STATUSES = ["active", "merged", "ejected"]


@dataclass(frozen=True)
class Event:
    """A merger or an ejection.

    Parameters
    ----------
    time : float
        When it happened (physical time).
    kind : {"merger", "ejection"}
        The event type.
    body : int
        Merger: the surviving body (the more massive of the pair), which now
        carries the combined mass. Ejection: the ejected body.
    partner : int or None
        Merger: the body absorbed into ``body``. Ejection: None.
    mass : float
        Mass of ``body`` after the event.
    position : numpy.ndarray
        (3,) position of ``body`` just after the event (for a merger, the
        pair's centre of mass).
    velocity : numpy.ndarray
        (3,) velocity of ``body`` just after the event.
    value : float
        Merger: relative speed of the pair at contact (physical velocity).
        Ejection: the escape energy per unit mass that qualified it (positive;
        physical velocity^2), relative to the rest of the bodies or to the
        potential centre depending on the ejection reference.
    """

    time: float
    kind: str
    body: int
    partner: Optional[int]
    mass: float
    position: np.ndarray
    velocity: np.ndarray
    value: float


@dataclass
class ScatteringResult:
    """Trajectories and outcome of one scattering experiment (physical units).

    Not usually constructed directly; returned by :func:`asmodean.integrate` and
    :meth:`load`.

    Parameters
    ----------
    labels : list of str
        Body names.
    times : numpy.ndarray
        (n_out,) sample times.
    positions : numpy.ndarray
        (n_out, n, 3) positions (NaN for inactive bodies).
    velocities : numpy.ndarray
        (n_out, n, 3) velocities (NaN for inactive bodies).
    masses : numpy.ndarray
        (n_out, n) masses (NaN for inactive bodies; a merger survivor's grows).
    energy : numpy.ndarray
        (n_out,) total energy of the active bodies.
    energy_removed : numpy.ndarray
        (n_out,) cumulative energy carried off by mergers and ejections.
    friction_work : numpy.ndarray
        (n_out,) cumulative work done by dynamical friction.
    events : list of Event
        Mergers and ejections, in time order.
    status : list of str
        Final status of each body: ``"active"``, ``"merged"`` or ``"ejected"``.
    merged_into : numpy.ndarray
        (n,) index of the body each merged body was absorbed into (-1
        otherwise).
    final_time : numpy.ndarray
        (n,) time of each body's final state (``t_end`` if still active).
    final_masses : numpy.ndarray
        (n,) mass of each body in its final state.
    final_positions : numpy.ndarray
        (n, 3) position of each body in its final state (for a merged body,
        at contact; for an ejected one, at ejection).
    final_velocities : numpy.ndarray
        (n, 3) velocity of each body in its final state.
    stop_reason : str
        Why the integration stopped (one of :data:`STOP_REASONS`).
    t_end : float
        Time the integration stopped.
    n_steps : int
        Accepted integration steps.
    n_failed_steps : int
        Rejected (retried) integration steps.
    period : float
        The reference period the time limit was set from.
    n_periods : float
        The time limit in units of ``period``.
    units : UnitSystem
        The internal unit system the core integrated in.
    metadata : dict
        Provenance (integration settings, potential and friction options).
    """

    labels: List[str]
    times: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    masses: np.ndarray
    energy: np.ndarray
    energy_removed: np.ndarray
    friction_work: np.ndarray
    events: List[Event]
    status: List[str]
    merged_into: np.ndarray
    final_time: np.ndarray
    final_masses: np.ndarray
    final_positions: np.ndarray
    final_velocities: np.ndarray
    stop_reason: str
    t_end: float
    n_steps: int
    n_failed_steps: int
    period: float
    n_periods: float
    units: UnitSystem
    metadata: dict = field(default_factory=dict)

    # ------------------------------------------------------------ building
    @classmethod
    def from_core(
        cls,
        raw: dict,
        labels: List[str],
        units: UnitSystem,
        period: float,
        n_periods: float,
        metadata: Optional[dict] = None,
    ) -> "ScatteringResult":
        """Convert the C++ core's result dict to physical units.

        Parameters
        ----------
        raw : dict
            Output of ``asmodean._core.integrate``.
        labels : list of str
            Body names.
        units : UnitSystem
            The internal unit system of the integration.
        period : float
            Reference period (physical time).
        n_periods : float
            Time limit in units of ``period``.
        metadata : dict, optional
            Provenance to attach.

        Returns
        -------
        result : ScatteringResult
            The result in physical units.
        """
        L, V, T, M, E = (
            units.length,
            units.velocity,
            units.time,
            units.mass,
            units.energy,
        )
        states = np.asarray(raw["states"])
        events = []
        for k in range(len(raw["event_time"])):
            kind = EVENT_TYPES[int(raw["event_type"][k])]
            partner = int(raw["event_partner"][k])
            value = float(raw["event_value"][k])
            events.append(
                Event(
                    time=float(raw["event_time"][k]) * T,
                    kind=kind,
                    body=int(raw["event_body"][k]),
                    partner=partner if partner >= 0 else None,
                    mass=float(raw["event_mass"][k]) * M,
                    position=np.asarray(raw["event_state"][k, :3]) * L,
                    velocity=np.asarray(raw["event_state"][k, 3:]) * V,
                    value=value * (V if kind == "merger" else V**2),
                )
            )
        final_state = np.asarray(raw["final_state"])
        return cls(
            labels=list(labels),
            times=np.asarray(raw["times"]) * T,
            positions=states[:, :, :3] * L,
            velocities=states[:, :, 3:] * V,
            masses=np.asarray(raw["masses"]) * M,
            energy=np.asarray(raw["energy"]) * E,
            energy_removed=np.asarray(raw["energy_removed"]) * E,
            friction_work=np.asarray(raw["friction_work"]) * E,
            events=events,
            status=[BODY_STATUSES[int(s)] for s in raw["status"]],
            merged_into=np.asarray(raw["merged_into"], dtype=np.int64),
            final_time=np.asarray(raw["final_time"]) * T,
            final_masses=np.asarray(raw["final_mass"]) * M,
            final_positions=final_state[:, :3] * L,
            final_velocities=final_state[:, 3:] * V,
            stop_reason=STOP_REASONS[int(raw["stop_reason"])],
            t_end=float(raw["t_end"]) * T,
            n_steps=int(raw["n_steps"]),
            n_failed_steps=int(raw["n_failed_steps"]),
            period=float(period),
            n_periods=float(n_periods),
            units=units,
            metadata=dict(metadata or {}),
        )

    # ----------------------------------------------------------- accessors
    @property
    def n_bodies(self) -> int:
        """Number of bodies at the start.

        Returns
        -------
        n : int
            The initial body count.
        """
        return len(self.labels)

    @property
    def n_samples(self) -> int:
        """Number of recorded samples.

        Returns
        -------
        n : int
            Length of :attr:`times`.
        """
        return len(self.times)

    @property
    def active(self) -> np.ndarray:
        """Which bodies are active at each sample.

        Returns
        -------
        active : numpy.ndarray
            (n_out, n) boolean mask.
        """
        return np.isfinite(self.masses)

    @property
    def mergers(self) -> List[Event]:
        """The merger events.

        Returns
        -------
        events : list of Event
            Mergers, in time order.
        """
        return [e for e in self.events if e.kind == "merger"]

    @property
    def ejections(self) -> List[Event]:
        """The ejection events.

        Returns
        -------
        events : list of Event
            Ejections, in time order.
        """
        return [e for e in self.events if e.kind == "ejection"]

    @property
    def remaining(self) -> List[int]:
        """Indices of the bodies still active at the end.

        Returns
        -------
        indices : list of int
            The surviving bodies.
        """
        return [i for i, s in enumerate(self.status) if s == "active"]

    def index(self, body: Union[int, str]) -> int:
        """Index of a body given by index or label.

        Parameters
        ----------
        body : int or str
            Index or label.

        Returns
        -------
        index : int
            The body index.
        """
        return (
            int(body)
            if isinstance(body, (int, np.integer))
            else self.labels.index(body)
        )

    def separation(
        self, body_1: Union[int, str], body_2: Union[int, str]
    ) -> np.ndarray:
        """Distance between two bodies at every sample.

        Parameters
        ----------
        body_1, body_2 : int or str
            The bodies (index or label).

        Returns
        -------
        separation : numpy.ndarray
            (n_out,) distance; NaN where either is inactive.
        """
        i, j = self.index(body_1), self.index(body_2)
        return np.linalg.norm(self.positions[:, i] - self.positions[:, j], axis=1)

    def centre_of_mass(self) -> Tuple[np.ndarray, np.ndarray]:
        """Centre of mass of the active bodies at every sample.

        Returns
        -------
        position : numpy.ndarray
            (n_out, 3) centre-of-mass position.
        velocity : numpy.ndarray
            (n_out, 3) centre-of-mass velocity.
        """
        m = np.nan_to_num(self.masses)
        m_tot = np.sum(m, axis=1)[:, None]
        pos = np.sum(m[:, :, None] * np.nan_to_num(self.positions), axis=1) / m_tot
        vel = np.sum(m[:, :, None] * np.nan_to_num(self.velocities), axis=1) / m_tot
        return pos, vel

    def energy_error(self) -> np.ndarray:
        """Relative error of the conserved energy budget.

        Returns
        -------
        error : numpy.ndarray
            (n_out,) ``(E + E_removed - W_df - E_0) / |E_0|``.

        Notes
        -----
        The error is relative to the initial total energy. Close encounters
        reach kinetic energies far above ``|E_0|``, so a given relative
        accuracy per step shows up amplified by ``KE_peri / |E_0|`` here.
        """
        budget = self.energy + self.energy_removed - self.friction_work
        return (budget - budget[0]) / np.abs(self.energy[0])

    def summary(self) -> str:
        """Human-readable summary of the outcome.

        Returns
        -------
        text : str
            Multi-line description.
        """
        lines = [
            f"Scattering of {self.n_bodies} bodies: stopped at t={self.t_end:.6g} "
            f"({self.t_end / self.period:.4g} periods of {self.period:.4g}) "
            f"-- {self.stop_reason}",
            f"  {self.n_steps} steps ({self.n_failed_steps} rejected), "
            f"max |energy error| {np.max(np.abs(self.energy_error())):.2e}",
        ]
        for e in self.events:
            if e.kind == "merger":
                lines.append(
                    f"  t={e.time:.6g}: merger {self.labels[e.partner]} -> "
                    f"{self.labels[e.body]} (mass {e.mass:.4g}, contact speed "
                    f"{e.value:.4g})"
                )
            else:
                lines.append(
                    f"  t={e.time:.6g}: {self.labels[e.body]} ejected (mass "
                    f"{e.mass:.4g}, escape energy {e.value:.4g})"
                )
        lines.append(
            "  remaining: "
            + (", ".join(self.labels[i] for i in self.remaining) or "none")
        )
        return "\n".join(lines)

    # ----------------------------------------------------------------- io
    def save(self, path: Union[str, os.PathLike]) -> str:
        """Write the result to a ``.npz`` archive.

        Parameters
        ----------
        path : str or os.PathLike
            Destination (``.npz`` is appended by NumPy if absent).

        Returns
        -------
        path : str
            The path written.
        """
        n_ev = len(self.events)
        arrays = {
            "_format": np.asarray(_RESULT_FORMAT),
            "labels": np.asarray(self.labels),
            "times": self.times,
            "positions": self.positions,
            "velocities": self.velocities,
            "masses": self.masses,
            "energy": self.energy,
            "energy_removed": self.energy_removed,
            "friction_work": self.friction_work,
            "event_time": np.asarray([e.time for e in self.events], dtype=np.float64),
            "event_kind": np.asarray([e.kind for e in self.events], dtype="U8"),
            "event_body": np.asarray([e.body for e in self.events], dtype=np.int64),
            "event_partner": np.asarray(
                [-1 if e.partner is None else e.partner for e in self.events],
                dtype=np.int64,
            ),
            "event_mass": np.asarray([e.mass for e in self.events], dtype=np.float64),
            "event_position": np.asarray(
                [e.position for e in self.events], dtype=np.float64
            ).reshape(n_ev, 3),
            "event_velocity": np.asarray(
                [e.velocity for e in self.events], dtype=np.float64
            ).reshape(n_ev, 3),
            "event_value": np.asarray([e.value for e in self.events], dtype=np.float64),
            "status": np.asarray(self.status),
            "merged_into": self.merged_into,
            "final_time": self.final_time,
            "final_masses": self.final_masses,
            "final_positions": self.final_positions,
            "final_velocities": self.final_velocities,
            "stop_reason": np.asarray(self.stop_reason),
            "t_end": np.asarray(self.t_end),
            "n_steps": np.asarray(self.n_steps, dtype=np.int64),
            "n_failed_steps": np.asarray(self.n_failed_steps, dtype=np.int64),
            "period": np.asarray(self.period),
            "n_periods": np.asarray(self.n_periods),
        }
        arrays.update(self.units.to_arrays("units/"))
        for key, value in self.metadata.items():
            arrays[f"meta/{key}"] = np.asarray(value)
        np.savez(path, **arrays)
        out = os.fspath(path)
        out = out if out.endswith(".npz") else out + ".npz"
        logger.info(f"Wrote scattering result ({self.n_samples} samples) to {out}")
        return out

    @classmethod
    def load(cls, path: Union[str, os.PathLike]) -> "ScatteringResult":
        """Read a result written by :meth:`save`.

        Parameters
        ----------
        path : str or os.PathLike
            A ``.npz`` file written by :meth:`save`.

        Returns
        -------
        result : ScatteringResult
            The reconstructed result.

        Raises
        ------
        ValueError
            If the file is not a asmodean result.
        """
        with np.load(path, allow_pickle=False) as npz:
            if "_format" not in npz or str(npz["_format"]) != _RESULT_FORMAT:
                raise ValueError(f"{os.fspath(path)!r} is not a asmodean result file")
            events = [
                Event(
                    time=float(npz["event_time"][k]),
                    kind=str(npz["event_kind"][k]),
                    body=int(npz["event_body"][k]),
                    partner=(
                        int(npz["event_partner"][k])
                        if npz["event_partner"][k] >= 0
                        else None
                    ),
                    mass=float(npz["event_mass"][k]),
                    position=np.asarray(npz["event_position"][k]),
                    velocity=np.asarray(npz["event_velocity"][k]),
                    value=float(npz["event_value"][k]),
                )
                for k in range(len(npz["event_time"]))
            ]
            metadata = {}
            for key in npz.files:
                if key.startswith("meta/"):
                    value = npz[key]
                    metadata[key[5:]] = value.item() if value.ndim == 0 else value
            return cls(
                labels=[str(s) for s in npz["labels"]],
                times=npz["times"],
                positions=npz["positions"],
                velocities=npz["velocities"],
                masses=npz["masses"],
                energy=npz["energy"],
                energy_removed=npz["energy_removed"],
                friction_work=npz["friction_work"],
                events=events,
                status=[str(s) for s in npz["status"]],
                merged_into=npz["merged_into"],
                final_time=npz["final_time"],
                final_masses=npz["final_masses"],
                final_positions=npz["final_positions"],
                final_velocities=npz["final_velocities"],
                stop_reason=str(npz["stop_reason"]),
                t_end=float(npz["t_end"]),
                n_steps=int(npz["n_steps"]),
                n_failed_steps=int(npz["n_failed_steps"]),
                period=float(npz["period"]),
                n_periods=float(npz["n_periods"]),
                units=UnitSystem.from_arrays(npz, "units/"),
                metadata=metadata,
            )

    # ------------------------------------------------------------ plotting
    def plot_trajectories(
        self,
        plane: Union[str, Sequence[str]] = "xy",
        frame: str = "inertial",
        ax=None,
        mark_events: bool = True,
        colour_by_time: bool = True,
        cmap: str = "viridis",
    ):
        """Projected trajectories of every body, optionally coloured by time.

        With ``colour_by_time`` (default) each trajectory is drawn as a line
        whose colour follows the sample time, on one colour scale shared by
        every body and every panel (from the first to the last sample, shown
        by a single colourbar); the bodies are then told apart by the shape of
        the marker at their starting point. Otherwise each body gets its own
        solid colour.

        Parameters
        ----------
        plane : {"xy", "xz", "yz", "all"} or sequence of str, optional
            Projection plane(s): one plane, several (one panel each), or
            ``"all"`` for x-y, x-z and y-z side by side.
        frame : {"inertial", "centre_of_mass"}, optional
            Plot positions as integrated (relative to the potential centre), or
            relative to the instantaneous centre of mass of the active bodies.
        ax : matplotlib.axes.Axes or sequence of Axes, optional
            Axes to draw into, one per plane; a new figure is created if
            omitted.
        mark_events : bool, optional
            Mark mergers (star) and ejections (cross).
        colour_by_time : bool, optional
            Colour the lines by time (default) rather than by body.
        cmap : str, optional
            Colormap for the time colouring.

        Returns
        -------
        ax : matplotlib.axes.Axes or numpy.ndarray of Axes
            The axes drawn on: a single Axes when one plane was requested
            as a string, otherwise an array with one Axes per plane.

        Raises
        ------
        ValueError
            If a plane, ``frame`` or the number of axes is not recognised.
        """
        from matplotlib.collections import LineCollection
        from matplotlib.colors import Normalize

        indices = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
        single = isinstance(plane, str) and plane != "all"
        planes = (
            ["xy", "xz", "yz"]
            if plane == "all"
            else [plane]
            if isinstance(plane, str)
            else list(plane)
        )
        for p in planes:
            if p not in indices:
                raise ValueError(
                    f"plane must be one of {sorted(indices)} or 'all', got '{p}'"
                )
        if frame not in ("inertial", "centre_of_mass"):
            raise ValueError(
                f"frame must be 'inertial' or 'centre_of_mass', got '{frame}'"
            )

        pos = self.positions
        com = None
        if frame == "centre_of_mass":
            com, _ = self.centre_of_mass()
            pos = pos - com[:, None, :]

        if ax is None:
            _, axes = plt.subplots(
                1,
                len(planes),
                figsize=(4.5 * len(planes) + 1, 4.5),
                squeeze=False,
                layout="constrained",
            )
            axes = axes[0]
        else:
            axes = np.atleast_1d(np.asarray(ax, dtype=object)).ravel()
            if len(axes) != len(planes):
                raise ValueError(
                    f"got {len(axes)} axes for {len(planes)} plane(s) {planes}"
                )

        norm = Normalize(vmin=self.times[0], vmax=max(self.times[-1], self.times[0]))
        colourmap = plt.get_cmap(cmap)
        markers = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "h"]
        # Each segment joins consecutive samples at which the body is active,
        # coloured by the time at its midpoint.
        t_mid = 0.5 * (self.times[1:] + self.times[:-1])
        mapped = None
        for axis, p in zip(axes, planes):
            i, j = indices[p]
            for b in range(self.n_bodies):
                xy = pos[:, b][:, [i, j]]
                label = self.labels[b]
                if colour_by_time:
                    ok = np.all(np.isfinite(xy[1:]), axis=1) & np.all(
                        np.isfinite(xy[:-1]), axis=1
                    )
                    segments = np.stack([xy[:-1], xy[1:]], axis=1)[ok]
                    mapped = LineCollection(segments, cmap=colourmap, norm=norm, lw=0.8)
                    mapped.set_array(t_mid[ok])
                    axis.add_collection(mapped)
                    axis.plot(
                        xy[0, 0],
                        xy[0, 1],
                        markers[b % len(markers)],
                        color=colourmap(norm(self.times[0])),
                        mec="k",
                        ms=6,
                        ls="none",
                        label=label,
                    )
                else:
                    line = axis.plot(xy[:, 0], xy[:, 1], lw=0.8, label=label)[0]
                    axis.plot(xy[0, 0], xy[0, 1], "o", color=line.get_color(), ms=4)
            if mark_events:
                for e in self.events:
                    q = e.position
                    if com is not None:
                        # Last sample at or before the event, whose centre of
                        # mass still includes the merged/ejected body.
                        k = max(
                            np.searchsorted(self.times, e.time, side="right") - 1, 0
                        )
                        q = q - com[k]
                    axis.plot(
                        q[i], q[j], "*" if e.kind == "merger" else "x", color="k", ms=9
                    )
            axis.autoscale_view()
            axis.set_xlabel(f"${p[0]}$")
            axis.set_ylabel(f"${p[1]}$")
            axis.set_aspect("equal", adjustable="datalim")
        axes[0].legend(loc="best", fontsize="small")
        if colour_by_time and mapped is not None:
            fig = axes[0].figure
            fig.colorbar(mapped, ax=list(axes), label="time")
        return axes[0] if single else np.asarray(axes, dtype=object)

    def plot_separations(self, pairs=None, ax=None):
        """Pairwise separations against time.

        Parameters
        ----------
        pairs : sequence of tuple, optional
            ``(body_1, body_2)`` pairs (index or label); default every pair.
        ax : matplotlib.axes.Axes, optional
            Axes to draw into; a new figure is created if omitted.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes drawn on.
        """
        if pairs is None:
            pairs = [
                (a, b)
                for a in range(self.n_bodies)
                for b in range(a + 1, self.n_bodies)
            ]
        if ax is None:
            _, ax = plt.subplots(figsize=(7, 4))
        for a, b in pairs:
            ia, ib = self.index(a), self.index(b)
            ax.plot(
                self.times,
                self.separation(ia, ib),
                lw=0.8,
                label=f"{self.labels[ia]}-{self.labels[ib]}",
            )
        for e in self.events:
            ax.axvline(
                e.time, color="k", lw=0.5, ls="--" if e.kind == "merger" else ":"
            )
        ax.set_yscale("log")
        ax.set_xlabel("time")
        ax.set_ylabel("separation")
        ax.legend(loc="best", fontsize="small")
        return ax

    def plot_energy(self, ax=None):
        """Energy-budget error against time (the integration-accuracy check).

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw into; a new figure is created if omitted.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes drawn on.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(7, 4))
        err = np.abs(self.energy_error())
        ax.plot(self.times[1:], np.maximum(err[1:], 1e-17), lw=0.8)
        ax.set_yscale("log")
        ax.set_xlabel("time")
        ax.set_ylabel(r"$|\Delta E_{\rm budget}| / |E_0|$")
        return ax
