"""Analyse a saved scattering result: outcome, final binaries and figures.

Prints the outcome and the final bound pairs, and writes figures of the
trajectories (projected on x-y, x-z and y-z, coloured by time), the pairwise separations, the energy-budget error, and the
orbital elements of the initially most bound pair.

    python scripts/analyse.py result.npz --figdir figures
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import asmodean as am


def plot_pair_elements(result, pair, path):
    """Semimajor axis and eccentricity of one pair against time.

    Parameters
    ----------
    result : asmodean.ScatteringResult
        The result.
    pair : asmodean.BoundPair
        The pair to follow.
    path : str
        Output figure path.
    """
    elements = am.pair_elements(result, pair.body_1, pair.body_2)
    a = np.array(
        [np.nan if e is None or not e.bound else e.semimajor_axis for e in elements]
    )
    ecc = np.array(
        [np.nan if e is None or not e.bound else e.eccentricity for e in elements]
    )
    fig, (ax_a, ax_e) = plt.subplots(2, 1, sharex=True, figsize=(7, 5))
    ax_a.plot(result.times, a, lw=0.8)
    ax_a.set_yscale("log")
    ax_a.set_ylabel("semimajor axis")
    ax_e.plot(result.times, ecc, lw=0.8)
    ax_e.set_ylabel("eccentricity")
    ax_e.set_xlabel("time")
    ax_a.set_title(f"{result.labels[pair.body_1]}-{result.labels[pair.body_2]}")
    fig.tight_layout()
    fig.savefig(path, dpi=300)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(type=str, help="result file from run_scattering.py", dest="file")
    ap.add_argument("--figdir", type=str, default="figures", help="figure directory")
    ap.add_argument(
        "--plane",
        choices=["xy", "xz", "yz", "all"],
        default="all",
        help="trajectory projection(s); 'all' plots x-y, x-z and y-z",
    )
    ap.add_argument(
        "--no-time-colour",
        action="store_true",
        help="colour trajectories by body instead of by time",
    )
    ap.add_argument(
        "--frame",
        choices=["inertial", "centre_of_mass"],
        default="inertial",
        help="frame of the trajectory plot",
    )
    args = ap.parse_args()
    os.makedirs(args.figdir, exist_ok=True)

    result = am.ScatteringResult.load(args.file)
    print(result.summary())
    outcome = am.classify_outcome(result)
    print(outcome.describe(result.labels))

    pairs = am.bound_pairs(result)
    print("Final bound pairs:" if pairs else "No bound pairs at the end")
    for p in pairs:
        print(
            f"  {result.labels[p.body_1]}-{result.labels[p.body_2]}: "
            f"a={p.semimajor_axis:.5g} e={p.eccentricity:.4f} "
            f"P={p.period:.5g} E_bind={p.binding_energy:.5g}"
        )
    for e in result.ejections:
        speed = np.sqrt(2 * e.value)
        print(f"  {result.labels[e.body]} escapes at ~{speed:.4g} (v_infinity)")

    ax = result.plot_trajectories(
        plane=args.plane, frame=args.frame, colour_by_time=not args.no_time_colour
    )
    np.atleast_1d(ax)[0].figure.savefig(
        os.path.join(args.figdir, "trajectories.png"), dpi=300
    )
    if result.n_bodies > 1:
        ax = result.plot_separations()
        ax.figure.savefig(os.path.join(args.figdir, "separations.png"), dpi=300)
    ax = result.plot_energy()
    ax.figure.savefig(os.path.join(args.figdir, "energy.png"), dpi=300)
    if outcome.initial_pair is not None:
        plot_pair_elements(
            result,
            outcome.initial_pair,
            os.path.join(args.figdir, "binary_elements.png"),
        )
    print(f"Figures written to {args.figdir}")


if __name__ == "__main__":
    main()
