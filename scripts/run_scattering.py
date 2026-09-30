"""Integrate a scattering experiment saved by make_system.py and save the result.

Any integration setting in the setup can be overridden from the command line.

    python scripts/run_scattering.py setup.npz -o result.npz
    python scripts/run_scattering.py setup.npz -o result.npz --n-periods 1000 --no-friction
"""

import argparse

import asmodean as am


def period_arg(value):
    """Parse --period: a number (physical time) or an estimation method."""
    try:
        return float(value)
    except ValueError:
        return value


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(type=str, help="setup file from make_system.py", dest="setup")
    ap.add_argument(
        "-o", "--output", type=str, default="result.npz", help="result file"
    )
    ap.add_argument("--n-periods", type=float, help="time limit in periods")
    ap.add_argument(
        "--period",
        type=period_arg,
        help=f"reference period: a time, or one of {am.scattering.PERIOD_CHOICES}",
    )
    ap.add_argument("--n-samples", type=int, help="number of output samples")
    ap.add_argument("--collision-distance", type=float, help="merger distance")
    ap.add_argument("--ejection-radius", type=float, help="ejection distance")
    ap.add_argument(
        "--ejection-reference", choices=["system", "centre"], help="ejection reference"
    )
    ap.add_argument("--rel-tol", type=float, help="relative tolerance")
    ap.add_argument("--abs-tol", type=float, help="absolute tolerance (internal units)")
    ap.add_argument("--stepper", choices=["rkf78", "dopri5"], help="integrator")
    ap.add_argument("--no-friction", action="store_true", help="disable friction")
    ap.add_argument("-v", "--verbose", action="store_true", help="INFO logging")
    args = ap.parse_args()
    am.print_package_info()
    if args.verbose:
        am.set_verbosity("INFO")

    setup = am.ScatteringSetup.load(args.setup)
    if args.no_friction:
        setup.friction = None
    overrides = {
        name: getattr(args, name)
        for name in (
            "n_periods",
            "period",
            "n_samples",
            "collision_distance",
            "ejection_radius",
            "ejection_reference",
            "rel_tol",
            "abs_tol",
            "stepper",
        )
        if getattr(args, name) is not None
    }
    result = setup.run(**overrides)
    print(result.summary())
    print(am.classify_outcome(result).describe(result.labels))
    path = result.save(args.output)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
