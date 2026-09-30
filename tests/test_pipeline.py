"""End-to-end: the driver scripts, from a TOML configuration to figures."""

import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import asmodean as am  # noqa: E402

pytest.importorskip("lanfear")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, "scripts")

CONFIG = """
[potential]
type = "hernquist"
mass = 10.0
scale_radius = 1.0
n_particles = 20000
n_max = 8
l_max = 0

[friction]
enabled = true
source = "potential"
r_min = 1e-4
r_max = 50.0
central_mass = "bodies"

[[binary]]
labels = ["BH1", "BH2"]
masses = [0.01, 0.01]
semimajor_axis = 0.005
eccentricity = 0.3
softening = 1e-5

[[incoming]]
label = "BH3"
mass = 0.005
v_infinity = 50.0
impact_parameter = 0.003
distance = 0.1
orientation_deg = [30.0, 0.0, 0.0]
softening = 1e-5

[integration]
n_periods = 200
period = "binary"
n_samples = 2000
collision_distance = 1e-5
ejection_radius = 0.05
"""


def run(*args, cwd):
    env = dict(os.environ, MPLBACKEND="Agg")
    env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")
    out = subprocess.run(
        [sys.executable, *args], cwd=cwd, env=env, capture_output=True, text=True
    )
    assert out.returncode == 0, out.stderr
    return out.stdout


def test_scripts(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text(CONFIG)
    stdout = run(
        os.path.join(SCRIPTS, "make_system.py"),
        str(config),
        "-o",
        "setup.npz",
        cwd=tmp_path,
    )
    assert "3 bodies" in stdout
    setup = am.ScatteringSetup.load(tmp_path / "setup.npz")
    assert setup.potential.kind == "scf"
    assert setup.friction.mode == "system"
    assert setup.settings.period == "binary"

    stdout = run(
        os.path.join(SCRIPTS, "run_scattering.py"),
        "setup.npz",
        "-o",
        "result.npz",
        "--n-periods",
        "100",
        cwd=tmp_path,
    )
    assert "Outcome:" in stdout
    result = am.ScatteringResult.load(tmp_path / "result.npz")
    assert result.n_periods == 100
    assert abs(result.energy_error()).max() < 1e-6

    stdout = run(
        os.path.join(SCRIPTS, "analyse.py"),
        "result.npz",
        "--figdir",
        "figs",
        cwd=tmp_path,
    )
    for name in ("trajectories.png", "separations.png", "energy.png"):
        assert (tmp_path / "figs" / name).exists()


def test_example_config_parses(tmp_path):
    """The shipped example configuration builds a valid setup."""
    run(
        os.path.join(SCRIPTS, "make_system.py"),
        os.path.join(SCRIPTS, "example_config.toml"),
        "-o",
        str(tmp_path / "setup.npz"),
        cwd=tmp_path,
    )
    setup = am.ScatteringSetup.load(tmp_path / "setup.npz")
    assert setup.system.labels == ["BH1", "BH2", "BH3"]
