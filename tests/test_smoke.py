"""End-to-end smoke tests that actually train against real data/checkpoints.

Excluded from the default `pytest tests/` run (see the `smoke` marker in
pyproject.toml) -- run explicitly with `pytest -m smoke` on a machine with
DATA_ROOT set and local_data populated (see
src/conf/experiment/smoke_tests/README.md). Failures here are meant to be
loud: if the environment isn't set up, the test fails with the captured
train.py output rather than skipping quietly.
"""

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT_DIR = REPO_ROOT / "src" / "flower" / "training"

SENTINEL_LINES = (
    "Base model weights loaded.",
    "Base model frozen.",
    "Model fitting completed.",
    "Model testing completed.",
)

MIN_CKPT_BYTES = 1_000_000  # a real model checkpoint, not a stub/empty file


def _run_smoke_test(
    experiment_name: str, tmp_path: Path
) -> tuple[subprocess.CompletedProcess, Path]:
    experiment_path = tmp_path / experiment_name

    result = subprocess.run(
        [
            sys.executable,
            "train.py",
            "-cn",
            f"experiment/smoke_tests/{experiment_name}/train",
            "hydra/launcher=local",
            "logger.wandb.mode=offline",
            f"paths.experiment_path={experiment_path}",
        ],
        cwd=TRAIN_SCRIPT_DIR,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    return result, experiment_path


def _job_log_text(experiment_path: Path) -> str:
    log_files = list(experiment_path.glob("multiruns/*/.submitit/*/*_log.out"))
    assert log_files, f"No submitit job log found under {experiment_path}"
    return "\n".join(f.read_text() for f in log_files)


def _assert_smoke_test_succeeded(
    result: subprocess.CompletedProcess, experiment_path: Path
) -> None:
    assert result.returncode == 0, (
        f"train.py exited with {result.returncode}.\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )

    job_log = _job_log_text(experiment_path)
    for sentinel in SENTINEL_LINES:
        assert sentinel in job_log, (
            f"Expected {sentinel!r} in the training job's log, not found.\n"
            f"--- job log ---\n{job_log}"
        )

    ckpts = list(experiment_path.glob("ckpts/*.ckpt"))
    assert len(ckpts) == 1, (
        f"Expected exactly one checkpoint under {experiment_path / 'ckpts'}, "
        f"found {ckpts}"
    )
    ckpt_size = ckpts[0].stat().st_size
    assert ckpt_size > MIN_CKPT_BYTES, (
        f"Checkpoint {ckpts[0]} is only {ckpt_size} bytes -- looks empty/truncated"
    )


def test_rgbmnist_flow_smoke(tmp_path):
    result, experiment_path = _run_smoke_test("rgbmnist_Flow_smoke_test", tmp_path)
    _assert_smoke_test_succeeded(result, experiment_path)


def test_dsprites_flow_smoke(tmp_path):
    result, experiment_path = _run_smoke_test("dsprites_Flow_smoke_test", tmp_path)
    _assert_smoke_test_succeeded(result, experiment_path)
