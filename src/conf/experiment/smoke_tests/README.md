# Smoke tests

Cut-down versions of the full experiment configs (fewer epochs, fewer
batches, `num_workers: 0`) used to verify a training pipeline runs
end-to-end -- model construction, pretrained weight loading, a few
training/validation/test steps, and checkpoint saving -- in well under a
minute, without needing a full training run.

## rgbmnist_Flow_smoke_test

Verifies `rgbmnist_Flow`: loads the pretrained VAE checkpoint
(`5252446_0.ckpt`), freezes it, trains the flow-matching model for 2 epochs
over 3 batches, validates over 2 batches, tests, and saves a checkpoint.

1. Activate the environment and load `.env` (needed for `DATA_ROOT` and W&B
   credentials):

   ```bash
   cd ~/Desktop/flower
   source .venv/bin/activate
   export $(grep -v '^#' .env | xargs)
   ```

2. Run the smoke test:

   ```bash
   cd src/flower/training
   python train.py -cn "experiment/smoke_tests/rgbmnist_Flow_smoke_test/train" hydra/launcher=local
   ```

3. Verify it worked:

   - Console/log output should show `✅ Base model weights loaded.` and
     `❄️ Base model frozen.` right after the lightning loader is
     instantiated, then `Model fitting completed.` and
     `Model testing completed.`.
   - A checkpoint should exist at
     `$DATA_ROOT/rgbmnist/rgbmnist_Flow_smoke_test/ckpts/<job_id>.ckpt`
     (the `<job_id>` is printed in the Hydra sweep output, e.g.
     `+seed=42` job `#0`).
   - Full run should complete in well under a minute (~20s typical).

## dsprites_Flow_smoke_test

Verifies `dsprites_Flow`: loads the pretrained BetaVAE checkpoint
(`7586427.ckpt`), freezes it, trains the flow-matching model for 2 epochs
over 3 batches, validates over 2 batches, tests, and saves a checkpoint.

1. Activate the environment and load `.env` (same as above):

   ```bash
   cd ~/Desktop/flower
   source .venv/bin/activate
   export $(grep -v '^#' .env | xargs)
   ```

2. Run the smoke test:

   ```bash
   cd src/flower/training
   python train.py -cn "experiment/smoke_tests/dsprites_Flow_smoke_test/train" hydra/launcher=local
   ```

3. Verify it worked:

   - Console/log output should show `✅ Base model weights loaded.` and
     `❄️ Base model frozen.` right after the lightning loader is
     instantiated, then `Model fitting completed.` and
     `Model testing completed.`.
   - A checkpoint should exist at
     `$DATA_ROOT/dsprites/dsprites_Flow_smoke_test/ckpts/<job_id>.ckpt`
     (the `<job_id>` is printed in the Hydra sweep output, e.g. `+seed=42`
     job `#0`).
   - Full run should complete in well under a couple of minutes (~85s
     typical -- slower than the rgbmnist smoke test because
     `on_test_epoch_end` fits sklearn classifiers/regressors on the
     collected latents).

## Verifying the output

Three checks, in increasing order of how specific the failure signal is:

1. **Exit status.** `train.py` wraps every stage in try/except that
   re-raises with context, and the submitit launcher marks the job failed
   on any exception -- a non-zero exit / `Error executing job` in the log
   is unambiguous. This is the cheapest check and catches most breakage
   (e.g. a bad `_target_` path, a shape mismatch on checkpoint load).
2. **Sentinel log lines.** `✅ Base model weights loaded.` and
   `❄️ Base model frozen.` (proves the pretrained state_dict loaded with
   `strict=True` against the current model code) followed by
   `Model fitting completed.` / `Model testing completed.` (proves the
   full train/val/test loop ran without a Lightning-level exception).
3. **Checkpoint existence and size.** A `.ckpt` file at the exact
   `${paths.experiment_path}/ckpts/<job_id>.ckpt` path is the thing
   actually being asserted by "saves as expected"; a 0-byte or missing
   file would flag a silent write failure even if the process exited 0.

The manual steps above are still useful for quick iteration/eyeballing, but
this is also automated -- see `tests/test_smoke.py`.

## Automated pytest test

`tests/test_smoke.py` runs both smoke tests via `subprocess` (the real
`train.py` CLI entrypoint, not an in-process shortcut) and asserts all
three checks above programmatically:

- `result.returncode == 0`
- the three sentinel strings (`Base model weights loaded.`,
  `Base model frozen.`, `Model fitting completed.`,
  `Model testing completed.`) all appear in the job's own log file
  (`<experiment_path>/multiruns/*/.submitit/*/*_log.out` -- the job's log
  output doesn't get forwarded to the parent process's stdout, so it has to
  be read from disk, not from the captured `subprocess.run` output)
- exactly one `.ckpt` file exists under `<experiment_path>/ckpts/`, above a
  minimum size threshold (catches empty/truncated writes even if the
  process exited 0)

**Isolation**: each test overrides `paths.experiment_path` to a
`tmp_path`-based directory via a Hydra CLI override. Since `ckpt_dir`,
`hydra.sweep.dir` (and therefore the `.submitit` job logs), the CSV
metrics dir, and the wandb save dir all derive from
`${paths.experiment_path}`, this one override fully isolates every
*output* artifact from the real `local_data` tree, while `paths.data_dir`
(hence the real dataset files and `meta.vae_ckpt_path`, which must point at
a real pretrained checkpoint) is left untouched -- there's no way to fake
those inputs. `logger.wandb.mode=offline` is also forced, so running the
test doesn't create real runs in the W&B project.

**CI / default run**: these tests need real data, a real pretrained
checkpoint, and take 20s-90s each, so they're marked `@pytest.mark.smoke`
and excluded from the default `pytest tests/` run via `-m "not smoke"` in
`pyproject.toml`'s `addopts` -- the existing ~3s no-GPU/no-data suite
(and CI) is untouched. Run them explicitly:

```bash
cd ~/Desktop/flower
source .venv/bin/activate
export $(grep -v '^#' .env | xargs)
pytest tests/test_smoke.py -m smoke -v
```

**On missing preconditions** (no `DATA_ROOT`, no checkpoint file, no data
on disk): the test fails loudly rather than skipping -- the assertion
message includes the full captured `stdout`/`stderr` from `train.py`, so
the real Hydra/instantiation error is visible directly in the pytest
failure output.
