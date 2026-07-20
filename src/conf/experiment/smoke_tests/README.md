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
