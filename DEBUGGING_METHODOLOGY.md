# Debugging methodology

A template for debugging silent hangs / "nothing happens, no error" bugs, distilled
from a real case: the `rgbmnist_Flow_smoke_test` training run hanging indefinitely
after model construction, with 0% GPU utilization and no traceback.

Use this as a checklist when a process hangs without erroring. The worked example
below shows each step applied.

## The general method

1. **Reproduce, then hold everything else constant.** Don't change two things at
   once. Each retry should test exactly one hypothesis.
2. **Rule out the environment before suspecting the code.** Shared machines
   (GPU contention, external drives, network) produce symptoms that look
   exactly like code bugs. Check them first because they're cheap to rule out
   and easy to fix if real -- but don't stop there if ruling them out doesn't
   change the symptom.
3. **When ruling out an environment factor, verify the fix actually landed.**
   E.g. after "fixing" GPU contention, confirm `nvidia-smi` shows 0 compute
   processes -- don't just assume the kill worked.
4. **Isolate the suspect component outside its normal orchestration.** If a
   Hydra + Lightning + submitit pipeline hangs, write a 20-line script that
   constructs just the suspected class (e.g. a `Dataset`/`DataLoader`) directly
   and times it. This either confirms or clears that component in seconds,
   instead of waiting minutes per full-pipeline retry.
5. **A config override "not working" is itself a clue, not just a dead end.**
   If setting `x=0` doesn't change behavior, don't just try something else --
   ask whether `0` is being silently treated as "unset" somewhere (a classic
   Python truthiness bug: `if not x:` treats `0`, `""`, `[]`, `None` all the
   same).
6. **Get a real stack trace before generating more hypotheses.** Once you're
   past 2-3 ruled-out guesses, stop guessing and get direct evidence of where
   the process is actually stuck. See the `faulthandler` technique below --
   it needs no special permissions and pinpoints the exact frame.
7. **Fix the root cause, not the symptom.** A workaround (e.g. force
   `num_workers=1` in one config) would have "fixed" this instance but left
   the underlying bug (silently ignoring `num_workers=0` everywhere) in place
   for the next person.
8. **Verify with the full test suite AND the original repro**, then remove any
   temporary diagnostic code before calling it done.

## Getting a stack trace without `sudo`/`ptrace`

The normal tool for "what is this stuck process actually doing" is `py-spy dump
--pid <pid>`. It needs `ptrace` access, which typically requires `sudo`. If
passwordless `sudo` isn't configured, `py-spy` will just fail with a permission
error and there's no way to unblock it non-interactively.

The fallback that needs **no elevated permissions at all**, because the
process dumps its own state: Python's built-in
[`faulthandler`](https://docs.python.org/3/library/faulthandler.html) module.

Add this near the top of the entrypoint's `main()`, temporarily:

```python
import faulthandler

faulthandler.enable()
dump_file = open("/tmp/<something>_faulthandler_dump.log", "w")
faulthandler.dump_traceback_later(20, repeat=True, file=dump_file)
```

- `dump_traceback_later(20, repeat=True, ...)` dumps every thread's current
  Python stack every 20 seconds, for as long as the process runs, to the given
  file. No signal needs to be sent from outside -- it's a self-scheduled
  timer inside the process.
- Run the reproduction as normal. Once it's hanging, `cat` the dump file (or
  poll for it with `until [ -s <file> ]; do sleep 2; done` if you're not sure
  yet whether/when it'll hang).
- Each thread's entry is a normal Python traceback (`File "...", line N in
  <func>`), most-recent-call-first. Skim for the *main* thread's entry --
  background threads (wandb heartbeats, progress-bar renderers, DataLoader
  prefetch threads) are usually idle in some `wait()`/`select()` and are noise.
- Remove the whole diagnostic block once you have your answer -- it's not
  something that should ship.

This is generally applicable: any Python process that hangs (not just
Hydra/Lightning) can be diagnosed this way as long as you can add a couple of
lines before the hang and re-run it.

## Worked example: the rgbmnist_Flow_smoke_test hang

**Symptom:** `python train.py -cn experiment/rgbmnist_Flow_smoke_test/train
hydra/launcher=local` would build the model, connect to W&B, print the
Lightning model summary table -- then hang forever. 0% GPU utilization,
near-zero CPU growth, no error, no traceback.

**Attempt 1 -- GPU contention.** `nvidia-smi` showed a local vLLM server
(`Qwen3.6-35B`) holding 67.8GB on the same GPU. Plausible: our job might never
get a scheduling slice. Killed the vLLM server, confirmed via
`nvidia-smi --query-compute-apps` that 0 processes held GPU memory, reran the
smoke test. **Same hang, identical stopping point.** Ruled out.

**Attempt 2 -- external-drive I/O.** The MNIST data lived on an external LaCie
drive; slow/flaky external reads seemed like a reasonable next suspect. Copied
the data to local disk, repointed `DATA_ROOT` at the local copy, reran.
**Same hang again.** Ruled out.

**Isolation script.** Rather than keep guessing from `ps`/`nvidia-smi` output,
wrote a ~100-line standalone script
(outside Hydra/Lightning/submitit entirely) that:
- Directly constructed `RGBMNIST` + its augmentation pipeline and timed 5 raw
  `__getitem__()` calls.
- Wrapped it in a plain `torch.utils.data.DataLoader` with `num_workers=0`,
  timed fetching one batch.
- Same again with `num_workers=1`.
- Repeated the `num_workers=1` case both before and after initializing CUDA in
  the parent process, to test the "forking after CUDA init" hypothesis
  specifically.

Result: `__getitem__` was instant (~0.5ms), `num_workers=0` fetched a 256-image
batch in 0.3s, but **`num_workers=1` hung indefinitely in both CUDA
conditions** -- so it wasn't a CUDA-fork interaction, just forking a
DataLoader worker at all, on this machine.

**The confusing part.** Added `data.loader.num_workers: 0` as a config
override to route around it and reran the *actual* smoke test. **Still hung,
same spot.** This was the moment to stop generating more hypotheses and get
direct evidence, per step 6 above.

**faulthandler dump.** `py-spy` needed `sudo`, which wasn't available
non-interactively. Added the `faulthandler.dump_traceback_later()` snippet
above to `train.py`, reran, and read `/tmp/flower_faulthandler_dump.log` once
it had content. The main thread's dumped stack:

```
_run_sanity_check
  -> evaluation_loop.run
  -> CombinedLoader.__next__
  -> DataLoader.__next__
  -> _next_data -> _get_data -> _try_get_data
  -> multiprocessing.queues.get -> connection.poll -> selectors.select
```

`_try_get_data` / `multiprocessing.queues` is **only** reachable when
`num_workers > 0` -- a `num_workers=0` DataLoader uses an entirely different,
single-process iterator with no queue at all. So despite the config override,
the validation dataloader was still spinning up a forked worker. That's the
smoking gun: the override wasn't taking effect.

**Root cause.** `flower/data/modules.py`, `FlowerDataLoader.__init__`:

```python
if not num_workers:
    self.num_workers = 1
else:
    self.num_workers = num_workers
```

`not 0` is `True` in Python. Passing `num_workers=0` was silently treated
identically to "not specified" and coerced back to `1` -- every time,
regardless of config. Combined with the isolation test's finding that
`num_workers=1` hangs on this machine, this fully explains the persistent
hang: no config value could ever actually disable the forked worker.

**Fix:**

```python
if num_workers is None:
    self.num_workers = 1
else:
    self.num_workers = num_workers
```

**Verification:** removed the temporary `faulthandler` block from `train.py`,
ran the full test suite (148 passed, no regressions), reran the smoke test --
completed end-to-end in ~24 seconds with sane loss values and a successful
W&B run.

## Takeaways for next time

- A hang with 0% CPU/GPU utilization and no traceback is not "still working,
  just slow" -- treat it as a real block after a couple of minutes and start
  eliminating causes systematically, rather than waiting longer.
- Environment-level suspects (shared GPU, external drives, network) are worth
  checking early because they're common and easy to rule out, but don't let
  ruling them out become a substitute for finding the actual cause.
- An isolation script that bypasses the framework is almost always faster than
  more `ps`/`nvidia-smi` archaeology once you have a specific component in
  mind.
- If an override doesn't change behavior, suspect the override mechanism
  itself before suspecting your understanding of what should happen --
  falsy-value bugs (`if not x`) are a very common cause in Python configs.
- `faulthandler.dump_traceback_later()` is the go-to when `py-spy`/`gdb`
  aren't available (no sudo, no ptrace) -- it needs nothing but a couple of
  lines in the code you already control.
