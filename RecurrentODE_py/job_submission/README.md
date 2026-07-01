# RecurrentODE_py job submission

SLURM submission for every Python module in `RecurrentODE_py/`. All
per-run settings (module, data-generating setting, knots, N grid, CI
flag, seed layout) live in a single JSON config — there is no need to
type them on the command line.

## Files

| file | role |
|---|---|
| `configs.json`   | Named configurations + defaults. **Edit here to change settings.** |
| `run_ode.sh`     | SBATCH script. Reads `configs.json` via `jq` and runs one config. |
| `submit_all.sh`  | Helper that `sbatch`es many configs in a loop. |

## Prerequisites (on the cluster)

* `jq` on the `$PATH` (standard; if missing, `module load jq` or
  `apt install jq` / `conda install jq`).
* A Python environment with `numpy`, `scipy`, etc. The current
  `run_ode.sh` runs `module load python/3.10-anaconda` — change that
  line if your cluster uses a different module name, or replace it
  with `conda activate <env>`.
* The repo checked out so that `RecurrentODE_py/` is importable from
  the working directory of the job. By default `run_ode.sh` does
  `cd "$REPO_ROOT"` where `REPO_ROOT` defaults to two directories
  above the script (i.e. the parent of `RecurrentODE_py/`). Override
  with `--export=REPO_ROOT=/path/to/checkout` if you want to run from
  a different checkout.

## `configs.json` layout

```json
{
  "defaults": {
    "n_values": [1000, 2000],
    "knots": "",
    "ci": 1,
    "seeds_per_task": 50,
    "seed_offset": 0
  },
  "configs": {
    "cox":    { "module": "cox",                 "data_setting": 1 },
    "aft":    { "module": "aft",                 "data_setting": 2, "knots": "quantile" },
    "ltm":    { "module": "ltm",                 "data_setting": 4, "knots": "K4" },
    "npmle":  { "module": "npmle",               "data_setting": 3, "knots": "equal",
                "n_values": [2000, 4000] },
    "re_cox": { "module": "random_effect.cox",   "data_setting": 1 },
    "re_aft": { "module": "random_effect.aft_rec","data_setting": 2, "knots": "quantile" },
    "re_ltm": { "module": "random_effect.ltm",   "data_setting": 1, "knots": "K4" }
  }
}
```

Resolution rule for every field: the per-config value wins, otherwise
the `defaults` value, otherwise a built-in fallback in `run_ode.sh`.

Recognised per-config fields:

| field            | type            | meaning                                         |
|------------------|-----------------|-------------------------------------------------|
| `module`         | string (req.)   | Dotted path under `RecurrentODE_py` (e.g. `cox`, `random_effect.ltm`). |
| `data_setting`   | integer (req.)  | Data-generating setting passed to `main(...)`.  |
| `knots`          | string          | `"K4"`, `"quantile"`, `"equal"`, or `""` to omit. |
| `n_values`       | array of int    | Sample sizes to run sequentially.               |
| `ci`             | 0 or 1          | Compute sandwich SE (passed to `main`).         |
| `seeds_per_task` | integer         | Seeds handled by one SLURM array task.          |
| `seed_offset`    | integer         | First seed − 1 (so default 0 ⇒ seeds start at 1).|

With `--array=1-20` (hard-coded in `run_ode.sh`) and
`seeds_per_task=50`, a submission covers seeds **1..1000**. Change
either number to cover more/fewer seeds.

## Submitting jobs

Run all commands from the repo root (the directory that contains
`RecurrentODE_py/`).

### 1. Submit one config

```bash
sbatch --export=CONFIG=cox    RecurrentODE_py/job_submission/run_ode.sh
sbatch --export=CONFIG=aft    RecurrentODE_py/job_submission/run_ode.sh
sbatch --export=CONFIG=ltm    RecurrentODE_py/job_submission/run_ode.sh
sbatch --export=CONFIG=npmle  RecurrentODE_py/job_submission/run_ode.sh
sbatch --export=CONFIG=re_cox RecurrentODE_py/job_submission/run_ode.sh
sbatch --export=CONFIG=re_aft RecurrentODE_py/job_submission/run_ode.sh
sbatch --export=CONFIG=re_ltm RecurrentODE_py/job_submission/run_ode.sh
```

### 2. Submit every config at once

```bash
bash RecurrentODE_py/job_submission/submit_all.sh
```

### 3. Submit a subset

```bash
bash RecurrentODE_py/job_submission/submit_all.sh cox ltm npmle
```

### 4. One-off overrides (no JSON edit)

You can still override individual variables from the command line;
they take precedence over the JSON values for that submission only.

```bash
# smaller pilot run — 5 seeds per task, seeds 1..100 total
sbatch --export=CONFIG=cox,SEEDS_PER_TASK=5 \
       --array=1-20 \
       RecurrentODE_py/job_submission/run_ode.sh

# skip the sandwich SE for a quick point-estimate run
sbatch --export=CONFIG=ltm,CI=0 \
       RecurrentODE_py/job_submission/run_ode.sh

# run seeds 1001..2000 as a follow-up batch
sbatch --export=CONFIG=cox,SEED_OFFSET=1000 \
       RecurrentODE_py/job_submission/run_ode.sh

# point at a different checkout of the repo
sbatch --export=CONFIG=cox,REPO_ROOT=/home/$USER/scratch/alt_checkout \
       RecurrentODE_py/job_submission/run_ode.sh

# use a different JSON file entirely
sbatch --export=CONFIG=my_expt,CONFIG_JSON=/home/$USER/my_configs.json \
       RecurrentODE_py/job_submission/run_ode.sh
```

## Adding a new configuration

Edit `configs.json` — add a key under `"configs"`:

```json
"cox_big": {
  "module": "cox",
  "data_setting": 1,
  "n_values": [4000, 8000],
  "seeds_per_task": 25
}
```

Then submit:

```bash
sbatch --export=CONFIG=cox_big RecurrentODE_py/job_submission/run_ode.sh
```

`submit_all.sh` will automatically pick the new key up the next time
it runs with no arguments, since it reads the key list from
`configs.json`.

## What each array task does

For array task `i` (1-indexed), `run_ode.sh` computes

```
seed_start = seed_offset + (i - 1) * seeds_per_task + 1
seed_end   = seed_offset + i       * seeds_per_task
```

and then, for every `seed ∈ [seed_start, seed_end]` and every
`N ∈ n_values`, invokes

```bash
# modules without a knots argument (cox, random_effect.cox)
python3 -m RecurrentODE_py.<module>.main <N> <seed> <data_setting> <ci>

# modules with a knots argument (aft, ltm, npmle, random_effect.aft_rec, random_effect.ltm)
python3 -m RecurrentODE_py.<module>.main <N> <seed> <data_setting> <knots> <ci>
```

Results are written by each module's `main.py` into its own
`data/` and `res/` subdirectories (see the source of the relevant
`RecurrentODE_py/<module>/main.py`).

## Monitoring and logs

```bash
# queue status for the current user
squeue -u $USER

# cancel a running array job
scancel <jobid>           # cancel whole array
scancel <jobid>_<task>    # cancel one array task

# logs (path set by SBATCH --output in run_ode.sh)
ls /home/$USER/random_effects/log/
tail -f /home/$USER/random_effects/log/ode_run-<jobid>_<task>.log
```

The `--output=...%x-%A_%a.log` pattern means one log file per array
task, e.g. `ode_run-123456_7.log` is array index 7 of job 123456.

## SBATCH resources (hard-coded in `run_ode.sh`)

```
--array=1-20
--nodes=2
--ntasks-per-node=1
--cpus-per-task=15
--mem-per-cpu=4GB
--time=05-10:00:00
--account=gongjun0
--partition=standard
```

Edit the header of `run_ode.sh` to change them — or override any of
them at submission time with the equivalent `sbatch` flag, e.g.
`sbatch --array=1-40 --time=02:00:00 --export=CONFIG=cox ...`.
