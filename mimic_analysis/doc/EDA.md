# MIMIC-III Recurrent-Event EDA

Initial exploratory analysis of the 35,643-admission cohort in
`processed-data/input.csv`, set up for the recurrent-event readmission
model. Companion script: `mimic_analysis/testing/eda_recurrent.py`.

## 1. Cohort layout

- `input.csv` has **one row per patient**: 35,643 rows ↔ 35,643 unique
  `SUBJECT_ID`s. Each row is the patient's **indexed admission** (the
  one with the V1..V26 covariates).
- To see the recurrent-event structure we pull **all** admissions for
  these patients from `raw-data/ADMISSIONS.csv`:

| Quantity | Value |
|---|---|
| Patients (= rows in `input.csv`) | 35,643 |
| Total admissions across all visits | 46,263 |
| Readmission events (= admissions − 1) | 10,620 |
| Mean events / patient | 0.30 |
| Median events / patient | 0 |
| Patients with **zero events** (single admission) | **29,284 (82.2%)** |
| Max admissions for one patient | 42 |
| Patients whose last admission ended in death | 4,723 (13.3%) |

Admissions-per-patient histogram (head):

| n_admissions | patients |
|---|---|
| 1 | 29,284 |
| 2 | 4,305 |
| 3 | 1,153 |
| 4 | 442 |
| 5 | 221 |
| 6 | 94 |
| 7 | 43 |
| 8 | 25 |
| 9 | 23 |
| 10 | 14 |
| ... | ... |

Follow-up days: median 8.5, mean 131, max 4,129 (~11.3 yr). A handful
of negative values exist from rare `DISCHTIME < ADMITTIME` records in
MIMIC.

## 2. Missingness in V1..V26

| Tier | Columns | % missing | Comment |
|---|---|---|---|
| **High** | `PaO2FiO2_vent_min` | 58.2% | only defined on ventilated patients with an ABG |
| **High** | `bilirubin_min`, `bilirubin_max` | 56.4% | non-routine lab, ordered on clinical suspicion |
| Medium | `tempc_min`, `tempc_max` | 1.9% | sporadic non-recording |
| Medium | `urineoutput` | 1.6% | requires urinary catheter or charted output |
| Low | `wbc`, `bicarbonate`, `bun`, `sodium`, `sysbp`, `gcs`, `heartrate`, `potassium` | 0.2–0.9% | almost always collected |
| Zero | `age`, `AIDS`, `HEM`, `METS`, `Adm_type` | 0% | demographic / categorical / always set |

### What `PaO2FiO2_vent_min` actually is

The **PaO₂/FiO₂ ratio** divides arterial oxygen partial pressure
(`PaO₂`, mmHg, from an arterial blood gas) by the fraction of inspired
oxygen (`FiO₂`, 0.21–1.0, from the ventilator setting). It is the
standard bedside measure of pulmonary gas-exchange impairment and is
the defining axis of the Berlin ARDS criteria:

| PaO₂/FiO₂ | Severity |
|---|---|
| ≤ 300 | mild ARDS |
| ≤ 200 | moderate ARDS |
| ≤ 100 | severe ARDS |

The `_vent_min` suffix means *minimum* value observed **while the
patient was mechanically ventilated** during the first-24-hour window.
A missing value typically means the patient was **not ventilated**, or
was ventilated but had no arterial blood gas drawn — both signals of a
less-acute respiratory state. (Conversely, a low recorded value is a
strong marker of severe respiratory failure.)

### What `bilirubin_min/max` actually are

Serum **total bilirubin** (mg/dL), a marker of hepatobiliary function
(liver disease, biliary obstruction) and red-cell breakdown
(hemolysis). Normal range is roughly 0.1–1.2 mg/dL. Elevated bilirubin
is a component of:

- **SOFA** liver subscore (used for organ-failure tracking)
- **MELD** (end-stage liver disease score)
- **APACHE / SAPS** severity scores

Bilirubin is **not** part of a standard daily metabolic panel — it is
ordered when there is clinical suspicion of liver dysfunction, jaundice
work-up, sepsis screening with hepatic concern, or post-operative
monitoring after hepatobiliary surgery. The 56% missingness reflects
this selective ordering pattern.

## 3. "Tested vs not-tested" as an informative signal

For several features in this dataset, **missingness is itself
informative** — the act of ordering or not ordering a test reflects the
clinician's prior judgment of the patient's clinical state. This is
especially true for:

| Column | Why missingness is informative |
|---|---|
| `PaO2FiO2_vent_min` | Missing ⇔ patient not ventilated / no ABG ⇔ less acute respiratory failure (a *protective* signal). Recorded ⇔ ICU-level respiratory support, often associated with worse prognosis. |
| `bilirubin_min/max` | Missing ⇔ no clinical reason to suspect liver involvement. Recorded ⇔ clinician triggered a liver work-up (often higher acuity). |
| `urineoutput` | Missing ⇔ no indwelling urinary catheter, often less critically ill. Recorded ⇔ catheterized, typically ICU-level care. |
| `tempc` | Missing ⇔ patient not actively monitored on telemetry/ICU floor at the time. |

### Why this matters for the model

The standard practice of "impute the median and proceed" **discards the
missingness signal** and can bias the recurrent-event hazard estimates
because the missingness mechanism is correlated with the outcome
(readmission and survival). Two reasonable mitigations:

1. **Missingness-indicator augmentation.** For each high-missingness
   feature add a binary `M_j = 1{X_j is missing}` column and impute
   the underlying value with the cohort mean (or any neutral value).
   The indicator absorbs the informative-missingness signal cleanly
   and lets the regression separate "what the test value was" from
   "whether the test was ordered." Best-suited to:
   - `PaO2FiO2_vent_min` → `M_vent`
   - `bilirubin_min/max` → `M_bilirubin`

2. **Joint missingness model (MNAR).** When the missingness mechanism
   itself depends on outcome, a joint model of the covariate process
   and the readmission process is more principled but considerably
   heavier to fit.

For the **informative censoring** problem (some patients are censored
because they died, and death is correlated with the X covariates),
indicator-based handling of missingness pairs naturally with a frailty
term that ties the censoring time to the recurrent-event process.

## 4. Correlation across V1..V26 (|r| ≥ 0.6)

All six high-correlation pairs are min/max readings of the same lab
within the 24-hour window — i.e., the spread between the two summaries
is small for that lab.

| Pair | r |
|---|---|
| `bilirubin_min` / `bilirubin_max` | **+0.983** |
| `bun_min` / `bun_max` | **+0.958** |
| `wbc_min` / `wbc_max` | **+0.873** |
| `bicarbonate_min` / `bicarbonate_max` | **+0.810** |
| `sodium_min` / `sodium_max` | +0.705 |
| `heartrate_max` / `heartrate_min` | +0.605 |

The first four are effectively collinear. Two ways to handle this:

- Replace each `(X_min, X_max)` pair with `(mean, range)` — same
  information, lower correlation.
- Drop one side per pair and keep only the more clinically
  informative summary (typically the *abnormal-direction* extreme:
  e.g., keep `bilirubin_max`, `bun_max`, `wbc_max`, drop the mins).

`sodium` and `heartrate` are borderline and probably fine to keep as
min/max.

## 5. Data-quality flags worth knowing

- **`age` max = 308.97**: MIMIC adds approximately 300 years to the
  DOB of any patient ≥ 89 at admission for de-identification. These
  rows should be re-coded to a fixed value (e.g., 90) before fitting,
  not left at face value.
- **`urineoutput`**: 6 negative values, max 98,280 mL — both are
  implausible and suggest sentinel/entry errors. Clip at clinically
  plausible bounds.
- **`PaO2FiO2_vent_min` max = 35,400**: physiologically implausible
  (normal upper bound ≈ 500). Likely a unit/charting error in a few
  rows; clip or drop.
- A few `DISCHTIME < ADMITTIME` records in `ADMISSIONS.csv` produce
  negative follow-up days at the patient level. Either drop those
  patients or set their follow-up to 0.

## 6. Verification of cohort assumptions

Companion script: `mimic_analysis/testing/verify_first_admission.py`.

### Is the indexed admission the patient's first hospital admission?

The PDF says `valid_aids.csv` covers patients alive 24h after their **first
ICU admission**. Empirically:

| Check | Patients | % |
|---|---|---|
| Indexed `HADM_ID` == patient's first hospital admit | 33,677 | **94.5%** |
| Indexed `HADM_ID` != first hospital admit (had earlier non-ICU stay) | 1,966 | 5.5% |

For the 1,966 with prior non-ICU admissions, the median gap between first
hospital admit and first ICU admit is **309 days**. The X covariates in
`input.csv` describe the first ICU stay, *not* the very first
hospitalization. Two options:

- Keep them, treat the cohort as "first ICU stay" (current default; we
  add a `is_first_hospital_admit` flag column).
- Drop them for clean "X at first hospital admission" semantics.

### `DISCHTIME < ADMITTIME` records

| Where | Count | Death rate (`HOSPITAL_EXPIRE_FLAG = 1`) |
|---|---|---|
| All of `ADMISSIONS.csv` | 98 | **80 (81.6%)** |
| In our cohort | **0** | — |

The 98 globally-bad records are dominantly in-hospital deaths with
mangled timestamps; **none are in our cohort**. However, **4 cohort
patients** still end up with negative *patient-level* follow-up (their
last `DISCHTIME` across all admissions is earlier than the first
`ADMITTIME`). All 4 died on their first admission → safe to drop as
data errors.

### Are X features available at first admission?

- **From the PDF**: `input.csv` is "non-temporal summary statistics
  within the first 24 hours after admission." `valid_aids.csv` patients
  are alive 24h after first ICU admit. So X is, by construction,
  available at the index admission and contains no per-event
  timestamps (it's already collapsed to one row per patient).
- The richer time-resolved version (24 hourly bins × 15 vars) lives in
  `imputed-normed-ep_1_24.npz` (`ep_tdata`).

## 7. Cohort cleaning (`clean_cohort.py`)

Companion script: `mimic_analysis/clean_cohort.py`. Outputs
`merged_data/cohort_clean.csv` (35,639 × 37). Each cleaning step prints
what it changed; summary below.

### What was cleaned

| Step | Rows / values affected | Action |
|---|---|---|
| **Age cap at 90** (MIMIC ~300 yr offset for ≥89) | 1,809 patients (5.1%) | `age = 90`; flag in `age_capped` |
| **Clip X to clinical bounds** | 287 values across 12 cols | `clip(lo, hi)` per column |
| **Missingness indicators** | 3 cols (PaO2FiO2, bilirubin_min/max) | added `M_<col>` binary cols |
| **Drop patients with `DISCHTIME < ADMITTIME` on any admission** | 13 patients | removed (unreliable timestamps) |
| **Drop residual negative follow-up** | 4 patients | removed |
| **Flag indexed-vs-first-hospital-admit** | 1,966 patients (5.5%) | added `is_first_hospital_admit` |

Final cohort after both drops: **35,626 patients**.

Bounds used (only the clipped columns shown):

| Column | (lo, hi) | n below lo | n above hi |
|---|---|---|---|
| `PaO2FiO2_vent_min` | (50, 600) | 73 | 39 |
| `urineoutput` | (0, 20000) | 6 | 7 |
| `tempc_min` | (25, 45) | 5 | 0 |
| `sysbp_min` | (30, 300) | 91 | 0 |
| `sysbp_max` | (30, 300) | 0 | 2 |
| `heartrate_min` | (20, 250) | 18 | 0 |
| `heartrate_max` | (20, 250) | 0 | 1 |
| `wbc_max` | (0, 500) | 0 | 3 |
| `potassium_min` | (1, 10) | 5 | 0 |
| `potassium_max` | (1, 10) | 0 | 22 |
| `sodium_min` | (100, 180) | 12 | 0 |
| `sodium_max` | (100, 180) | 0 | 3 |

Bilirubin and BUN bounds were specified but no values fell outside —
the data is well-behaved there.

### Final cohort: 35,626 patients

| Statistic | Value |
|---|---|
| Patients | 35,626 |
| Mean events / patient | 0.298 |
| Patients with 0 events | 29,280 (82.2%) |
| Patients died at last admit (death-censored) | 4,712 (13.2%) |
| Patients with `is_first_hospital_admit = 1` | 33,666 (94.5%) |

### Follow-up time distribution

| Quantile | days |
|---|---|
| 25% | 4.8 |
| **50%** (median) | **8.5** |
| 75% | 20.3 |
| 90% | 234.9 |
| 95% | 903.9 |
| 99% | 2,443.9 |
| max | 4,129 (~11.3 yr) |

**Heavy right tail.** 75% of patients are followed ≤ 20 d, but the top
1% extends past 6.7 years. For numerical stability you may want to cap
follow-up at, e.g., 5 years (1,825 d) — this affects only the top
~3-4% of patients but trims the long noisy tail of rare-late events.

### Output schema (`cohort_clean.csv`, 37 cols)

```
HADM_ID, SUBJECT_ID,
V1..V26 (cleaned/clipped),
age_capped, M_PaO2FiO2_vent_min, M_bilirubin_min, M_bilirubin_max,
n_admissions, n_events, follow_up_days, died_at_end,
is_first_hospital_admit
```

## 8. Long-format recurrent-event data (`build_long_format.py`)

The recurrent-event estimator (`recurrent_ode.fit`) takes long format:
one row per event + one censoring row per subject, with the X covariate
vector repeated on every row.

**Modeling convention used here**: each patient has **one X vector**
(the V1..V26 + indicators from their first admission). They contribute
**zero or more event rows** (`delta = 1`, one per *subsequent*
admission) plus exactly **one censoring row** (`delta = 0`, the final
row).

### Time origin and censoring

| Quantity | Definition |
|---|---|
| Time origin (`time = 0`) | first admission's `ADMITTIME` |
| Event time (k-th readmission) | `ADMITTIME_k − ADMITTIME_1` (days) |
| Censor time | `max(last DEATHTIME, last DISCHTIME, last ADMITTIME) − ADMITTIME_1` |
| `is_death_censored` flag | 1 if `last DEATHTIME` is non-null (informative censoring) |

The fallback to `last ADMITTIME` in the censor-time formula handles
admissions with missing `DISCHTIME` (transfers / open episodes), which
otherwise can produce a censor row earlier than the last event row.
After this fix, **every patient's last row has `delta = 0`**.

### Long-format summary

| Statistic | Value |
|---|---|
| Patients (unique `id`) | 35,626 |
| Total rows | 46,227 |
| Event rows (`delta = 1`) | 10,601 |
| Censoring rows (`delta = 0`) | 35,626 |
| Rows per patient: min / median / max | 1 / 1 / 42 |
| Death-censored patients | 4,712 (13.2%) |

A patient with no readmissions contributes exactly one row (their
censor). A patient with k readmissions contributes k + 1 rows.

### Output schema (`cohort_long.csv`, 34 cols)

```
id, time, delta,
V1..V26 (constant within id),
age_capped, M_PaO2FiO2_vent_min, M_bilirubin_min, M_bilirubin_max,
is_death_censored
```

Example (`id = 17`, one readmission at day 133, censored at day 137):

```
 id       time  delta   age  is_death_censored
 17 133.288889      1 47.45                  0
 17 137.309028      0 47.45                  0
```

### Format match against `recurrent_ode.fit()`

`recurrent_ode.fit()` (and the underlying `_save_data_npz`) expects:

```python
data = {
    'id':    np.ndarray,   # shape (n,) or (n,1), int
    'time':  np.ndarray,   # shape (n,) or (n,1), float
    'delta': np.ndarray,   # shape (n,) or (n,1), int (0 or 1)
    'x':     np.ndarray,   # shape (n_rows, p), float
}
```

The build script writes both a CSV (`cohort_long.csv`) and a ready-to-fit
NPZ (`cohort_long.npz`). Side-by-side:

| Field | Model expects | `cohort_long.npz` (ours) | Match |
|---|---|---|---|
| `id` | int (n,) | `(46227,) int64` | ✓ |
| `time` | float (n,) | `(46227,) float64` | ✓ |
| `delta` | int 0/1 (n,) | `(46227,) int64` | ✓ |
| `x` | float (n, p) | `(46227, 30) float64` | ✓ |

Plus the model assumes:

- Last row per `id` is the censoring row (`delta = 0`) — **OK**
- Times within `id` are non-decreasing — **OK**
- No NaN in `x` — **OK** (NaNs filled with 0; the M_* indicator
  columns retain the "test was/wasn't ordered" signal)

### NaN handling

The model does not accept NaN. Each missing entry in `V1..V26` is filled
with `0` and the missingness signal is preserved separately by the
`M_*` indicator columns. Imputation breakdown for the long-format file:

| Column | NaN → 0 |
|---|---|
| `PaO2FiO2_vent_min` | 27,757 |
| `bilirubin_min` / `bilirubin_max` | 25,385 each |
| `urineoutput` | 1,073 |
| `tempc_min` / `tempc_max` | 805 / 803 |
| `wbc_min/max`, `bicarbonate_min/max` | 380 / 365 each |
| `bun_min/max`, `sodium_min/max`, `gcs_min`, `potassium_min/max` | < 200 each |

Total imputed values: **83,905** (across 16 columns).

If you'd rather impute with the cohort mean (or fit-fold mean), swap
the `np.nan_to_num(x, nan=0.0)` line in `build_long_format.py`.

### Loading for the model

```python
import numpy as np
from recurrent_ode import fit

d = np.load('mimic_analysis/merged_data/cohort_long.npz')
data = {'id': d['id'], 'time': d['time'],
        'delta': d['delta'], 'x': d['x']}

est = fit(data, model='aft', knots='quantile', ci=True)
print(est.beta)
```

## 9. Outputs

| File (under `merged_data/`) | What |
|---|---|
| `missingness.csv` | Per-column NaN / zero / negative / min / max |
| `corr_matrix.csv` | Full 26 × 26 Pearson correlation matrix |
| `per_patient_summary.csv` | One row per patient: `n_admissions`, `n_events`, `follow_up_days`, `died_at_end` |
| **`cohort_clean.csv`** | **35,626 × 37 — wide format, one row per patient** |
| **`cohort_long.csv`** | **46,227 × 34 — long format, human-readable** |
| **`cohort_long.npz`** | **dict with `id`/`time`/`delta`/`x` — ready for `recurrent_ode.fit()`** |

Reproduce with:

```bash
python3 mimic_analysis/testing/eda_recurrent.py           # raw EDA tables
python3 mimic_analysis/testing/verify_first_admission.py  # cohort checks
python3 mimic_analysis/clean_cohort.py                    # cleaned wide cohort
python3 mimic_analysis/build_long_format.py               # long-format input
```
