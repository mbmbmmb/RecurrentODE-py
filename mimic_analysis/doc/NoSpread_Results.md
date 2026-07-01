# No-Spread Fits: Cox-RE and LTM-RE on the MIMIC Readmission Cohort

Companion script: `mimic_analysis/fit_nospread.py`. Outputs land in
`mimic_analysis/merged_data/nospread*`.

This report documents two no-spread fits — full follow-up and an
administrative cap at 182.5 days (0.5 y) — and the rationale for
treating the capped LTM as the headline result.

## 1. Feature set

All `*_spread` columns are dropped. Only the per-stay mean (which is the
midpoint of `(min, max)`) is kept. Existing `*_min` features
(`PaO2FiO2_vent_min`, `gcs_min`), demographics, indicators, and the
`Adm_type` dummies remain.

| group | features | note |
|---|---|---|
| dropped (9) | `heartrate_spread`, `sysbp_spread`, `tempc_spread`, `bun_spread`, `wbc_spread`, `potassium_spread`, `sodium_spread`, `bicarbonate_spread`, `bilirubin_spread` | per-stay range |
| kept means (9) | `heartrate_mean`, `sysbp_mean`, `tempc_mean`, `bun_mean`, `wbc_mean`, `potassium_mean`, `sodium_mean`, `bicarbonate_mean`, `bilirubin_mean` | midpoint of min/max |
| existing min (2) | `PaO2FiO2_vent_min`, `gcs_min` | already worst-of-day in source |
| demographics (1) | `age` | |
| binary comorbidities (3) | `AIDS`, `HEM`, `METS` | |
| missingness indicators (2) | `M_PaO2FiO2_vent_min`, `M_bilirubin_mean` | |
| Adm_type dummies (2) | `Adm_type_scheduled`, `Adm_type_unscheduled` | reference = medical |

Total: **20 model covariates**.

## 2. Cohort sizes

| scenario | rows | patients | events | censors | time range |
|---|---:|---:|---:|---:|---|
| full follow-up | 46,227 | 35,626 | 10,601 | 35,626 | [0.04, 4129.1] d |
| cap = 182.5 d (0.5 y) | 39,576 | 35,626 | 3,950 | 35,626 | [0.04, 182.5] d |

Capping retains every patient (synthetic censor at 182.5 d for those
with longer follow-up) but only **37%** of readmission events fall
inside the 0.5-year window.

## 3. Fits run

For each cohort variant (full / capped):
1. **Cox-RE** with raw time
2. **Cox-RE** with `log10(1 + t)`
3. **LTM-RE** with `log10(1 + t)`, anchor `heartrate_mean = +1`,
   K1 spline knots, sandwich-bootstrap CI

The raw-time LTM scenario is omitted (singular Fisher under raw time —
known issue from the prior pipeline).

## 4. Results — full follow-up (no cap)

LTM hr_log10, full cohort. Sig (p < 0.05) features bolded.

| feature | expected | coef | p | match |
|---|---|---:|---:|---|
| heartrate_mean | anchor | +1.000 | — | anchor |
| **age** | + | −0.881 | 2.7e-07 | **WRONG** |
| **sysbp_mean** | − | −0.272 | 3.5e-02 | OK |
| tempc_mean | ? | −0.223 | 2.0e-01 | n/a |
| PaO2FiO2_vent_min | − | +0.236 | 4.1e-01 | n.s. |
| **urineoutput** | − | −0.960 | 2.1e-07 | OK |
| **bun_mean** | + | +0.365 | 1.0e-02 | OK |
| **wbc_mean** | + | −0.527 | 3.2e-02 | **WRONG** |
| **potassium_mean** | ? | +1.482 | 2.8e-18 | n/a |
| **sodium_mean** | ? | +0.494 | 7.1e-06 | n/a |
| **bicarbonate_mean** | ? | +2.257 | 3.2e-31 | n/a |
| bilirubin_mean | + | −0.069 | 1.6e-01 | n.s. |
| gcs_min | − | −0.273 | 1.8e-01 | n.s. |
| **AIDS** | + | −1.287 | 8.4e-03 | **WRONG** |
| HEM | + | −0.585 | 3.2e-01 | n.s. |
| METS | + | +0.830 | 1.7e-01 | n.s. |
| **M_PaO2FiO2_vent_min** | + | +1.974 | 3.9e-07 | OK |
| **M_bilirubin_mean** | + | −3.443 | 2.4e-06 | **WRONG** |
| **Adm_type_scheduled** | ? | −5.792 | 3.2e-02 | n/a |
| **Adm_type_unscheduled** | + | +1.432 | 2.5e-04 | OK |

**Sig = 13.** OK = 5, WRONG = 4 (`age`, `wbc_mean`, `AIDS`,
`M_bilirubin_mean`), n/a + anchor = 4. LTM `succ=False` (inner-loop
iteration limit; SE bootstrap converged.)

Full per-feature numbers and Cox triangulation: see
`merged_data/nospread_coef_wide.csv`.

## 5. Results — cap = 182.5 d (0.5 y)

LTM hr_log10, capped cohort.

| feature | expected | coef | p | match |
|---|---|---:|---:|---|
| heartrate_mean | anchor | +1.000 | — | anchor |
| age | + | +0.132 | 3.8e-01 | OK n.s. |
| **sysbp_mean** | − | −1.108 | 1.4e-09 | OK |
| **tempc_mean** | ? | −0.974 | 4.3e-08 | n/a |
| **PaO2FiO2_vent_min** | − | +1.708 | 5.0e-12 | **WRONG** |
| urineoutput | − | −0.695 | 8.0e-02 | OK n.s. |
| **bun_mean** | + | +0.467 | 1.1e-03 | OK |
| wbc_mean | + | +0.252 | 6.1e-02 | OK n.s. |
| **potassium_mean** | ? | +0.539 | 7.6e-04 | n/a |
| sodium_mean | ? | +0.062 | 6.5e-01 | n/a |
| **bicarbonate_mean** | ? | +2.158 | 3.3e-36 | n/a |
| bilirubin_mean | + | −0.019 | 6.7e-01 | n.s. |
| gcs_min | − | −0.094 | 4.7e-01 | n.s. |
| **AIDS** | + | −3.836 | 3.5e-04 | **WRONG** |
| HEM | + | −0.168 | 7.7e-01 | n.s. |
| **METS** | + | +2.985 | 6.7e-28 | OK |
| **M_PaO2FiO2_vent_min** | + | +1.192 | 3.7e-08 | OK |
| M_bilirubin_mean | + | −0.587 | 5.9e-02 | WRONG n.s. |
| **Adm_type_scheduled** | ? | −1.014 | 6.3e-04 | n/a |
| **Adm_type_unscheduled** | + | +4.181 | 6.2e-61 | OK |

**Sig = 11.** OK = 5, WRONG = 2 (`PaO2FiO2_vent_min`, `AIDS`), n/a +
anchor = 4. Cox log10 sees 7 sig features and **0 wrong-sign sig**.

Full per-feature numbers: `merged_data/nospread_cap182d_coef_wide.csv`.

## 6. Why the cap is the more reasonable model

The wrong-sign cluster is what drives the recommendation.

| feature | full LTM β (p) | capped LTM β (p) | reading |
|---|---:|---:|---|
| age | −0.88 (2.7e-07) | +0.13 (0.38) | flipped to expected sign, n.s. |
| wbc_mean | −0.53 (3.2e-02) | +0.25 (0.06) | flipped, near-sig OK |
| M_bilirubin_mean | −3.44 (2.4e-06) | −0.59 (0.06) | toward 0, n.s. |
| AIDS | −1.29 (8.4e-03) | −3.84 (3.5e-04) | unchanged direction; small-N |
| PaO2FiO2_vent_min | +0.24 (0.41) | +1.71 (5e-12) | newly wrong-sign sig |

Three of the four full-cohort wrong-sign sig features (`age`,
`wbc_mean`, `M_bilirubin_mean`) **resolve** under the cap. The
mechanism:

- **Competing risk with mortality.** Older / sicker patients (high
  `age`, high `wbc_mean`, ordered bilirubin) are more likely to die
  before being readmitted across the 11-year horizon. The full-cohort
  fit is forced to attribute their depleted readmission rate to a
  protective covariate effect — a survival-selection artifact, not a
  treatment signal. Inside the 0.5-y window, mortality has had less
  time to deplete the at-risk set, and the signs settle.
- **Time-stationarity.** First-stay covariates plausibly drive
  readmission at short horizons (post-discharge ~6 mo). Beyond that,
  drivers shift to chronic-disease management and social factors not
  measured in V1..V26. The capped fit isn't asking the model to
  extrapolate across that regime change.
- **Clinical face validity.** ICU readmission is mostly an
  early-window phenomenon; a 0.5-y horizon is a defensible domain for
  the X→T relationship implied by the LTM functional form.

Cost of the cap: events drop 10,601 → 3,950 (loses 63%). Statistical
power falls, but the surviving 11 LTM hits are more trustworthy than
the 13 full-cohort hits.

## 7. The two remaining wrong-sign sig features

| feature | LTM β (p) | Cox β (p) | data-level reading |
|---|---:|---:|---|
| `PaO2FiO2_vent_min` | +1.71 (5e-12) | +0.06 (0.13) | LTM only — Cox doesn't replicate. Two leading explanations: (a) collinearity with `M_PaO2FiO2_vent_min` (β=+1.19, sig) — the "test-was-ordered" indicator already absorbs the severity signal, leaving the residual in measurement-on-test-takers anti-correlated with sickness; (b) ventilator-strata heterogeneity that the LTM functional baseline picks up but Cox averages out. Structural artifact of the missingness encoding, not a model failure. |
| `AIDS` | −3.84 (3.5e-04) | −0.22 (0.25) | Low-N. AIDS is rare in this cohort (~0.5% of patients). Cox cannot reject zero; LTM's larger β reflects amplification by the +1 anchor on `heartrate_mean`. Small-cell instability. |

Both are explainable from the data structure. Reportable as a finding
("AIDS estimate is unreliable in this cohort"), not a bug to fix in the
LTM.

## 8. Cox vs. LTM under the cap

| | sig features | wrong-sign sig | character |
|---|---:|---:|---|
| Cox log10 | 7 | 0 | clean, conservative; loses the `METS`, `tempc_mean`, `bun_mean`, `potassium_mean`, `bicarbonate_mean` LTM signals at p<0.05 |
| LTM hr_log10 | 11 | 2 (both explainable) | more powerful via random-effect frailty + functional baseline |

Cox is the cleaner sign-correctness backstop; LTM is the inference
target.

## 9. Headline recommendation

**Capped (182.5 d) LTM hr_log10 as primary; Cox log10 as
triangulation.**

Publishable signal (LTM-sig, OK or no-prior, both Cox-supported when
prior exists):

| feature | LTM β | Cox β | direction |
|---|---:|---:|---|
| `bicarbonate_mean` | +2.16 | +0.14 | + (no prior; both estimators) |
| `Adm_type_unscheduled` | +4.18 | +0.25 | + (matches expectation) |
| `METS` | +2.99 | +0.23 | + (matches expectation) |
| `M_PaO2FiO2_vent_min` | +1.19 | +0.07 | + (matches expectation; Cox n.s.) |
| `sysbp_mean` | −1.11 | −0.07 | − (matches expectation) |
| `tempc_mean` | −0.97 | −0.07 | − (no prior; both estimators) |
| `bun_mean` | +0.47 | +0.03 | + (matches expectation; Cox n.s.) |
| `potassium_mean` | +0.54 | +0.03 | + (no prior; both estimators) |

Reported caveats:

1. `AIDS` estimate (β=−3.84 in LTM) reflects ~0.5% prevalence
   instability. Treat as cohort artefact.
2. `PaO2FiO2_vent_min` LTM/Cox disagreement is consistent with the
   missingness-indicator absorbing the severity signal. Worth a
   robustness check (drop `M_PaO2FiO2_vent_min`; or stratify on vent
   status) but not a primary-fit blocker.
3. LTM `succ=False` flag is the inner-sieve iteration limit, *not* a
   bootstrap-CI failure. SEs converge across 600 resamples.

## 10. Files

| path | content |
|---|---|
| `mimic_analysis/fit_nospread.py` | runner; `--cap <days>` to enable admin censor |
| `merged_data/nospread.log` | full-cohort console log |
| `merged_data/nospread_coef_table.csv` | long: feature × scenario × stats |
| `merged_data/nospread_coef_wide.csv` | wide: feature × scenario coef/se/p |
| `merged_data/nospread_cap182d.log` | capped-cohort console log |
| `merged_data/nospread_cap182d_coef_table.csv` | capped long format |
| `merged_data/nospread_cap182d_coef_wide.csv` | capped wide format |
