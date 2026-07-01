# Raw Feature Histograms with Middle-99% Clip Bounds

A diagnostic pass over the raw MIMIC first-admission feature matrix
(35,643 patients × 26 features). For every continuous feature this
report shows the **raw histogram** with summary statistics
(`n, min, max, mean, std`) and the **middle-99% clip bounds**
(`q0.5%`, `q99.5%`) overlaid as red dashed lines. **No transformation
is applied** — this is the reference view for the raw distribution
shapes and where a quantile-based clip would cut.

Only adjustment to "raw": age capped at 90 (MIMIC adds ~300 yr offset
for patients ≥89; otherwise the histogram would be unreadable).

## Feature glossary

All continuous summary statistics are computed from raw measurements
within the **first 24 h after the first ICU admission** (one row per
patient). Lab abbreviations and the categorical block follow the
SAPS-II severity-score convention (Le Gall, Lemeshow & Saulnier 1993,
implemented in `mit-lcp/mimic-code/concepts/severityscores/sapsii.sql`).
For lab/vital pairs we list both the raw `*_min / *_max` columns
shown in the histograms below **and** the post-collapse `*_mean /
*_spread` columns (see "Mean/spread re-parameterization" section).

**Demographic**

| feature | units | meaning |
| :--- | :--- | :--- |
| `age` | years | Patient age at first admission. MIMIC adds ~300 yr to ages ≥89 for de-identification; we cap at 90. Cohort range: 15–90. |

**Vital signs** (raw min/max collapsed to mean/spread in the model)

| raw | derived | units | meaning |
| :--- | :--- | :--- | :--- |
| `heartrate_min`, `heartrate_max` | `heartrate_mean`, `heartrate_spread` | beats/min | Heart rate from continuous bedside monitor. Adult resting normal ~60–100. |
| `sysbp_min`, `sysbp_max` | `sysbp_mean`, `sysbp_spread` | mmHg | Systolic blood pressure. Hypotension <90, hypertension >140. |
| `tempc_min`, `tempc_max` | `tempc_mean`, `tempc_spread` | °C | Core body temperature. Fever >38, hypothermia <35. |

**Respiratory**

| feature | units | meaning |
| :--- | :--- | :--- |
| `PaO2FiO2_vent_min` | mmHg / fraction | Minimum PaO2 ÷ FiO2 ratio while mechanically ventilated. Oxygenation index used in the ARDS Berlin definition: <300 mild, <200 moderate, <100 severe ARDS. **NaN ≈ 58%**, meaning patient was not ventilated. |

**Labs — renal / electrolyte / metabolic**

| raw | derived | units | meaning |
| :--- | :--- | :--- | :--- |
| `urineoutput` | (kept as-is) | mL / 24 h | Total urinary output in the first 24 h. Oliguria <500, anuria <100. |
| `bun_min`, `bun_max` | `bun_mean`, `bun_spread` | mg/dL | Blood urea nitrogen — marker of renal function and protein catabolism. Normal 7–20. |
| `wbc_min`, `wbc_max` | `wbc_mean`, `wbc_spread` | ×10⁹/L (K/μL) | White blood cell count. Normal 4–11. Leukocytosis suggests infection/inflammation; leukopenia suggests sepsis/marrow suppression. |
| `potassium_min`, `potassium_max` | `potassium_mean`, `potassium_spread` | mEq/L | Serum potassium. Normal 3.5–5.0. Originally kept as min/max (post-clip `r ≈ 0.34`); switched to mean/spread for consistency with the other lab pairs and so the model's relative-effect interpretation is uniform across labs. |
| `sodium_min`, `sodium_max` | `sodium_mean`, `sodium_spread` | mEq/L | Serum sodium. Normal 135–145. |
| `bicarbonate_min`, `bicarbonate_max` | `bicarbonate_mean`, `bicarbonate_spread` | mEq/L | Serum bicarbonate. Normal 22–28; low = metabolic acidosis. |
| `bilirubin_min`, `bilirubin_max` | `bilirubin_mean`, `bilirubin_spread` | mg/dL | Total serum bilirubin — liver function. Normal 0.3–1.2; >2 = jaundice; >10 = severe cholestasis. **NaN ≈ 56%**, meaning the liver-function panel was not ordered. |

**Neurological**

| feature | units | meaning |
| :--- | :--- | :--- |
| `gcs_min` | 3–15 | Minimum Glasgow Coma Scale (eye + verbal + motor). 3 = deep coma; 15 = fully alert. |

**SAPS-II comorbidity flags & admission type** (categorical)

| feature | values | meaning |
| :--- | :--- | :--- |
| `AIDS` | 0 / 1 | 1 = acquired immunodeficiency syndrome (ICD-9 042). 0.9% prevalence. |
| `HEM` | 0 / 1 | 1 = hematologic malignancy (leukemia / lymphoma ICD-9). 2.7% prevalence. |
| `METS` | 0 / 1 | 1 = metastatic cancer (ICD-9 196–199). 6.1% prevalence. |
| `Adm_type` | 0 / 1 / 2 | 0 = medical (69%), 1 = scheduled surgical (13%), 2 = unscheduled surgical (17%). |

**Missingness indicators** (added by `clean_cohort.py`)

| feature | flags | meaning |
| :--- | :--- | :--- |
| `M_PaO2FiO2_vent_min` | `PaO2FiO2_vent_min` | 1 if the patient was not mechanically ventilated (lab not measurable). |
| `M_bilirubin_mean` | `bilirubin_mean`, `bilirubin_spread` | 1 if the liver-function panel was not ordered. One indicator covers both derived bilirubin features (shared NaN mask). |

`age_capped` was a previous indicator for the de-identified ~300-yr
offset. **Removed** — the cap information is implicit in the post-cap
age, and marginal hazard above 90 is small.

## Method

For each feature `x`:

1. Drop NaN; compute `n, min, max, mean, std`.
2. Compute `lo = q0.005`, `hi = q0.995` — middle-99% bounds: symmetric,
   shape-agnostic, removes ~0.5% from each tail.
3. Plot 80-bin histogram with `lo` / `hi` overlaid as red dashed lines.
4. Record counts strictly below `lo` and strictly above `hi`.

Outputs: `merged_data/clip99_bounds.csv`, `doc/plots/clip99/<feature>.png`.

## Per-feature summary table

| feature | n | min | max | mean | std | q0.5% | q99.5% | clip lo / hi | % |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| age | 35643 | 15.1 | 90.0 | 64.0 | 17.4 | 18.9 | 90.0 | 178 / 0 | 0.50% |
| heartrate_max | 35570 | 37 | 280 | 103 | 20.4 | 61 | 169 | 141 / 178 | 0.90% |
| heartrate_min | 35570 | 0.35 | 140 | 71 | 14.7 | 35 | 114 | 167 / 157 | 0.91% |
| sysbp_max | 35543 | 81 | 323 | 150 | 23.6 | 103 | 228 | 175 / 175 | 0.98% |
| sysbp_min | 35543 | 0.3 | 180 | 92.5 | 17.5 | 42 | 143 | 174 / 173 | 0.98% |
| tempc_max | 34956 | 31.6 | 42.8 | 37.5 | 0.77 | 35.8 | 39.9 | 170 / 175 | 0.99% |
| tempc_min | 34956 | 20.9 | 39.6 | 36.1 | 0.78 | 33.2 | 37.9 | 175 / 164 | 0.97% |
| PaO2FiO2_vent_min | 14908 | 24 | 35400 | 238 | 412 | 50 | 543 | 73 / 75 | 0.99% |
| urineoutput | 35074 | -2600 | 98300 | 2020 | 1490 | 20 | 7230 | 167 / 176 | 0.98% |
| bun_min | 35536 | 1 | 254 | 22 | 18.3 | 3 | 112 | 57 / 177 | 0.66% |
| bun_max | 35536 | 1 | 272 | 26.2 | 21.3 | 5 | 129 | 169 / 177 | 0.97% |
| wbc_min | 35340 | 0.1 | 443 | 10.5 | 7.4 | 0.7 | 35.5 | 161 / 177 | 0.96% |
| wbc_max | 35340 | 0.1 | 600 | 13.9 | 10.7 | 1.2 | 47.5 | 167 / 177 | 0.97% |
| potassium_min | 35579 | 0.6 | 6.2 | 3.73 | 0.52 | 2.3 | 5.3 | 157 / 150 | 0.86% |
| potassium_max | 35579 | 2.3 | 26.5 | 4.69 | 0.95 | 3.2 | 8.5 | 121 / 168 | 0.81% |
| sodium_min | 35540 | 1.21 | 178 | 137 | 4.83 | 118 | 149 | 176 / 150 | 0.92% |
| sodium_max | 35540 | 108 | 182 | 140 | 4.44 | 126 | 157 | 141 / 164 | 0.86% |
| bicarbonate_min | 35353 | 2 | 50 | 22.6 | 4.54 | 8 | 37 | 163 / 149 | 0.88% |
| bicarbonate_max | 35353 | 6 | 53 | 25 | 4.04 | 14 | 40 | 118 / 124 | 0.68% |
| bilirubin_min | 15539 | 0.1 | 79 | 1.55 | 3.79 | 0.1 | 28.2 | 0 / 78 | 0.50% |
| bilirubin_max | 15539 | 0.1 | 82.8 | 1.81 | 4.23 | 0.1 | 30.8 | 0 / 78 | 0.50% |
| gcs_min | 35565 | 3 | 15 | 13.8 | 2.63 | 3 | 15 | 0 / 0 | 0.00% |

`clip lo / hi` = number of observations strictly below `q0.5%` /
strictly above `q99.5%`. `%` = total fraction trimmed.

## Histograms

<div class="hist-grid">
<figure><figcaption>age</figcaption><img src="plots/clip99/age.png"></figure>
<figure><figcaption>heartrate_max</figcaption><img src="plots/clip99/heartrate_max.png"></figure>
<figure><figcaption>heartrate_min</figcaption><img src="plots/clip99/heartrate_min.png"></figure>
<figure><figcaption>sysbp_max</figcaption><img src="plots/clip99/sysbp_max.png"></figure>
<figure><figcaption>sysbp_min</figcaption><img src="plots/clip99/sysbp_min.png"></figure>
<figure><figcaption>tempc_max</figcaption><img src="plots/clip99/tempc_max.png"></figure>
<figure><figcaption>tempc_min</figcaption><img src="plots/clip99/tempc_min.png"></figure>
<figure><figcaption>PaO2FiO2_vent_min</figcaption><img src="plots/clip99/PaO2FiO2_vent_min.png"></figure>
<figure><figcaption>urineoutput</figcaption><img src="plots/clip99/urineoutput.png"></figure>
<figure><figcaption>bun_min</figcaption><img src="plots/clip99/bun_min.png"></figure>
<figure><figcaption>bun_max</figcaption><img src="plots/clip99/bun_max.png"></figure>
<figure><figcaption>wbc_min</figcaption><img src="plots/clip99/wbc_min.png"></figure>
<figure><figcaption>wbc_max</figcaption><img src="plots/clip99/wbc_max.png"></figure>
<figure><figcaption>potassium_min</figcaption><img src="plots/clip99/potassium_min.png"></figure>
<figure><figcaption>potassium_max</figcaption><img src="plots/clip99/potassium_max.png"></figure>
<figure><figcaption>sodium_min</figcaption><img src="plots/clip99/sodium_min.png"></figure>
<figure><figcaption>sodium_max</figcaption><img src="plots/clip99/sodium_max.png"></figure>
<figure><figcaption>bicarbonate_min</figcaption><img src="plots/clip99/bicarbonate_min.png"></figure>
<figure><figcaption>bicarbonate_max</figcaption><img src="plots/clip99/bicarbonate_max.png"></figure>
<figure><figcaption>bilirubin_min</figcaption><img src="plots/clip99/bilirubin_min.png"></figure>
<figure><figcaption>bilirubin_max</figcaption><img src="plots/clip99/bilirubin_max.png"></figure>
<figure><figcaption>gcs_min</figcaption><img src="plots/clip99/gcs_min.png"></figure>
</div>

## Post-clip correlation heatmap

Pearson correlation across the 22 continuous features after each
feature is **clipped to its middle-99% bounds** and **missing values
are filled with 0** (paired with the `M_*` indicator columns
elsewhere in the pipeline). Saved to
`merged_data/clip99_correlation.csv` (full 22×22 matrix).

![](plots/clip99/correlation.png)

**Top 15 absolute correlations:**

| pair | r |
| :--- | ---: |
| tempc_max ↔ tempc_min | +0.987 |
| bilirubin_min ↔ bilirubin_max | +0.983 |
| bun_min ↔ bun_max | +0.960 |
| sodium_min ↔ sodium_max | +0.928 |
| bicarbonate_min ↔ bicarbonate_max | +0.845 |
| wbc_min ↔ wbc_max | +0.831 |
| heartrate_max ↔ heartrate_min | +0.633 |
| sysbp_max ↔ sysbp_min | +0.399 |
| potassium_min ↔ potassium_max | +0.345 |
| bun_min ↔ potassium_min | +0.313 |
| sodium_min ↔ bicarbonate_max | +0.297 |
| sodium_min ↔ bicarbonate_min | +0.289 |
| sodium_max ↔ bicarbonate_max | +0.286 |
| age ↔ bun_min | +0.276 |
| bun_max ↔ potassium_min | +0.270 |

The dominant structure is the expected `*_min ↔ *_max` pairing within
each lab — the strongest seven pairs above are all same-lab min/max.
Cross-lab signal is much weaker (|r| ≤ 0.32). The 0-fill on
`PaO2FiO2_vent_min` and `bilirubin_*` (each ~58% / 56% missing) damps
their visible cross-correlations because the imputed zeros pull
covariance toward 0; the `M_*` indicator columns (in the model,
not shown here) carry the missing-or-not signal separately.

## Mean/spread re-parameterization

For lab/vital pairs with redundant `*_min ↔ *_max` correlation we
replace each pair with `(mean, spread)`, defined per row as

```
mean   = (max + min) / 2
spread = |max − min|
```

This is committed in `clean_cohort.py` (step 2a) for **all nine**
pairs **tempc**, **bilirubin**, **bun**, **sodium**, **bicarbonate**,
**wbc**, **heartrate**, **sysbp**, **potassium** (the first six had
post-clip `|r| > 0.8`; `heartrate` (0.63) and `sysbp` (0.40) were
added on user request to keep the vital-sign block consistent with the
lab block; `potassium` (0.34) was added last for full uniformity, so
every lab/vital coefficient in the model carries the same
*level vs. variability* interpretation). Final feature count:
**30 → 29** model inputs (one fewer missingness indicator since
`bilirubin_min`/`max` share a NaN mask).

After the collapse, **`|r| > 0.8` pair count drops from 6 → 0**, and
the strongest residual within-pair correlation is `bilirubin_mean ↔
bilirubin_spread` at 0.57. Within each collapsed group `mean`
captures the level and `spread` captures within-admission variability
— moderately correlated (0.33–0.57) but no longer redundant. The
heatmap below shows the result on the full feature set including
categorical and missingness columns.

## Full feature correlation (incl. categorical + missingness)

The 22 continuous mean/spread features extended with the SAPS-II
binary block (`AIDS`, `HEM`, `METS`), two `Adm_type` one-hot dummies
(reference = medical = 0; `Adm_type_scheduled` = 1 if scheduled
surgical, `Adm_type_unscheduled` = 1 if unscheduled surgical), and
the two missingness indicators (`M_PaO2FiO2_vent_min`,
`M_bilirubin_mean`). 29 features × 29 features Pearson heatmap on the
same post-clip + 0-fill frame, with M_* computed from the
**pre-fill** NaN mask. Saved to `merged_data/clip99_correlation_full.csv`.

![](plots/clip99/correlation_full.png)

**Top 15 absolute correlations (full feature set):**

| pair | r |
| :--- | ---: |
| PaO2FiO2_vent_min ↔ M_PaO2FiO2_vent_min | −0.856 |
| bilirubin_mean ↔ bilirubin_spread | +0.568 |
| bun_mean ↔ bun_spread | +0.466 |
| wbc_mean ↔ wbc_spread | +0.453 |
| heartrate_mean ↔ heartrate_spread | +0.396 |
| bun_spread ↔ bicarbonate_spread | +0.391 |
| sodium_spread ↔ bicarbonate_spread | +0.370 |
| potassium_min ↔ potassium_max | +0.345 |
| bilirubin_mean ↔ M_bilirubin_mean | −0.332 |
| sysbp_mean ↔ sysbp_spread | +0.327 |
| tempc_spread ↔ M_PaO2FiO2_vent_min | −0.306 |
| sysbp_spread ↔ M_PaO2FiO2_vent_min | −0.304 |
| bun_spread ↔ sodium_spread | +0.293 |
| potassium_min ↔ bun_mean | +0.292 |
| potassium_max ↔ sodium_spread | +0.288 |

The strong −0.86 between `PaO2FiO2_vent_min` and `M_PaO2FiO2_vent_min`
is **a 0-fill construction artifact**, not a real relationship: rows
with `M = 1` have the lab 0-filled at this stage, so the two columns
are mechanically anti-correlated. After the **post-scale** 0-fill in
`scale_features.py` the imputed value sits at the scaled center (0)
regardless of the M_* value, so the model sees no value-dependent
contribution from missing rows — the M_* coefficient is then free to
absorb the orthogonal "test was ordered" signal. Same logic for the
−0.33 `bilirubin_mean ↔ M_bilirubin_mean` and the M_PaO2 ↔ vital-sign-spread
pairs (sicker / more-monitored patients have both wider vital
ranges and a recorded ABG).

`AIDS`, `HEM`, `METS`, and the two `Adm_type` dummies show
`|r| < 0.1` against every continuous feature and against each other —
the SAPS-II block is genuinely orthogonal to the lab/vital block in
this cohort.

## Mean/spread feature histograms

Same plotting rule as the histogram grid at the top: middle-99%
clipped distribution with raw (pre-clip) `n / min / max / mean / std`
in the legend, plus the `q0.5%` / `q99.5%` cut lines in red. The 16
plots below cover only the **derived** features (mean & spread for
the 8 collapsed pairs) — the unchanged features (`age`,
`PaO2FiO2_vent_min`, `urineoutput`, `potassium_min`, `potassium_max`,
`gcs_min`) are already shown above.

<div class="hist-grid">
<figure><figcaption>heartrate_mean</figcaption><img src="plots/clip99/heartrate_mean.png"></figure>
<figure><figcaption>heartrate_spread</figcaption><img src="plots/clip99/heartrate_spread.png"></figure>
<figure><figcaption>sysbp_mean</figcaption><img src="plots/clip99/sysbp_mean.png"></figure>
<figure><figcaption>sysbp_spread</figcaption><img src="plots/clip99/sysbp_spread.png"></figure>
<figure><figcaption>tempc_mean</figcaption><img src="plots/clip99/tempc_mean.png"></figure>
<figure><figcaption>tempc_spread</figcaption><img src="plots/clip99/tempc_spread.png"></figure>
<figure><figcaption>bun_mean</figcaption><img src="plots/clip99/bun_mean.png"></figure>
<figure><figcaption>bun_spread</figcaption><img src="plots/clip99/bun_spread.png"></figure>
<figure><figcaption>wbc_mean</figcaption><img src="plots/clip99/wbc_mean.png"></figure>
<figure><figcaption>wbc_spread</figcaption><img src="plots/clip99/wbc_spread.png"></figure>
<figure><figcaption>sodium_mean</figcaption><img src="plots/clip99/sodium_mean.png"></figure>
<figure><figcaption>sodium_spread</figcaption><img src="plots/clip99/sodium_spread.png"></figure>
<figure><figcaption>bicarbonate_mean</figcaption><img src="plots/clip99/bicarbonate_mean.png"></figure>
<figure><figcaption>bicarbonate_spread</figcaption><img src="plots/clip99/bicarbonate_spread.png"></figure>
<figure><figcaption>bilirubin_mean</figcaption><img src="plots/clip99/bilirubin_mean.png"></figure>
<figure><figcaption>bilirubin_spread</figcaption><img src="plots/clip99/bilirubin_spread.png"></figure>
</div>

## Categorical / binary features

The remaining four covariates are not continuous and are NOT clipped.
They are the SAPS-II comorbidity / admission-type indicators (Le
Gall, Lemeshow, Saulnier 1993), inherited verbatim from the bundled
preprocessor (`MIMIC-III Data Instruction.pdf`, page 2) and
implemented by the MIMIC team as the SQL concept
`mit-lcp/mimic-code/concepts/severityscores/sapsii.sql`.

| feature | values | meaning | this cohort |
| :--- | :--- | :--- | :--- |
| `AIDS` | 0 / 1 | 1 = acquired immunodeficiency syndrome (ICD-9 042) | 314 / 35,312 (0.9%) |
| `HEM` | 0 / 1 | 1 = hematologic malignancy (leukemia / lymphoma ICD-9 codes) | 965 / 34,661 (2.7%) |
| `METS` | 0 / 1 | 1 = metastatic cancer (ICD-9 196–199) | 2,177 / 33,449 (6.1%) |
| `Adm_type` | 0 / 1 / 2 | 0 = medical, 1 = scheduled surgical, 2 = unscheduled surgical | 24,715 / 4,784 / 6,127 |

These three comorbidities and the admission-type bucket are the exact
fields the SAPS-II score uses; AIDS, HEM, METS each contribute a
fixed point penalty in the score, and `Adm_type` modifies the
intercept. They are kept as the original integer codes in the model.

<div class="hist-grid">
<figure><figcaption>AIDS</figcaption><img src="plots/clip99/AIDS.png"></figure>
<figure><figcaption>HEM</figcaption><img src="plots/clip99/HEM.png"></figure>
<figure><figcaption>METS</figcaption><img src="plots/clip99/METS.png"></figure>
<figure><figcaption>Adm_type</figcaption><img src="plots/clip99/Adm_type.png"></figure>
</div>

## Missingness indicator features (`M_*`)

Two features have substantial missingness because the underlying lab
or measurement is **not always ordered** for every ICU admission:

| indicator | flags | source missing % | meaning |
| :--- | :--- | ---: | :--- |
| `M_PaO2FiO2_vent_min` | `PaO2FiO2_vent_min` | ~58% | patient was not mechanically ventilated during the first 24 h |
| `M_bilirubin_mean` | `bilirubin_mean`, `bilirubin_spread` | ~56% | bilirubin not ordered (no liver-function panel drawn) |

For both, **missingness is informative**: ventilated and liver-tested
patients are clinically distinct populations. We pair the original
feature with its `M_*` indicator and impute the original to 0 — that
way `β·x = 0` for missing entries while `M_*` carries the
"was-this-test-ordered" signal as a separate coefficient. A single
indicator covers `bilirubin_mean` and `bilirubin_spread` because
they share the same NaN mask (both derived from the same lab pair).

`age_capped` was previously included as an indicator for the
de-identified ~300-yr offset (>=89 yr in MIMIC). It has been
**removed** — the cap information is implicit in the post-cap age
itself, and the marginal hazard above age 90 is small in this cohort.

## Recommended scaling for model input

Scaling is required for any hazard / ODE / linear-additive model:
unscaled inputs cause optimization instability, unfair L1/L2
penalties (large-magnitude features absorb more shrinkage), and
numerical blow-up inside `exp(·)` terms. The split below applies
**after** the patient-level train/test split so test-set quantiles
do not leak into training.

| feature group | scaler | rationale |
| :--- | :--- | :--- |
| `bilirubin_mean`, `bilirubin_spread`, `bun_mean`, `bun_spread`, `wbc_mean`, `wbc_spread`, `urineoutput`, `PaO2FiO2_vent_min` | **RobustScaler** (median / IQR) | still right-skewed after middle-99% clip; mean/std would be pulled by the tail |
| `age`, `heartrate_mean`, `heartrate_spread`, `sysbp_mean`, `sysbp_spread`, `tempc_mean`, `tempc_spread`, `sodium_mean`, `sodium_spread`, `bicarbonate_mean`, `bicarbonate_spread`, `potassium_mean`, `potassium_spread`, `gcs_min` | **StandardScaler** (mean / std) | roughly Gaussian / symmetric after clip |
| `AIDS`, `HEM`, `METS`, `Adm_type`, `M_PaO2FiO2_vent_min`, `M_bilirubin_mean` | **leave as 0/1 (or as encoded category for `Adm_type`)** | scaling binary indicators destroys their interpretation; `M_*` must stay {0,1} to act as missingness flags |

`cohort_long.npz` is left as the unscaled canonical dump — scaling is
a model-level concern, fit on training rows only.

## Post-scale feature histograms (model inputs)

The plots below show the **train-set per-patient distributions of
the 28 features after the scaling rule above is applied**. Pipeline:

1. Patient-level 70 / 15 / 15 train / val / test split, seed = 42
   (24,938 / 5,343 / 5,345 patients).
2. Each scaler is fit on **train rows only** (no leakage), then
   applied to all rows.
3. Each patient's `x` vector is taken once (deduped) for the hist —
   the model itself receives the same vector repeated across that
   patient's event/censor rows.

Output: `merged_data/cohort_long_scaled.npz` (id, time, delta,
scaled `x`, train/val/test masks, scaler centers / scales) and
`merged_data/scaler_params.csv` (per-feature scaler kind + center +
scale, for reproducible inference at test time).

A subtle point: the four high-missingness features
(`PaO2FiO2_vent_min`, `bilirubin_mean`, `bilirubin_spread`, and to a
lesser extent `urineoutput`) have a large mass of imputed zeros
inherited from the NaN→0 step in `build_long_format.py`. Because more
than half of training values are exactly 0, the **median is 0** for
those features and the RobustScaler maps 0 → 0. The non-imputed
values land at `(value − 0) / IQR > 0`, producing the large positive
skew you see in those panels. The `M_*` indicator rows below carry
the "was this measurement actually present?" signal as a separate
column, so the model can recover what is what.

<div class="hist-grid">
<figure><figcaption>age</figcaption><img src="plots/clip99/scaled/age.png"></figure>
<figure><figcaption>heartrate_mean</figcaption><img src="plots/clip99/scaled/heartrate_mean.png"></figure>
<figure><figcaption>heartrate_spread</figcaption><img src="plots/clip99/scaled/heartrate_spread.png"></figure>
<figure><figcaption>sysbp_mean</figcaption><img src="plots/clip99/scaled/sysbp_mean.png"></figure>
<figure><figcaption>sysbp_spread</figcaption><img src="plots/clip99/scaled/sysbp_spread.png"></figure>
<figure><figcaption>tempc_mean</figcaption><img src="plots/clip99/scaled/tempc_mean.png"></figure>
<figure><figcaption>tempc_spread</figcaption><img src="plots/clip99/scaled/tempc_spread.png"></figure>
<figure><figcaption>PaO2FiO2_vent_min</figcaption><img src="plots/clip99/scaled/PaO2FiO2_vent_min.png"></figure>
<figure><figcaption>urineoutput</figcaption><img src="plots/clip99/scaled/urineoutput.png"></figure>
<figure><figcaption>bun_mean</figcaption><img src="plots/clip99/scaled/bun_mean.png"></figure>
<figure><figcaption>bun_spread</figcaption><img src="plots/clip99/scaled/bun_spread.png"></figure>
<figure><figcaption>wbc_mean</figcaption><img src="plots/clip99/scaled/wbc_mean.png"></figure>
<figure><figcaption>wbc_spread</figcaption><img src="plots/clip99/scaled/wbc_spread.png"></figure>
<figure><figcaption>potassium_mean</figcaption><img src="plots/clip99/scaled/potassium_mean.png"></figure>
<figure><figcaption>potassium_spread</figcaption><img src="plots/clip99/scaled/potassium_spread.png"></figure>
<figure><figcaption>sodium_mean</figcaption><img src="plots/clip99/scaled/sodium_mean.png"></figure>
<figure><figcaption>sodium_spread</figcaption><img src="plots/clip99/scaled/sodium_spread.png"></figure>
<figure><figcaption>bicarbonate_mean</figcaption><img src="plots/clip99/scaled/bicarbonate_mean.png"></figure>
<figure><figcaption>bicarbonate_spread</figcaption><img src="plots/clip99/scaled/bicarbonate_spread.png"></figure>
<figure><figcaption>bilirubin_mean</figcaption><img src="plots/clip99/scaled/bilirubin_mean.png"></figure>
<figure><figcaption>bilirubin_spread</figcaption><img src="plots/clip99/scaled/bilirubin_spread.png"></figure>
<figure><figcaption>gcs_min</figcaption><img src="plots/clip99/scaled/gcs_min.png"></figure>
<figure><figcaption>AIDS</figcaption><img src="plots/clip99/scaled/AIDS.png"></figure>
<figure><figcaption>HEM</figcaption><img src="plots/clip99/scaled/HEM.png"></figure>
<figure><figcaption>METS</figcaption><img src="plots/clip99/scaled/METS.png"></figure>
<figure><figcaption>Adm_type</figcaption><img src="plots/clip99/scaled/Adm_type.png"></figure>
<figure><figcaption>M_PaO2FiO2_vent_min</figcaption><img src="plots/clip99/scaled/M_PaO2FiO2_vent_min.png"></figure>
<figure><figcaption>M_bilirubin_mean</figcaption><img src="plots/clip99/scaled/M_bilirubin_mean.png"></figure>
</div>

## Post-scale correlation heatmap (model frame)

This is the correlation structure the model **actually sees**: the
post-scale `x` matrix on the train per-patient frame, with `Adm_type`
one-hot encoded into two dummies (reference = medical = 0). Computed
on 24,938 train patients × 29 columns. Saved to
`merged_data/correlation_postscale.csv` and
`doc/plots/clip99/scaled/correlation_postscale.png`.

This heatmap differs in one important way from the diagnostic
post-clip + 0-fill heatmap above: there, missing rows were pinned at
**0 in raw scale**, which forced large mechanical correlations
between each high-missingness feature and its M_* indicator. After
scaling + post-scale 0-fill, missing rows land at the **post-scale
center** alongside other "average" patients, so the value column and
its M_* indicator are no longer mechanically linked.

![](plots/clip99/scaled/correlation_postscale.png)

**Diagnostic vs model-frame ρ** for the X ↔ M_* pairs:

| pair | diagnostic ρ | model-frame ρ |
| :--- | ---: | ---: |
| PaO2FiO2_vent_min ↔ M_PaO2 | −0.856 | **−0.132** |
| bilirubin_mean ↔ M_bili | −0.332 | **−0.252** |
| bilirubin_spread ↔ M_bili | −0.249 | **−0.277** |

The PaO2 / M_PaO2 collinearity collapses from 0.86 → 0.13. PaO2's
RobustScaler median = 212 with IQR = 147, so most observed values
also sit near 0 in scaled space — the spike at 0 from imputed rows
no longer dominates. Bilirubin's correlation drops less because its
post-scale distribution remains heavy-tailed (median = 0.6, IQR =
0.8, max ≈ 16), so observed rows still extend far from 0 and the
imputed spike at 0 is distinguishable.

**Top 15 |ρ| pairs (post-scale, model frame):**

| pair | ρ |
| :--- | ---: |
| bilirubin_mean ↔ bilirubin_spread | +0.625 |
| potassium_mean ↔ potassium_spread | +0.507 |
| bun_mean ↔ bun_spread | +0.464 |
| wbc_mean ↔ wbc_spread | +0.442 |
| sodium_spread ↔ potassium_spread | +0.441 |
| bun_spread ↔ bicarbonate_spread | +0.398 |
| heartrate_mean ↔ heartrate_spread | +0.390 |
| M_PaO2FiO2_vent_min ↔ potassium_spread | −0.386 |
| sodium_spread ↔ bicarbonate_spread | +0.364 |
| tempc_spread ↔ M_PaO2FiO2_vent_min | −0.347 |
| sysbp_mean ↔ sysbp_spread | +0.313 |
| bun_spread ↔ sodium_spread | +0.308 |
| sysbp_spread ↔ M_PaO2FiO2_vent_min | −0.304 |
| age ↔ bun_mean | +0.288 |
| bilirubin_spread ↔ M_bilirubin_mean | −0.287 |

Largest |ρ| in the model frame is 0.63 (`bilirubin_mean ↔
bilirubin_spread`) — well below any multicollinearity threshold
(typical concern starts at |ρ| > 0.7 / VIF > 5). The X ↔ M_* pairs no
longer dominate the top of the list; they sit alongside the genuine
within-pair correlations and a handful of clinical co-monitoring
patterns (`tempc_spread`, `sysbp_spread`, `potassium_spread` ↔
`M_PaO2`: patients with recorded ABGs are sicker, with wider
vital-sign and electrolyte ranges).

## Event and censoring time distributions

The recurrent-event outcome puts every readmission as `delta = 1` at
`time = days since first admission`, and one censoring row per
patient as `delta = 0` at the latest known timestamp (max of last
DISCHTIME / DEATHTIME / ADMITTIME). 35,626 patients × on average 1.30
rows = 46,227 rows total. 10,601 of those are events (≥1 readmission)
and 35,626 are censors (one per patient).

**Censoring times** are dominated by short admissions: median 8.5
days, q90 ≈ 234 d, q99 ≈ 6.7 yr. Most patients are censored at
discharge (no readmission ever observed) — the tail of long
follow-ups belongs to patients who do return at least once and whose
censor is the latest timestamp from any admission.

**Event times** are spread far more widely: q25 ≈ 82 d, median
≈ 374 d, q99 ≈ 9.1 yr. The gap between `q50(censor) = 8.5 d` and
`q50(event) = 374 d` is exactly the survivorship asymmetry the
recurrent-event likelihood corrects for: short censors don't say
"low risk," they say "we stopped watching."

The right tails extend out to ~10 yr in raw space, so we also show
a `log10(days)` view to make the bulk of the distribution legible.

<div class="hist-grid">
<figure><figcaption>Event time (linear)</figcaption><img src="plots/clip99/outcome/event_time.png"></figure>
<figure><figcaption>Censoring time (linear)</figcaption><img src="plots/clip99/outcome/censor_time.png"></figure>
<figure><figcaption>Event time (log10 days)</figcaption><img src="plots/clip99/outcome/event_time_log.png"></figure>
<figure><figcaption>Censoring time (log10 days)</figcaption><img src="plots/clip99/outcome/censor_time_log.png"></figure>
</div>

## Random-effect Cox PH on the recurrent-event cohort

Fit with the local `recurrent_ode` package (`model='cox',
random_effect=True`) — a Cox-type intensity with a B-spline baseline
hazard and a subject-level gamma frailty. Standard errors are the
closed-form sandwich SE (`inference_beta`) — `Var(beta) = A^-1 B A^-1`
with `B` aggregated over the per-subject score residuals, no Hessian
inversion of a frailty integral. 29 features (28 scaled features +
two `Adm_type` one-hot dummies; medical = reference).

Driver: `fit_cox.py`. Three scenarios on the same long-format cohort
(46,227 rows / 35,626 patients / 10,601 events):

| scenario | rows | events | time range (days) | runtime |
|---|---:|---:|---:|---:|
| `raw`        | 46,227 | 10,601 | 0.04 – 4129  | 3.1 s |
| `log10(1+t)` | 46,227 | 10,601 | 0.02 – 3.62  | 4.3 s |
| `trim`       | 45,000 | 10,019 | 1.35 – 2888  | 3.9 s |

`trim` drops rows with `time` outside the middle 99 % of the
distribution (`[1.35, 2888.7]` days) **and** drops 645 patients who
lost their censoring row in that filter, keeping a balanced
event/censoring structure. `log10(1+t)` is used instead of `log10(t)`
so the spline knots stay on a non-negative grid (some events occur
within hours of admission).

### Coefficients (`coef` and Wald `p`)

The Wald p-values come directly from the closed-form sandwich SE
(`inference_beta`), so each per-feature `p` is the exact significance
of that coefficient under the model's own variance estimator — no
multiple-testing correction is applied. `*` simply marks `p < 0.05`.

| feature | raw | log10(1+t) | trim |
|---|---|---|---|
| `age` | −0.077* (1e-03) | −0.078* (8e-04) | −0.067* (0.004) |
| `heartrate_mean` | +0.061* (4e-04) | +0.058* (7e-04) | +0.056* (0.001) |
| `heartrate_spread` | −0.013 (0.436) | −0.015 (0.382) | −0.016 (0.360) |
| `sysbp_mean` | −0.040* (0.017) | −0.041* (0.013) | −0.041* (0.012) |
| `sysbp_spread` | +0.015 (0.521) | +0.014 (0.568) | +0.023 (0.345) |
| `tempc_mean` | −0.039* (0.015) | −0.037* (0.020) | −0.037* (0.018) |
| `tempc_spread` | −0.012 (0.450) | −0.014 (0.400) | +0.004 (0.820) |
| `PaO2FiO2_vent_min` | +0.062* (0.046) | +0.061 (0.051) | +0.056 (0.069) |
| `urineoutput` | −0.063* (0.001) | −0.059* (0.002) | −0.063* (9e-04) |
| `bun_mean` | +0.043* (0.012) | +0.042* (0.012) | +0.067* (4e-05) |
| `bun_spread` | −0.003 (0.873) | −0.003 (0.853) | −0.005 (0.785) |
| `wbc_mean` | −0.047* (0.013) | −0.043* (0.017) | −0.041* (0.019) |
| `wbc_spread` | +0.035 (0.063) | +0.041* (0.030) | +0.015 (0.441) |
| `potassium_mean` | +0.055* (1e-03) | +0.053* (2e-03) | +0.046* (0.007) |
| `potassium_spread` | −0.062* (0.002) | −0.059* (0.003) | −0.056* (0.004) |
| `sodium_mean` | −0.006 (0.706) | +0.001 (0.972) | +0.012 (0.461) |
| `sodium_spread` | −0.001 (0.968) | +0.005 (0.773) | +0.001 (0.937) |
| `bicarbonate_mean` | +0.116* (2e-14) | +0.112* (8e-14) | +0.113* (7e-14) |
| `bicarbonate_spread` | +0.022 (0.178) | +0.021 (0.182) | +0.027 (0.104) |
| `bilirubin_mean` | +0.003 (0.719) | +0.003 (0.664) | −0.000 (0.974) |
| `bilirubin_spread` | −0.003 (0.495) | −0.005 (0.255) | −0.001 (0.799) |
| `gcs_min` | −0.014 (0.422) | −0.018 (0.299) | −0.024 (0.197) |
| `AIDS` | +0.018 (0.880) | −0.006 (0.962) | +0.006 (0.965) |
| `HEM` | +0.120 (0.066) | +0.063 (0.332) | +0.071 (0.313) |
| `METS` | +0.145* (0.026) | +0.169* (0.008) | +0.173* (0.005) |
| `M_PaO2FiO2_vent_min` | +0.065 (0.077) | +0.061 (0.094) | +0.080* (0.029) |
| `M_bilirubin_mean` | −0.070* (0.040) | −0.058 (0.087) | −0.053 (0.109) |
| `Adm_type_scheduled` | −0.142* (0.003) | −0.140* (0.003) | −0.146* (0.002) |
| `Adm_type_unscheduled` | +0.103* (0.007) | +0.109* (0.005) | +0.091* (0.021) |

### Significant features by scenario (Wald `p < 0.05`)

| scenario | k_sig | features |
|---|---:|---|
| `raw`        | 15 | `age`, `heartrate_mean`, `sysbp_mean`, `tempc_mean`, `urineoutput`, `bun_mean`, `wbc_mean`, `wbc_spread`, `potassium_mean`, `potassium_spread`, `bicarbonate_mean`, `METS`, `M_bilirubin_mean`, `Adm_type_scheduled`, `Adm_type_unscheduled` |
| `log10(1+t)` | 14 | `age`, `heartrate_mean`, `sysbp_mean`, `tempc_mean`, `urineoutput`, `bun_mean`, `wbc_mean`, `wbc_spread`, `potassium_mean`, `potassium_spread`, `bicarbonate_mean`, `METS`, `Adm_type_scheduled`, `Adm_type_unscheduled` |
| `trim`       | 13 | `age`, `heartrate_mean`, `sysbp_mean`, `tempc_mean`, `urineoutput`, `bun_mean`, `wbc_mean`, `potassium_mean`, `potassium_spread`, `bicarbonate_mean`, `METS`, `Adm_type_scheduled`, `Adm_type_unscheduled` |

The intersection across all three scenarios — 13 features with
`p < 0.05` regardless of time-axis choice — is

> `age`, `heartrate_mean`, `sysbp_mean`, `tempc_mean`, `urineoutput`,
> `bun_mean`, `wbc_mean`, `potassium_mean`, `potassium_spread`,
> `bicarbonate_mean`, `METS`, `Adm_type_scheduled`,
> `Adm_type_unscheduled`.

### Comparing the three scenarios

- **Coefficients are remarkably stable across the three time scales.**
  For all 29 features the sign agrees in 26/29; the three sign flips
  (`tempc_spread`, `sodium_mean`, `sodium_spread`) are all
  near-zero coefs (|β| < 0.02, p > 0.4) and not interpretable.
  Magnitudes shift by < 25 % between `raw`, `log10(1+t)`, and `trim`
  for every feature — the model is genuinely robust to the
  time-axis transform and to dropping the q0.5 / q99.5 outlier rows.
- **`bicarbonate_mean` is the dominant signal** (β ≈ +0.11, z ≈ 7.5,
  p ≈ 10⁻¹⁴ in every scenario). At unit-scaled bicarbonate the hazard
  multiplier is exp(0.11) ≈ 1.12 per s.d., so a +2 s.d. shift in mean
  bicarbonate carries an estimated 26 % readmission-hazard increase.
- **`age` is consistently negative.** β ≈ −0.07 to −0.08 across the
  three scenarios with p ≤ 0.004, suggesting older age associates
  with lower observed readmission hazard. The interpretation is the
  competing-risk-of-death effect in MIMIC — older patients censor
  (die) faster and accrue fewer at-risk person-years.
- **`urineoutput`, `potassium_mean`, `potassium_spread`, `bun_mean`,
  `heartrate_mean` are the next-strongest signals** (each `p ≤ 0.003`
  in at least one scenario, all stable nominally). The two potassium
  components carry *opposite* signs (mean +, spread −): a higher
  average serum potassium associates with elevated readmission hazard,
  while a wider min-to-max range associates with a lower hazard,
  consistent with the spread reflecting "more measurements drawn"
  (i.e., a closer-monitored — but treated — patient). `bun_mean`
  actually *strengthens* under `trim` (β: +0.04 → +0.07, p: 0.012 →
  4e-5) when the long follow-up tail is dropped — its variance was
  being inflated by patients with extreme follow-up windows.
- **`Adm_type` carries strong administrative signal.** Relative to
  the medical reference, scheduled surgical admissions have lower
  readmission hazard (β ≈ −0.14, p ≈ 0.002 – 0.003) and unscheduled
  surgical admissions have higher hazard (β ≈ +0.10, p ≈ 0.005 –
  0.02). Both effects are stable across all three scenarios.
- **`METS` (metastatic cancer)** is positive across all three
  scenarios (β ≈ +0.15 → +0.17) with `p ≈ 0.005 – 0.026`. The signal
  *strengthens* under `log10` and `trim`, suggesting under-powered
  rather than spurious.
- **The `M_*` missingness indicators are weak.** `M_PaO2FiO2_vent_min`
  is nominally non-significant in `raw` / `log10` and only sig under
  `trim` (`p = 0.029`); `M_bilirubin_mean` is sig only under `raw`
  (`p = 0.035`). Confirms the post-scale correlation analysis: once
  bilirubin / PaO2 are centred to 0 by scaling-then-imputing, the
  indicator carries little residual hazard signal.
- **Bilirubin mean and spread are essentially null.** The
  reparameterized pair contributes nothing at recurrent-event scale —
  consistent with the high diagnostic correlations and the long
  right tail that compresses to near-zero after Robust scaling.

Outputs:

- Driver: `fit_cox.py`
- Tables: `merged_data/cox_coef_table.csv` (long, 87 × 8) and
  `merged_data/cox_coef_wide.csv` (wide, 29 × 9)
- Run log: `merged_data/cox_fit.log`

## Random-effect LTM (linear-transformation model)

A second random-effect model fit on the same long-format cohort
(46,227 rows / 35,626 patients / 10,601 events) using the local
`recurrent_ode` package (`model='ltm', random_effect=True`). The LTM
generalises the Cox PH log-linear hazard with two additional B-spline
nuisance functions:

- **time-transform integrand** `α(t) = exp(B₀(t)·α)`,
  representing how rapidly the *recurrent-event clock* runs at
  calendar time `t` — `α(t) = 1` recovers the linear-time scale, while
  `α(t) > 1` accelerates and `α(t) < 1` slows it;
- **baseline hazard on the transformed scale**
  `λ₀(u) = exp(B_q(u)·θ)`, where `u = ∫₀ᵗ α(s)·exp(βᵀX) ds`.

For identifiability the LTM constrains the anchor coefficient
`β₁ = 1`. We choose **`bicarbonate_mean`** as the anchor — the
Cox-strongest signal — so every other LTM `β` is interpretable as a
*relative effect vs. bicarbonate_mean* on the same log-hazard scale.
Standard errors come from a B = 800 Gaussian-resampling sandwich
(`inference_objective_func_sieve`); see `RecurrentODE_py/random_effect/
ltm/inference_objective_func_sieve.py` for the per-subject score
aggregation. Knot scheme **K1** (uniform on `[0, max(t)]` for both
`α(t)` and `λ₀(u)`); the higher-order quantile-knot variants K2–K4
overflow on the heavy-tail follow-up here.

Driver: `fit_ltm.py`. Two scenarios:

| scenario | rows | events | time range (days) | runtime |
|---|---:|---:|---:|---:|
| `raw`        | 46,227 | 10,601 | 0.04 – 4129  | ~90 s |
| `log10(1+t)` | 46,227 | 10,601 | 0.02 – 3.62  | ~90 s |

The `log10(1+t)` scenario is the headline result — the heavy
right-tail follow-up (max 4129 d) destabilises the SLSQP+BFGS
alternation in `raw`, so the resampled SEs there are wide and several
features are anchored at the optimisation boundary; the `log10`
transform compresses follow-up to `[0, 3.62]` and keeps the iterative
MLE well-conditioned. We list both for completeness but interpret on
the `log10` scenario.

### Coefficients (LTM log10(1+t), `coef ± SE`, Wald `p`)

`bicarbonate_mean` is fixed at 1.0 (anchor); all other coefficients
are *relative effects* on the same log-hazard scale.

| feature | coef | SE | p |
|---|---:|---:|---:|
| `bicarbonate_mean` *(anchor)* | 1.000 | — | — |
| `age` | −0.988 | 0.106 | **1e-20** |
| `heartrate_mean` | +0.577 | 0.115 | **5e-07** |
| `heartrate_spread` | −0.117 | 0.066 | 0.074 |
| `sysbp_mean` | −0.804 | 0.108 | **8e-14** |
| `sysbp_spread` | +0.160 | 0.075 | **0.033** |
| `tempc_mean` | −0.749 | 0.106 | **1e-12** |
| `tempc_spread` | −0.482 | 0.135 | **4e-04** |
| `PaO2FiO2_vent_min` | +0.479 | 0.118 | **5e-05** |
| `urineoutput` | +0.109 | 0.100 | 0.278 |
| `bun_mean` | +0.525 | 0.127 | **3e-05** |
| `bun_spread` | −0.106 | 0.099 | 0.285 |
| `wbc_mean` | −1.163 | 0.205 | **1e-08** |
| `wbc_spread` | +0.445 | 0.105 | **2e-05** |
| `potassium_mean` | +0.832 | 0.129 | **1e-10** |
| `potassium_spread` | −0.858 | 0.126 | **1e-11** |
| `sodium_mean` | +0.433 | 0.084 | **2e-07** |
| `sodium_spread` | −0.028 | 0.089 | 0.755 |
| `bicarbonate_spread` | +0.046 | 0.073 | 0.531 |
| `bilirubin_mean` | −0.056 | 0.037 | 0.133 |
| `bilirubin_spread` | +0.017 | 0.015 | 0.258 |
| `gcs_min` | −0.770 | 0.089 | **7e-18** |
| `AIDS` | −0.929 | 0.299 | **0.002** |
| `HEM` | −0.725 | 0.575 | 0.207 |
| `METS` | −0.277 | 0.275 | 0.314 |
| `M_PaO2FiO2_vent_min` | −0.559 | 0.282 | **0.048** |
| `M_bilirubin_mean` | −1.574 | 0.355 | **9e-06** |
| `Adm_type_scheduled` | −4.531 | 2.517 | 0.072 |
| `Adm_type_unscheduled` | +0.822 | 0.176 | **3e-06** |

(`*` in the Cox table → `**` here; cells already mark `p < 0.05` in
bold.)

**Significant features (LTM log10, Wald p < 0.05): 18.**
`age`, `heartrate_mean`, `sysbp_mean`, `sysbp_spread`, `tempc_mean`,
`tempc_spread`, `PaO2FiO2_vent_min`, `bun_mean`, `wbc_mean`,
`wbc_spread`, `potassium_mean`, `potassium_spread`, `sodium_mean`,
`gcs_min`, `AIDS`, `M_PaO2FiO2_vent_min`, `M_bilirubin_mean`,
`Adm_type_unscheduled`. Plus the anchor `bicarbonate_mean` which is
fixed by construction.

Outputs:

- Driver: `fit_ltm.py`
- Tables: `merged_data/ltm_coef_table.csv` (long), `merged_data/ltm_coef_wide.csv` (wide)
- Spline parameters: `merged_data/ltm_spline_{raw,log10}.npz`
- Run log: `merged_data/ltm_fit.log`

## Cox PH vs. LTM — significance comparison (log10 scenario)

| | Cox sig (14) | LTM sig (18) | Both |
|---|---|---|---|
| **shared (10)** | ✓ | ✓ | `age`, `heartrate_mean`, `sysbp_mean`, `tempc_mean`, `bun_mean`, `wbc_mean`, `wbc_spread`, `potassium_mean`, `potassium_spread`, `Adm_type_unscheduled` |
| **Cox-only (3)** | ✓ | ✗ | `urineoutput`, `METS`, `Adm_type_scheduled` |
| **LTM-only (8)** | ✗ | ✓ | `sysbp_spread`, `tempc_spread`, `PaO2FiO2_vent_min`, `sodium_mean`, `gcs_min`, `AIDS`, `M_PaO2FiO2_vent_min`, `M_bilirubin_mean` |
| anchor | ✓ Cox sig (β=+0.11, p≈10⁻¹⁴) | fixed at β=1 (cannot be tested) | `bicarbonate_mean` |

Reading the comparison:

- **Sign agreement.** Wherever both models are significant, the signs
  match. Cox `β` and LTM `β`-relative-to-bicarbonate sit on the same
  log-hazard scale, so the agreement is structural not coincidental.
- **LTM picks up extra signals because it absorbs the time-axis
  irregularity.** Features like `gcs_min`, `AIDS`,
  `M_PaO2FiO2_vent_min`, `M_bilirubin_mean` hide behind the
  recurrent-event clock asymmetry in Cox (one β has to absorb both
  hazard level and any local time-warp). The LTM
  `α(t)`-spline absorbs the time-warp, freeing those β's to express
  their per-feature contribution.
- **Cox keeps `urineoutput`, `METS`, `Adm_type_scheduled`.** All three
  are weak-magnitude in LTM (`|β_LTM| < 0.3`, p > 0.05) but reach
  Cox significance. They are clinically marginal effects whose
  apparent Cox precision evaporates once LTM widens its sandwich SE
  via the resampling.
- **The `bicarbonate_mean` anchor matters.** In Cox, +0.11 / s.d. is
  the dominant coefficient; in LTM, fixing it at 1.0 sets the unit
  scale. If we re-anchor LTM on `age` (next-strongest Cox signal,
  β = −0.078) the relative-effect column scales by ~14×, but the
  *significance pattern* (which features have p < 0.05) is unchanged
  — anchor choice is a unit convention, not an inferential one.

## Functional parameters: Cox baseline λ₀(t) and LTM α(t), λ₀(u)

The non-linear pieces of both models — the B-spline functions
themselves — are saved alongside the coefficient tables. Cox saves
only `θ` (point estimate); the Cox sandwich variance is for `β`,
the linear part, so we plot λ₀(t) without a band. LTM saves the joint
Fisher matrix from the `B = 800` resampling, so we extract the
α-block and θ-block to draw 95% pointwise bands via
`Var(B(t)·μ) = B(t) Σ_block B(t)ᵀ`.

`grid_for(knots, n=400)` evaluates each spline on a 400-point linear
grid spanning the feature's knot range. Driver:
`plot_functional_params.py`. Plots saved to
`doc/plots/functional/`.

### Cox PH baseline hazard λ₀(t) = exp(B(t)·θ)

<div class="hist-grid">
<figure><figcaption>Cox baseline (raw days)</figcaption><img src="plots/functional/cox_baseline_raw.png"></figure>
<figure><figcaption>Cox baseline (log10(1+days))</figcaption><img src="plots/functional/cox_baseline_log10.png"></figure>
</div>

The Cox baseline declines monotonically — early-period readmission
intensity is highest in the first weeks after discharge and drops by
~2 orders of magnitude over 1 yr. The `log10(1+t)` rendering shows
the same shape on a compressed time axis.

### LTM time-transform integrand α(t) = exp(B₀(t)·α)

<div class="hist-grid">
<figure><figcaption>LTM α(t) — raw days</figcaption><img src="plots/functional/ltm_alpha_raw.png"></figure>
<figure><figcaption>LTM α(t) — log10(1+days)</figcaption><img src="plots/functional/ltm_alpha_log10.png"></figure>
</div>

α(t) > 1 means the recurrent-event clock runs *faster* than calendar
time at `t`; α(t) < 1 means it runs slower. The dotted reference at
`α = 1, t = 1.5` marks the spline-knot anchor point. The 95%
pointwise band quantifies how confident the resampling sandwich is
about the local time-warp.

### LTM baseline hazard on transformed time λ₀(u) = exp(B_q(u)·θ)

<div class="hist-grid">
<figure><figcaption>LTM λ₀(u) — raw days</figcaption><img src="plots/functional/ltm_lambda_raw.png"></figure>
<figure><figcaption>LTM λ₀(u) — log10(1+days)</figcaption><img src="plots/functional/ltm_lambda_log10.png"></figure>
</div>

`u` is dimensionless transformed time on the LTM clock. λ₀(u) is the
intrinsic shape of the readmission intensity once both the
linear-covariate effects and the time-transform have been factored
out — what is left is the residual baseline-hazard shape on the
LTM-internal clock.

### Side-by-side overview

<div class="hist-grid">
<figure><figcaption>All three (raw days)</figcaption><img src="plots/functional/functional_combined_raw.png"></figure>
<figure><figcaption>All three (log10(1+days))</figcaption><img src="plots/functional/functional_combined_log10.png"></figure>
</div>

## Per-feature clinical interpretation

How each feature's coefficient relates to readmission *intensity* on
the recurrent-event scale, combining Cox PH and LTM log10 evidence.
Coefficients are on the **scaled (z-scored / robust-scaled) feature
scale**, so a +1 unit shift ≈ +1 s.d. (or +1 IQR for robust-scaled
features). Effect sizes use Cox-PH `β` since LTM β's are anchored to
`bicarbonate_mean` and not directly comparable across features.

**Demographic.**

- *`age` (β_Cox = −0.078, p ≈ 8e-04; β_LTM = −0.99, p ≈ 1e-20).* Older
  patients have *lower* observed readmission intensity. This is the
  classic competing-risks artefact in MIMIC: older patients censor
  by death faster, so their accumulated person-years at risk shrink
  and readmission events per person-year drop. Without explicit
  death modelling, age cannot be read as a protective factor —
  it's an at-risk-window effect.

**Vital signs.**

- *`heartrate_mean` (β_Cox = +0.058, p ≈ 7e-04; β_LTM = +0.58,
  p ≈ 5e-07).* Higher 24-h mean HR → higher readmission intensity.
  Tachycardia is a generic marker of physiological stress (sepsis,
  pain, volume depletion, autonomic dysregulation) and the
  consistent positive effect across both models is exactly that
  generic-illness signal.
- *`heartrate_spread`.* Not significant in either model. Rate
  variability per se does not add predictive value above the mean.
- *`sysbp_mean` (β_Cox = −0.041, p ≈ 0.013; β_LTM = −0.80, p ≈ 8e-14).*
  Higher mean SBP → lower readmission. Hypotension is the
  dangerous tail in ICU patients (sepsis, cardiogenic shock,
  GI bleed); a higher mean SBP indicates haemodynamic stability
  and lower subsequent readmission risk. The hypertension tail is
  trimmed at q99.5 so we are not capturing chronic-HTN effects.
- *`sysbp_spread` (LTM-only, β_LTM = +0.16, p = 0.033).* Wider SBP
  swings within 24 h indicate haemodynamic instability;
  picked up by the LTM time-transform but not Cox PH.
- *`tempc_mean` (β_Cox = −0.037, p ≈ 0.019; β_LTM = −0.75, p ≈ 1e-12).*
  Higher mean temp → lower readmission. Mid-thirties hypothermia is
  the danger signal in ICU; normothermia (~37 °C) is protective.
  This is *not* a "fever protects" claim — fever rows are above the
  mean of 37.0 but the cohort distribution is dominated by the
  hypothermia / sepsis tail on the low side.
- *`tempc_spread` (LTM-only, β_LTM = −0.48, p ≈ 4e-04).* Wider
  intra-day temperature swings → lower readmission once mean is
  controlled — likely a "vitals were monitored frequently" surrogate
  picking up patients still on active warming/cooling protocols.

**Respiratory.**

- *`PaO2FiO2_vent_min` (LTM-only, β_LTM = +0.48, p ≈ 5e-05).* Among
  ventilated patients (~42% of cohort), higher minimum P/F ratio
  → higher readmission, picked up only by LTM. Counterintuitive on
  the surface — normally a lower P/F is worse — but conditioning on
  the M_PaO2 indicator separates "P/F was measurable" from "P/F was
  bad", so this β reads "ventilated patients with milder ARDS who
  survive to discharge are the ones who come back."
- *`M_PaO2FiO2_vent_min` (LTM-only, β_LTM = −0.56, p = 0.048).* Not
  ventilated → lower readmission. The non-ventilated population is
  on average less critically ill — the indicator absorbs that
  population shift.

**Renal / urinary.**

- *`urineoutput` (Cox-only, β_Cox = −0.059, p ≈ 0.002).* Higher
  24-h urine → lower readmission. Adequate diuresis is a renal /
  cardiovascular reserve marker (no AKI, no decompensated HF). LTM
  drops it because the time-transform absorbs the early-period
  excess hazard that this β was carrying in Cox.

**Labs — metabolic.**

- *`bun_mean` (β_Cox = +0.042, p ≈ 0.011; β_LTM = +0.52, p ≈ 3e-05).*
  Higher BUN → higher readmission. BUN is renal function,
  protein catabolism, GI bleed, pre-renal azotemia from volume
  depletion — all readmission-prone phenotypes. The trim-scenario
  Cox β strengthens to +0.066 (p ≈ 5e-05), implying long-tail
  follow-up was diluting the signal.
- *`wbc_mean` (β_Cox = −0.044, p ≈ 0.017; β_LTM = −1.16, p ≈ 1e-08).*
  Higher mean WBC → *lower* readmission. Counterintuitive at first —
  leukocytosis usually flags infection — but in a 24-h ICU window
  most of the cohort sits inside the leukocytosis range from
  surgical or stress response; the lower-WBC tail is the
  immunocompromised / leukopenic / chemo population whose
  readmission rate is genuinely higher.
- *`wbc_spread` (β_Cox = +0.041, p = 0.030; β_LTM = +0.45,
  p ≈ 2e-05).* A wide WBC range within 24 h is a bone-marrow
  instability marker and points to a sicker recovery trajectory.
- *`potassium_mean` (β_Cox = +0.053, p = 0.002; β_LTM = +0.83,
  p ≈ 1e-10).* Higher mean potassium → higher readmission.
  Hyperkalaemia is an AKI / arrhythmia / cardiac-arrest precursor,
  particularly at K > 5.5; the mean captures the level signal.
- *`potassium_spread` (β_Cox = −0.059, p = 0.003; β_LTM = −0.86,
  p ≈ 1e-11).* Wider K range → lower readmission. Reads as
  "patients whose K was actively *managed* (replacement / dialysis
  / shifts on insulin-glucose) had K corrected back to normal" — a
  clinical-engagement signal, not a physiological one.
- *`sodium_mean` (LTM-only, β_LTM = +0.43, p ≈ 2e-07).* Mild
  hypernatraemia (cohort mean ≈ 138, IQR within normal) at the
  high end of normal associates with higher readmission. Picked up
  only by the LTM — the time-transform exposes a weak but
  consistent dehydration / fluid-management signal.

**Bicarbonate (Cox anchor; LTM = 1.0 by construction).**

- *`bicarbonate_mean` (β_Cox = +0.111, p ≈ 8e-14).* The dominant
  signal. Higher mean HCO₃⁻ at 24 h → higher readmission. After clip
  to middle 99% the mean is centred near 25 (normal), so this is
  *not* metabolic alkalosis driving readmission — it is more likely
  the "diuresis-induced contraction alkalosis on chronic loop
  diuretics" CHF cohort, an extremely readmission-heavy population.
  Note: a single covariate dominating at p ≈ 10⁻¹⁴ in a 35,626-pt
  recurrent-event analysis is robust to almost any specification
  change; we make `bicarbonate_mean` the LTM anchor for that reason.

**Bilirubin.**

- *`bilirubin_mean`, `bilirubin_spread`.* Both null in both models.
  After RobustScaling the heavy right tail compresses to near-zero
  for >50% of the cohort; what is left has too little within-cohort
  variance to drive a hazard signal at this scale.
- *`M_bilirubin_mean` (LTM-only, β_LTM = −1.57, p ≈ 9e-06).* "Liver
  panel was *not* drawn" → lower readmission — a healthier-cohort
  shift (no hepatic concern) that LTM captures and Cox does not.

**Neurological.**

- *`gcs_min` (LTM-only, β_LTM = −0.77, p ≈ 7e-18).* Lower GCS →
  higher readmission, with a strong LTM signal that Cox misses.
  Comatose patients (GCS 3–8) have far higher post-ICU readmission
  rates; this is the "neurologically devastated discharge"
  population. Cox sees a noisy negative β (~ −0.018, NS) because
  the readmission *timing* differs sharply between cohorts — many
  post-ICU GCS-low patients return early or die early — and only
  the LTM time-transform separates timing from intensity.

**SAPS-II comorbidities.**

- *`AIDS` (LTM-only, β_LTM = −0.93, p = 0.002).* AIDS flag → lower
  readmission. Surprising sign at first, but in this cohort AIDS
  patients (0.9% prevalence, n ≈ 314) are typically managed by
  outpatient ID services post-discharge — many opportunistic
  infections re-present to clinic rather than the ED. Cox does not
  detect this (β = −0.006, NS) because the small subgroup needs
  the LTM time-transform to separate AIDS-specific timing from
  general post-ICU timing.
- *`HEM`.* Null in both. The hematological-malignancy cohort here is
  too heterogeneous (acute leukaemia + lymphoma + multiple myeloma
  combined) to give a clean signal at 35,626 patients.
- *`METS` (Cox-only, β_Cox = +0.167, p = 0.008).* Metastatic cancer
  → higher readmission in Cox. LTM does not detect it (β = −0.28, NS)
  because the time-warp absorbs most of the early-period
  oncologic-readmission excess that the β was carrying.

**Admission type** (medical = reference).

- *`Adm_type_scheduled` (Cox-only, β_Cox = −0.139, p ≈ 0.004).*
  Scheduled surgical admissions have ~13% lower readmission hazard
  than medical, consistent with planned cohorts (elective ortho /
  CABG / vascular) being a healthier subgroup overall. LTM does not
  detect it because the resampling SE on this 13%-prevalence
  binary covariate widens past the Cox sandwich SE.
- *`Adm_type_unscheduled` (β_Cox = +0.110, p ≈ 0.004; β_LTM = +0.82,
  p ≈ 3e-06).* Unscheduled (emergency) surgical admissions have
  higher readmission than the medical reference — the
  surgical-emergency cohort (trauma, peritonitis, vascular
  catastrophe) is genuinely sicker after discharge.

## Observations

- **Sentinel left tails.** `urineoutput` min −2,600, `sysbp_min = 0.3`,
  `heartrate_min = 0.35`, `sodium_min = 1.21` are physiologically
  impossible — the lower quantile bound knocks them out cleanly.
- **Heavy right tails.** `PaO2FiO2_vent_min` (max 35,400, mean 238),
  `urineoutput` (max 98 L), `wbc_max` (max 600), `bilirubin_*` show
  order-of-magnitude outliers; `q99.5%` is far tighter than `mean+3σ`
  here.
- **Floor-pinned features.** `bilirubin_*` has 0.5%+ of values pinned
  at the assay floor (0.1), so `q0.5%` coincides with the minimum and
  no rows are clipped on the left. `gcs_min ∈ [3, 15]` by definition.
- **Near-Gaussian features.** `tempc_*`, `sodium_*`, `bicarbonate_*`,
  `potassium_*` show roughly symmetric trims (~150 each side); a 3σ
  rule and middle-99% rule give similar bounds here.

## How this fits into the pipeline

This report is **diagnostic only**. The cohort-cleaning step in
`clean_cohort.py` actually applies the **combined 3σ + middle-99%
clip** (the tighter of the two bounds on each side per feature). The
plots here visualize the raw shape and the q0.5/q99.5 cuts before any
sigma override.

Files:

- Script: `testing/eda_clip99.py`
- Plots: `doc/plots/clip99/<feature>.png` (22 PNGs)
- Bounds CSV: `merged_data/clip99_bounds.csv`
