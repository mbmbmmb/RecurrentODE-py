# MIMIC-III Processed Data — Files, Shapes, and Relationships

Companion to `MIMIC Data/MIMIC-III Data Instruction.pdf`. The PDF describes
**what** each file contains; this note adds **shapes**, **column names**,
**how the files line up**, and the **subset relationships** between them.

All paths below are relative to `MIMIC Data/`.

## 1. Files in `processed-data/`

| File | Shape | Header? | Index/key |
|---|---|---|---|
| `input.csv` | **35,643 × 26** | no | row-aligned with `valid_aids.csv` |
| `valid_aids.csv` | **35,643 × 1** | no | the column **is** `HADM_ID` |
| `label.csv` | **35,304 × 3** | yes (`HADM_ID, censor, time`) | `HADM_ID` |
| `imputed-normed-ep_1_24.npz` | dict of arrays, leading dim = 35,643 | — | row-aligned with `valid_aids.csv` |

### Column names for `input.csv` (V1..V26 from the PDF)

```
V1  age                  V14 potassium_min
V2  heartrate_max        V15 potassium_max
V3  heartrate_min        V16 sodium_min
V4  sysbp_max            V17 sodium_max
V5  sysbp_min            V18 bicarbonate_min
V6  tempc_max            V19 bicarbonate_max
V7  tempc_min            V20 bilirubin_min
V8  PaO2FiO2_vent_min    V21 bilirubin_max
V9  urineoutput          V22 gcs_min
V10 bun_min              V23 AIDS         (categorical 0/1)
V11 bun_max              V24 HEM          (categorical 0/1)
V12 wbc_min              V25 METS         (categorical 0/1)
V13 wbc_max              V26 Adm_type     (categorical 0/1/2)
```

The PDF says "17 features", which expand into **26 summary statistics**
(min/max pairs for most continuous features).

### Keys in `imputed-normed-ep_1_24.npz`

```
ep_tdata          (35643, 24, 15)   time-varying features (15 vars × 24 hrs)
ep_tdata_masking  (35643, 24, 15)   missingness mask for ep_tdata
ep_data           (35643, 360)      ep_tdata flattened (24*15=360)
ep_data_masking   (35643, 360)      missingness mask for ep_data
adm_features_all  (35643, 5)        time-static admission features
adm_labels_all    (35643, 6)        admission-level labels
y_mor             (35643, 1)        in-hospital mortality
y_icd9            (35643, 20)       ICD-9 chapter labels
y_los             (35643,)          length of stay
```

The leading dimension (35,643) of every array is **row-aligned with
`valid_aids.csv`** in the same order.

## 2. Files in `raw-data/`

| File | Notes |
|---|---|
| `ADMISSIONS.csv` | source for length of stay = `DISCHTIME - ADMITTIME`; can be merged on `HADM_ID` |
| `PATIENTS.csv` | demographics; merge via `SUBJECT_ID` (chase through `ADMISSIONS` first) |

## 3. Files in `imputed-split-data/`

10 random 3:1:1 train/valid/test splits, mean-imputed and standardized
using train-fold means/stds.

```
imputed-split-data/series/   {train,valid,test}_{1..10}.pkl   time-series tensors
imputed-split-data/static/   {train,valid,test}_{1..10}.pkl   static features
```

## 4. Subset relationships

The `HADM_ID` (hospital admission ID) is the join key across all files.

```
HADM_ID(input.csv) == HADM_ID(valid_aids.csv) == leading axis of npz
                                        ⊃
                              HADM_ID(label.csv)
```

Concretely, with the IDs computed in `merge_data.py`:

| Set | |HADM_ID| | Notes |
|---|---|---|
| `valid_aids.csv` (= `input.csv`'s rows) | **35,643** | row-aligned to `input.csv` and to the npz |
| `label.csv` | **35,304** | strict subset of the above |
| `valid_aids.csv` ∩ `label.csv` | **35,304** | every label row matches an input row |
| `valid_aids.csv` ∖ `label.csv` | **339** | input rows with no label (NaN after left join) |
| `label.csv` ∖ `valid_aids.csv` | **0** | nothing label-only |

So `label.csv ⊂ input.csv` (by `HADM_ID`), and **339** of the 35,643
admissions in `input.csv` have no survival outcome.

## 5. Merge artifacts produced by `merge_data.py`

Outputs land in `mimic_analysis/merged_data/`:

| File | Shape | What it is |
|---|---|---|
| `input_named.csv` | **35,643 × 27** | `HADM_ID` + V1..V26 (named columns) |
| `merged.csv`      | **35,643 × 29** | `input_named` ⟕ `label.csv` on `HADM_ID` (left join, so all 35,643 rows kept; 339 have NaN `censor`/`time`) |

### Why `merged.csv` has more rows than `label.csv`

```
35,643 (merged.csv)
  = 35,304 (HADM_IDs in both input and label, with censor/time)
  +    339 (HADM_IDs in input only, censor/time = NaN)
```

If you only want labeled admissions, swap the left join in
`merge_data.py` for an inner join — that produces a **35,304 × 29** frame.

### Event / censoring breakdown (from `merged.csv`)

Among the **35,304** rows that have a label:

| `censor` | Count | Meaning |
|---|---|---|
| 1 | 13,765 | event observed (death) |
| 0 | 21,539 | censored (alive at end of follow-up) |

`time` is the time to mortality after admission (units per the original
benchmark — typically years).

## 6. The 339 unlabeled admissions

`diagnose_unlabeled.py` cross-references the 339 input-only `HADM_ID`s
with `ADMISSIONS.csv` and `PATIENTS.csv`. Findings:

### Cohort characteristics (unlabeled vs labeled)

| Statistic | Unlabeled (n=339) | Labeled (n=35,304) |
|---|---|---|
| Mean age at admission | **94.4 yr** | 74.4 yr |
| Median LOS | **1.2 days** | 7.1 days |
| Died in hospital (`HOSPITAL_EXPIRE_FLAG=1`) | **98.5%** | 10.0% |
| Has `DEATHTIME` recorded | **98.5%** | 10.0% |
| Has `DOD` (date of death) on file | **100%** | 36.4% |
| Emergency admission | **95.9%** | 84.0% |

The unlabeled cohort is a tightly-clustered group: very old patients,
very short stays, almost all dying in hospital. **Every single one has a
recorded date of death** in `PATIENTS.csv`.

### One admission per patient

```
339 unlabeled admissions  ↔  339 unique SUBJECT_IDs
```

None of these subjects has any *other* admission in `valid_aids.csv`.
They are all single-admission patients.

### Model interpretation: these are death-censored subjects

In the recurrent-event readmission model used here, the **event of
interest is hospital readmission**, not death. Death precludes any
future readmission, so for these 339 patients death acts as a
**censoring mechanism** — specifically a **death censor** — not as an
observed event.

This explains the upstream label-generation choice cleanly:

- These patients died in their first (and only) admission.
- They will never have a subsequent readmission to observe.
- Their contribution to the recurrent-event likelihood is purely the
  "no event observed by time T" censoring term, with
  `T = DOD - ADMITTIME`.

So the 339 missing `label.csv` rows are not noise or missingness — they
correspond to a structurally well-defined subgroup that the readmission
model treats as right-censored at time of death.

### Implications for downstream use

If you want to use all 35,643 admissions:

| Field | Value for the 339 |
|---|---|
| `censor` | **0** (right-censored — no readmission event observed) |
| `time`   | `DOD - ADMITTIME` (or `DISCHTIME - ADMITTIME` if treating discharge as the censoring time) |

Computing this is straightforward because `DOD` is known for all 339.
The diagnostic CSV (`merged_data/unlabeled_diagnostic.csv`) carries the
fields needed to perform this imputation.

If you only want patients at-risk for readmission, **drop the 339** and
work with the 35,304 labeled rows — that is what `label.csv` already
encodes.

## 7. Reproducing this layout

```bash
python3 mimic_analysis/merge_data.py
```

Reads from `MIMIC Data/processed-data/` and writes
`mimic_analysis/merged_data/{input_named.csv, merged.csv}`. Re-run any
time `processed-data/` changes upstream.
