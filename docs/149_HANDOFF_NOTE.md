# Handoff note — challenge n°149 (validations per station)

Lab notebook of the project: what was tried, in what order, and why.
Latest update: 2026-10-27.

## Goal

Predict the **number of validations per day and per station** (`y`) for 439 stations,
from 1 January to 30 June 2023. **Metric: MAPE.** The challenge benchmark copies 2022,
shifted so that weekdays line up.

## Data (`149_validations_gares/data/`, not included)

- `train.csv`: `date, station, job, ferie, vacances` (2015-2022, 1.24 M rows)
- `y_train.csv`: `index` (= `date_station`), `y` (the target)
- `test.csv`: same columns + `index`, for 2023 (181 days × 439 stations = 78,652 rows)
- Expected submission format: **`index,y`** (78,652 rows), e.g. `2023-01-01_1J7,10000`.

## Run

```bash
python 149_validations_gares/train_predict.py
# -> prints the 2022 validation MAPE (and the benchmark's) and writes submission.csv
```

## Results (validation on January-June 2022, the same season as the test)

| Version | Change | Validation MAPE | Public score |
|---|---|---|---|
| v1 | XGBoost (squared error) on log(1+y), depth 8 | 1.20 | 141.33 |
| v2 | XGBoost (MAE/y) on log(1+y), depth 4, ×0.8 | 0.79 (0.80-0.87 depending on the seed) | not submitted |
| v3 | two stages (level + P(closure)), prediction minimising the expected MAPE | 0.66 | 79.43 |
| v4 | level anchored on Q4 of the previous year, trained on non-COVID years | 0.62 (one lucky seed; 0.635 over 3 seeds) | not submitted |
| v5 | + bridge days, 2023 strikes (uniform multiplier), 3 seeds | 0.607 | 69.22 |
| v6 | + per-station strike effect, higher uncertainty on strike days, 30 strike days, anchor per weekday | 0.6035 | 63.84 (1st) |
| v7 | per-station strike effect shrunk less (2 days), extra ×0.85 on strike days | 0.6035 (identical outside strike days) | **63.16 (1st)** |

- The trend-based level model of v3 under-estimated 2022 by 27% (COVID in the trend).
  With the anchor, the bias drops to 3% and the level MAPE from 0.31 to 0.23.
- Strike effects cannot be measured on the validation set (no strike in H1 2022). They
  were checked with a **backtest on the December 2019 strike** (training on 2016 + 2018
  only), MAPE on the strike days: no correction 8.31, uniform reduction (v5) 7.46,
  per-station effect + extra caution (v6) 6.60.
- With a single seed, the validation MAPE varies by ±0.05: only comparisons averaged
  over several seeds are reliable.

**Key to the metric.** About 1% of the rows (closure days: y≈5 at a station that usually
has several thousand) account for about 70% of the MAPE. Over-predicting is unbounded
(8,000 predicted for 5 actual = 160,000%), while under-predicting costs at most 100%
(predicting 0 everywhere gives a MAPE of exactly 1.0). The model therefore handles
closures explicitly. For each row, it chooses the prediction that minimises the expected
MAPE. As soon as a large station's risk of closure exceeds about 1-2%, predicting 0 (a
fixed cost of 100%) is cheaper on average than predicting its normal level.
Closures: about 1.4% of days, 8× more frequent at weekends. A station closed in H2 is 3
to 5× more likely to be closed in the following H1.

**Tried without gain** (2022 H1 validation): removing the COVID period, a COVID flag,
last year's level as a lag feature, deeper or more trees; for the closure classifier:
closures in the last weeks, neighbouring stations (co-closures), fewer features;
per-station uncertainty; weather (Open-Meteo); more weight on recent years;
XGBoost + LightGBM ensemble; removing counting outages.

## Bugs found in the original code (fixed here)

1. **Submission in the wrong format and incomplete**: `station,date,prediction` with only
   6 dates (2,612 rows), whereas the platform expects `index,y` with 78,652 rows.
2. **No trend modelling** (traffic grows ~6% a year): a one-hot year cannot extrapolate
   to 2023. A continuous time index was added, then replaced by the Q4 anchor.
3. **Evaluation on the training data**, which gave a misleading score. It was replaced
   by a time-based validation.
4. **`y = 0` imputed with means**, which distorted the target. The zeros are kept, and
   the MAPE ignores them.
5. **XGBoost trained directly on `y`**, so only the large stations mattered. It now
   trains on log(1+y).
6. **About 20 untitled notebooks** with duplicated logic, consolidated into one script.
