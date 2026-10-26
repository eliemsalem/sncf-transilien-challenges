# Notes — challenge n°166 (platform waiting time)

Lab notebook of the project. Latest update: 2026-10-26.

## Goal

Predict `p0q0`, the difference (whole minutes) between the theoretical and the actual
waiting time, for a train located two stations upstream. **Metric: MAE.** The official
benchmark (a random forest) scores 0.893 on the public leaderboard.

## Data (`166_temps_attente/data/`, not included)

- `x_train_final.csv`: 667,264 rows, April → November 2023; `x_test_final.csv`:
  20,657 rows, 13 Nov. → 22 Dec. 2023.
- `train` (identifier unique per day), `gare` (station), `date`, `arret` (stop number).
- `p2q0`, `p3q0`, `p4q0`: differences of the previous trains at the same station.
- `p0q2`, `p0q3`, `p0q4`: differences of the same train at upstream stations.
- Submission format: `,p0q0` (test index + prediction), like `y_sample_final.csv`.

## Run

```bash
python 166_temps_attente/train_predict.py              # submission, ≈ 35 min on a GPU
python 166_temps_attente/train_predict.py --validate   # + validation
```

## Key points

- **The test set only contains stops 12, 18, 24, 30 and 36** (one in six), whereas the
  training set covers stops 7 to 42. The validation is restricted to those stops and
  weighted like the test set. A first version mistakenly kept rare stops (13, 17, 19…)
  and gave lower MAEs (0.631 instead of 0.678).
- **No features from neighbouring stops**: in the test set, a train only has 2 or 3
  non-consecutive stops, and later stops would leak future information.
- **Integer target, MAE**: we predict the median of the distribution predicted by a
  classifier (-5..+5 minutes), which minimises the expected absolute error.
- The test period (Nov.-Dec.) is a bit easier than the validation period: public score
  ≈ 0.958 × validation MAE.
- **Trains go both ways and come from many origins**: 817 origin-destination pairs, and
  373 of 709 station-to-station links are also travelled in the other direction. A
  station appears with about 21 different stop numbers (median).

## Results

Validation: last 20 days of the training set (Oct.-Nov.), stops 12/18/24/30/36 weighted
like the test set.

| Model | Validation MAE | Public score |
|---|---|---|
| Predict 0 | 0.940 | |
| XGBoost classifier → median (v1) | 0.678 | 0.6497 (35th) |
| + neural network + day context (v2) | 0.668 | 0.6407 (8th) |
| + service pattern (v3) | 0.661 | 0.6299 (3rd) |
| + calendar, train context, large ensemble of 33 models (v4) | not measured as a whole (components: 0.6599-0.6604) | **0.6264 (2nd)** |

**Service pattern**: for each row, the stations served by the same train at the other
test stops (12, 18, 24, 30). This is timetable information, in the spirit of the
"timetable patterns" of the reference paper (Coulaud, Keribin, Stoltz, WCRR 2022,
line H). It is computed at test density.

**Day context (grey area)**: averages of the measurements over the station-day and over
the day, computed on the test rows (and, during training, at the same density). They
include trains that ran later in the day, which is not available in real time. The
**train context** (averages over the train's other stops) is in the same grey area.

**Remaining error**: rows at ≤ -5 min (incidents, 1.2%) account for about 20% of the MAE,
with an average error of about 11 minutes. Widening the classes down to -10 did not help:
the predicted risk of an incident rarely exceeds 50%.

**Tried without gain**: historical medians per station × stop and per service pattern,
training on test stops only, the train's stop k-6, context of neighbouring and upstream
stations, pseudo-labelling, classes down to -10, CatBoost, ExtraTrees, LightGBM (weaker
than XGBoost), a threshold other than 0.5, non-integer predictions.
