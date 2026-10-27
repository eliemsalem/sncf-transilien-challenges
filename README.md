# SNCF-Transilien forecasting challenges (Challenge Data, ENS)

Two forecasting challenges from SNCF-Transilien, the operator of Paris suburban trains, on
the [Challenge Data](https://challengedata.ens.fr) platform:

- **n°149: station footfall.** Forecast the number of ticket validations per day for 439
  stations over the first half of 2023 (MAPE). The public score went from **141.33 to
  63.16**, which is **1st** on the public leaderboard.
- **n°166: platform waiting time.** Forecast, in real time, the error of the waiting
  time displayed on platform screens for a train two stations upstream (MAE). The public
  score went from **0.6497 to 0.6050**, which is **2nd** on the public leaderboard.

Leaderboard positions as of 27 October 2026.

## Data

The data is **not included**: it belongs to SNCF-Transilien and is only available to
registered participants.

- Challenge 149: https://challengedata.ens.fr/participants/challenges/149/
- Challenge 166: https://challengedata.ens.fr/participants/challenges/166/

| File | Where to put it | Content |
|---|---|---|
| `train.csv`, `y_train.csv`, `test.csv` | `149_validations_gares/data/` | daily validations per station, 2015-2022, and the 2023 test dates |
| `x_train_final.csv`, `y_train_final_j5KGWWK.csv`, `x_test_final.csv`, `y_sample_final.csv` | `166_temps_attente/data/` | train stops with the 6 past measurements, and the target |

External information used for challenge 149: the calendar of 2023 strike days, compiled
by hand from the Transilien "infotrafic" bulletins, Wikipedia and the press. It is
written directly in the script (`STRIKES_2023`, with its sources). Reference paper for
challenge 166: [Coulaud, Keribin, Stoltz, WCRR 2022](https://hal.science/hal-03835496).

## Approach

The same method was used for both challenges:

1. **A validation that mimics the test set.** For 149, the model is trained up to 2021
   and validated on January-June 2022. For 166, the validation covers the last 20 days,
   only at the stops present in the test set and weighted the same way.
2. **One change at a time**, kept only if it improves the validation score averaged over
   several random seeds. A single seed was too noisy: ±0.05 MAPE on 149.
3. **Respect the metric.**
   - With MAPE, over-predicting is unbounded while under-predicting costs at most 100%.
     The 149 model therefore picks, for each row, the value that minimises the
     *expected* MAPE.
   - With an integer target and MAE, the 166 models predict a distribution and keep its
     median.

### Challenge 149: two-stage model

- **Normal level of the station.** XGBoost on log(1+y), expressed as a deviation from the
  station's level in Q4 of the previous year, per weekday. Anchoring on the recent level
  avoids a trend distorted by COVID, and the weekday profile captures remote work:
  Fridays are 6% lower since 2020.
- **Probability of closure.** An XGBoost classifier using each station's closure history.
  On a closure day, a station usually at 8,000 records 5 validations, so predicting its
  normal level costs 160,000%.
- **Atypical days.** Bridge days (known in advance) and 2023 strike days. The strike
  effect is learned per station on past strikes (2016, 2018, 2019-2020) and checked with
  a backtest on the December 2019 strike.

### Challenge 166: ensemble of distribution models

- **Features.** The 6 past delays given by the challenge, plus:
  - the **service pattern** of the train: the stations it serves at its other test stops,
    i.e. timetable information;
  - the **calendar**: school holidays, bridge days, eves of holidays;
  - the **day and train contexts**: averages of the measurements over the station-day
    and over the train's other stops;
  - the **local context**: the measurements of the 1, 3 and 8 trains just before and
    just after at the same station and day, and the time gaps to the nearest ones;
  - the **position of the train within its day**, taken from the order of the rows in the
    files (trains appear in roughly chronological order). It is a proxy for the time of
    day, which the data does not provide.
- **Models.** 21 multiclass XGBoost models and 12 neural networks with embeddings. Their
  predicted distributions over -5..+5 minutes are averaged (60% XGBoost, 40% networks),
  and the median is kept.

## Results

### Challenge 149 (MAPE, lower is better)

| Version | Change | Validation (2022 H1) | Public |
|---|---|---|---|
| v1 | XGBoost on log(1+y), time trend | 1.20 | 141.33 |
| v3 | two stages + "expected MAPE" decision | 0.66 | 79.43 |
| v5 | + bridge days, 2023 strikes (uniform), 3 seeds | 0.607 | 69.22 |
| v6 | + per-station strike effect, anchor per weekday | 0.6035 | 63.84 |
| v7 | stronger strike-day reduction (×0.85, less shrinkage) | 0.6035 | **63.16** |

Validation scores are on the 0-1 scale; the platform reports MAPE × 100.

Did not help: weather data (Open-Meteo), recent closures and neighbouring stations for the
closure classifier, per-station uncertainty, more weight on recent years, an XGBoost +
LightGBM ensemble, removing counting outages, a COVID flag, lag features.

### Challenge 166 (MAE in minutes, lower is better)

| Version | Change | Validation | Public |
|---|---|---|---|
| benchmark | random forest (organisers) | | 0.8930 |
| v1 | XGBoost classifier → median | 0.678 | 0.6497 |
| v2 | + neural network, day context | 0.668 | 0.6407 |
| v3 | + service pattern | 0.661 | 0.6299 |
| v4 | + calendar, train context, ensemble of 33 models | not measured as a whole* | 0.6264 |
| v5 | + position of the train within its day, 60% weight on XGBoost | 0.6559 (XGBoost + 3 networks) | 0.6260 |
| v6 | + local context: trains just before and after at the same station | 0.6355 | **0.6050** |

\* The components of v4 score 0.6599-0.6604 in validation. The full ensemble was not
re-validated before submission.

Did not help: historical medians per station × stop and per service pattern, context of
upstream and neighbouring stations, pseudo-labelling, classes widened to -10, CatBoost,
ExtraTrees, LightGBM.

## What drives the models

Share of the total gain of the XGBoost models (`total_gain`), grouped by family. These
models were trained on the full training set. For high-cardinality categorical features
(439 stations for 149, 84 for 166), XGBoost tends to over-state the importance, so read
the station share as an upper bound.

**Challenge 149, normal-level model**

| Family | Share |
|---|---|
| Calendar: public and school holidays, day of year, weekday, bridge days | 51% |
| Station identity (`station`) | 31% |
| Recent level of the station: Q4 anchors and `lvl_q4` | 18% |

The anchor is also the offset of the target (log(1+y) − anchor), so the recent level
matters more than its 18% suggests. Strike effects are not features: they are applied
after the model, per station.

**Challenge 149, closure classifier**

| Family | Share |
|---|---|
| Calendar: weekday vs weekend, day of year, holidays | 46% |
| Station identity | 34% |
| Closure history of the station (`cl_*`) | 16% |
| Recent level of the station | 4% |

**Challenge 166, XGBoost (main variant of v5)**

| Family | Share |
|---|---|
| Station identity and stop number (`gare`, `arret`) | 40% |
| Day context (`gd_*`, `d_*`) | 18% |
| Combinations of the past delays (means, min/max, differences) | 15% |
| Service pattern (`m12`…`m30`) | 11% |
| Train context (`tr_*`) | 6% |
| The 6 past delays given by the challenge | 5% |
| Position of the train within its day (`pos_day`) | 4% |
| Calendar and weekday | < 1% |

Taken one by one, the most important features after the station are the stop number,
the stations served at stops 18 and 24, the position of the train within its day and the
train context of `p2q0`. The 6 raw past delays have a small share because their
combinations carry the same information in a more usable form.

**Challenge 166, permutation importance across versions.** Increase of the validation MAE
when one family of features is shuffled (one XGBoost per version, then the v5 network and
the v5 blend). Gain-based shares overrate high-cardinality features such as the station;
permutation does not.

| Family | v1 | v2 | v3 | v4 | v5 | v5 network | v5 blend |
|---|---|---|---|---|---|---|---|
| Station | 0.326 | 0.211 | 0.206 | 0.201 | 0.204 | 0.184 | 0.181 |
| Past delays and combinations | 0.177 | 0.120 | 0.091 | 0.087 | 0.086 | 0.118 | 0.094 |
| Day context | | 0.040 | 0.032 | 0.042 | 0.035 | 0.088 | 0.051 |
| Service pattern | | | 0.036 | 0.037 | 0.031 | 0.016 | 0.018 |
| Position in the day | | | | | 0.008 | 0.004 | 0.005 |
| Train context | | | | 0.005 | 0.004 | 0.006 | 0.001 |
| Calendar, weekday | ≈0 | ≈0 | ≈0 | ≈0 | ≈0 | ≈0 | ≈0 |

Each new family takes over part of the weight of the past delays, and the stop number
becomes almost useless (0.040 → 0.003) once the service pattern is known. The networks
rely twice as much on the day context as XGBoost does, which is why blending them helps.
The error breakdown of v5 pointed to the next step: within a station and a day, the
error of a train is correlated with that of the trains just before and after (0.28),
which the day-wide averages missed. This local context (v6) brought the largest gain
since v2.

## Repository layout

```
README.md                              this file
requirements.txt                       Python packages and versions used
149_validations_gares/train_predict.py challenge 149: validation + submission
166_temps_attente/train_predict.py     challenge 166: submission (--validate adds the validation)
docs/149_HANDOFF_NOTE.md               lab notebook for challenge 149 (all versions, bugs, dead ends)
docs/166_NOTES.md                      lab notebook for challenge 166
```

## How to reproduce

```bash
pip install -r requirements.txt
# put the challenge files in each data/ folder (see the Data section), then:
python 149_validations_gares/train_predict.py   # ~10 min on CPU, writes 149_validations_gares/submission.csv
python 166_temps_attente/train_predict.py       # ~35 min on a GPU, writes 166_temps_attente/submission.csv
```

## Limitations

- **Grey areas.** Two features use information that was not available at forecast
  time. The platform rules allow enriching the data, but the spirit of the task is
  debatable:
  - the 2023 strike calendar (149) was only known afterwards;
  - the day, train and local contexts (166) include trains that ran later the same day;
  - the position of the train within its day (166) relies on how the organisers sorted
    the files, not on a provided variable.
- **Strike effects are not validated locally.** The 2022 validation period has no
  strikes, so their effect could only be checked on the leaderboard and with a backtest
  on the December 2019 strike.
- **Scores are public-leaderboard scores.** The private leaderboard is computed on another
  part of the test set and may rank differently.
- **166 is not bit-for-bit reproducible.** The neural networks run on a GPU, so a rerun
  can differ from the submitted file on a few rows.
- **166: the script produces v6** (0.6050). The gain of v5 over v4 was small (0.6264 →
  0.6260, whereas the validation predicted about 0.622): the position feature helped
  less on the test period than on the validation period.
- **149: the script reproduces v7** (63.16) exactly. v6 (63.84) differs only on strike
  days: `STRIKE_SHRINK = 5` and `STRIKE_EXTRA = 1.0`.
