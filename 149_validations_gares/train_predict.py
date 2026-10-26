#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SNCF-Transilien challenge n°149 — forecasting the number of validations per day and station.

Metric: MAPE (Mean Absolute Percentage Error).
Official benchmark: a copy of 2022 shifted so that weekdays line up.

Two-stage approach:
  1) a model of the station's NORMAL LEVEL (trained on open days), expressed as a
     deviation from its level in Q4 of the previous year (the "anchor");
  2) a model of the PROBABILITY OF CLOSURE (a day with y≈5 at a station that usually has 8,000).
For each row, we then predict the value that minimises the expected MAPE.
Closures (~1.4% of days) dominate the MAPE: over-predicting is unbounded
(8,000 predicted for 5 actual = 160,000%), while under-predicting costs at most 100%.

v5: + BRIDGE DAYS (feature, known in advance), + STRIKE DAYS (2023 calendar cross-checked
against the press and Transilien bulletins), + average over 3 seeds (the validation MAPE
varied by ±0.05 from one seed to another).
v6: strike effect SPECIFIC TO EACH STATION (learned on 2016, 2018, 2019-2020), higher
uncertainty on strike days, completed 2023 calendar, anchor per weekday (remote work).
v7: per-station strike effect shrunk less (2 days instead of 5), extra 15% reduction on
strike days.

Usage:
    python train_predict.py           # time-based validation + submission
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
from xgboost import XGBClassifier, XGBRegressor

HERE = Path(__file__).parent
DATA = HERE / "data"

# Uncertainty (standard deviation of log y) of an open day around the predicted level.
SIGMA = 0.2
# The classifier under-estimates future closures (0.9% predicted vs 1.5% actual
# on 2022 H1): we multiply it by 1.5. SIGMA and P_MUL were chosen on 2022 H1.
P_MUL = 1.5
SEEDS = (42, 7, 123)

# 2023 strikes (test period): 3 = very strong, 2 = strong, 1 = disrupted / moderate.
# Sources: transilien.com "infotrafic" bulletins, Wikipedia (2023 pension reform protests),
# sortiraparis (20 April). Intensity = published forecasts.
# v6: weekends of 11-12 and 25-26 March ("similar disruptions" announced), 24 March raised
# (RER B/C/E 1 in 2, D/R 2 in 5), 31 March (industrial action, line L),
# 16-22 March: rolling strike with no bulletin found -> low intensity, to be safe.
STRIKES_2023 = {"2023-01-19": 3, "2023-01-31": 3, "2023-02-07": 2, "2023-02-08": 1, "2023-02-11": 1,
                "2023-02-16": 1, "2023-03-07": 3, "2023-03-08": 2, "2023-03-09": 2, "2023-03-10": 2,
                "2023-03-11": 2, "2023-03-12": 2, "2023-03-13": 2, "2023-03-14": 2, "2023-03-15": 2,
                "2023-03-16": 1, "2023-03-17": 1, "2023-03-20": 1, "2023-03-21": 1, "2023-03-22": 1,
                "2023-03-23": 3, "2023-03-24": 2, "2023-03-25": 1, "2023-03-26": 1, "2023-03-28": 2,
                "2023-03-31": 1, "2023-04-06": 1, "2023-04-13": 1, "2023-04-20": 1, "2023-06-06": 1}
# Strike periods in the history (2016 labour law, 2018 rolling strike, 2019-2020 pensions,
# 2022): the intensity of each day is measured on network traffic.
STRIKE_WINDOWS = [("2016-03-09", "2016-06-30"), ("2018-03-22", "2018-06-28"),
                  ("2019-12-05", "2020-01-20"), ("2022-10-18", "2022-10-18"), ("2022-11-10", "2022-11-10")]
# Strike effect PER STATION and per intensity: median of log(y / usual level) over past
# strikes, shrunk towards the network-wide effect (weight of 2 days).
STRIKE_SHRINK = 2
# v7: extra reduction on strike days. Going from v5 to v6 (strike-day predictions -17%)
# gained 5.4 points on the leaderboard, and the backtest on the December 2019 strike also
# shows that a stronger reduction pays off with the MAPE.
STRIKE_EXTRA = 0.85


def load_train():
    """Loads the features and the target, aligned on `index` = date_station."""
    X = pd.read_csv(DATA / "train.csv")
    X["index"] = X["date"] + "_" + X["station"]
    y = pd.read_csv(DATA / "y_train.csv")          # columns: index, y
    return X.merge(y, on="index")


def add_features(df, station_categories=None):
    """Calendar features + a continuous TREND (traffic grows by about 6% a year)."""
    d = pd.to_datetime(df["date"])
    df = df.copy()
    df["year"] = d.dt.year
    df["month"] = d.dt.month
    df["weekday"] = d.dt.day_of_week
    df["quarter"] = d.dt.quarter
    df["week_of_year"] = d.dt.isocalendar().week.astype(int)
    df["day_of_year"] = d.dt.day_of_year
    df["is_weekend"] = (d.dt.day_of_week >= 5).astype(int)
    # continuous time index since the first day: captures long-term growth
    df["time_idx"] = (d - pd.Timestamp("2015-01-01")).dt.days
    # station as a category, with a FIXED set of categories (train = reference)
    if station_categories is None:
        df["station"] = df["station"].astype("category")
    else:
        df["station"] = pd.Categorical(df["station"], categories=station_categories)
    return df


def calendar(train, test):
    """Per date: `bridge` (Monday before / Friday after a public holiday) and `strike` (0-3).
    Past strikes are graded by the day's network traffic, relative to the median of the
    same weekdays nearby (±4 weeks)."""
    cal = pd.concat([train[["date", "ferie", "job"]], test[["date", "ferie", "job"]]]).groupby("date").max()
    cal.index = pd.to_datetime(cal.index)
    cal = cal.sort_index()
    wd = cal.index.dayofweek
    prev_f = cal.ferie.shift(1, fill_value=0).values
    next_f = cal.ferie.shift(-1, fill_value=0).values
    cal["bridge"] = ((((wd == 4) & (prev_f == 1)) | ((wd == 0) & (next_f == 1))) & (cal.ferie == 0)).astype(int)

    tot = train.assign(date=pd.to_datetime(train.date)).groupby("date").agg(y=("y", "sum"), wd=("weekday", "first"))
    ref = tot.groupby("wd").y.transform(lambda s: s.rolling(9, center=True, min_periods=4).median())
    ratio = (tot.y / ref).reindex(cal.index)
    strike = pd.Series(0, index=cal.index)
    for a, b in STRIKE_WINDOWS:
        m = (cal.index >= a) & (cal.index <= b) & (cal.ferie == 0).values
        strike[m] = np.select([ratio[m] < 0.55, ratio[m] < 0.75, ratio[m] < 0.9], [3, 2, 1], 0)
    for day, level in STRIKES_2023.items():
        strike[pd.Timestamp(day)] = level
    cal["strike"] = strike
    cal.index = cal.index.strftime("%Y-%m-%d")
    return cal[["bridge", "strike"]]


def strike_effects(df):
    """Effect of past strikes, per station and per intensity, and spread per intensity.
    Usual level = rolling median of the same weekdays (±4 weeks)."""
    d = df.sort_values(["station", "date"])
    usual = d.groupby(["station", "weekday"], observed=True)["y"].transform(
        lambda s: s.rolling(9, center=True, min_periods=4).median())
    d = d.assign(lr=np.log((d["y"] + 1) / (usual + 1)))
    d = d[(d.strike > 0) & (d.closed == 0) & (usual >= 50)]
    level = d.groupby("strike").lr.median()
    st = d.groupby(["station", "strike"], observed=True).lr.agg(["median", "size"]).reset_index()
    st["effect"] = ((st["size"] * st["median"] + STRIKE_SHRINK * st.strike.map(level))
                    / (st["size"] + STRIKE_SHRINK))
    resid = d.lr - d.set_index(["station", "strike"]).index.map(
        st.set_index(["station", "strike"])["effect"]).values
    # robust spread (median absolute deviation): the standard deviation is inflated by a few
    # near-closed stations, which would make the decision overly cautious
    spread = resid.abs().groupby(d.strike).median() * 1.4826
    st["station"] = st["station"].astype(str)
    return st.set_index(["station", "strike"])["effect"], level, spread


def flag_closures(df):
    """Closed day = y <= 5% of the station's rolling median (±28 days)."""
    df = df.sort_values(["station", "date"]).reset_index(drop=True)
    df["norm"] = df.groupby("station", observed=True)["y"].transform(
        lambda s: s.rolling(57, center=True, min_periods=10).median())
    df["closed"] = ((df["y"] <= 0.05 * df["norm"]) & (df["norm"] >= 100)).astype(int)
    return df


def add_station_history(target, hist, year):
    """Closure history and level of each station, as seen on 1 January of `year`
    (computed only on years < year: no leakage)."""
    h = hist[hist.year < year]
    s2 = h[(h.year == year - 1) & (h.month >= 7)]
    g = lambda d: d.groupby("station", observed=True)
    f = pd.DataFrame({
        "cl_hist": g(h).closed.mean(),
        "cl_hist_we": g(h[h.is_weekend == 1]).closed.mean(),
        "cl_hist_wd": g(h[h.is_weekend == 0]).closed.mean(),
        "cl_s2": g(s2).closed.mean(),
        "cl_s2_we": g(s2[s2.is_weekend == 1]).closed.mean(),
        "cl_q4": g(s2[s2.month >= 10]).closed.mean(),
        "lvl_q4": np.log1p(g(s2[s2.month >= 10]).norm.median()),
    })
    cats = target["station"].cat.categories
    out = target.merge(f, left_on="station", right_index=True, how="left")
    out["station"] = pd.Categorical(out["station"].astype(str), categories=cats)
    return out


def add_anchor(target, hist):
    """Anchor = log1p of the station's median level in Q4 of the previous year (open days,
    excluding public and school holidays), separately for weekdays / weekends.
    Anchoring on the recent level avoids extrapolating a trend: the one learned on
    2015-2021 is pulled down by COVID (-27% bias on 2022 H1)."""
    q4 = hist[(hist.month >= 10) & (hist.closed == 0) & (hist.ferie == 0)
              & (hist.vacances == 0) & (hist.y > 0)]
    anc = np.log1p(q4.groupby(["station", "year", "is_weekend"], observed=True).y.median())
    anc = anc.rename("anchor").reset_index()
    anc["year"] += 1
    anc["station"] = anc["station"].astype(str)
    key = target[["station", "year", "is_weekend"]].assign(station=target["station"].astype(str))
    a = key.merge(anc, on=["station", "year", "is_weekend"], how="left")["anchor"].values
    target = target.copy()
    target["anchor"] = a
    # fallback 1: last known level of the station (last 90 days of history);
    # fallback 2 (station without history): a LOW level, the safe choice for the MAPE
    for yr in target.loc[target["anchor"].isna(), "year"].unique():
        h = hist[(hist.year < yr) & (hist.closed == 0) & (hist.y > 0)]
        last = h.groupby("station", observed=True)["date"].transform("max")
        recent = h[pd.to_datetime(h["date"]) >= pd.to_datetime(last) - pd.Timedelta(days=90)]
        lvl = np.log1p(recent.groupby(recent["station"].astype(str)).y.median())
        rows = target["anchor"].isna() & (target["year"] == yr)
        fill = target.loc[rows, "station"].astype(str).map(lvl)
        low = np.nanquantile(target.loc[target["year"] == yr, "anchor"], 0.1)
        target.loc[rows, "anchor"] = fill.fillna(low).values
    # v6: anchor per station × WEEKDAY (remote-work profile: since COVID, Fridays are ~6%
    # below Tuesday-Thursday); the weekday/weekend anchor stays as a feature (`anchor_we`)
    # and is used as a fallback
    q4 = q4[(q4.strike == 0) & (q4.bridge == 0)]
    awd = np.log1p(q4.groupby(["station", "year", "weekday"], observed=True).y.median())
    awd = awd.rename("a_wd").reset_index()
    awd["year"] += 1
    awd["station"] = awd["station"].astype(str)
    key = target[["station", "year", "weekday"]].assign(station=target["station"].astype(str))
    v = key.merge(awd, on=["station", "year", "weekday"], how="left")["a_wd"].values
    target["anchor_we"] = target["anchor"]
    target["anchor"] = np.where(np.isnan(v), target["anchor_we"].values, v)
    return target


LEVEL_FEATURES = ["station", "job", "ferie", "vacances", "month", "weekday",
                  "week_of_year", "day_of_year", "is_weekend", "anchor", "lvl_q4", "bridge", "anchor_we"]
# the level model only learns from normal years (2015 has no anchor; 2020-2021 COVID)
LEVEL_EXCLUDED_YEARS = (2015, 2020, 2021)
CLF_FEATURES = ["station", "job", "ferie", "vacances", "month", "weekday",
                "quarter", "week_of_year", "day_of_year", "is_weekend",
                "cl_hist", "cl_hist_we", "cl_hist_wd", "cl_s2", "cl_s2_we", "cl_q4", "lvl_q4", "bridge"]


def mape_nonzero(y_true, y_pred):
    """MAPE ignoring y=0 (the MAPE is undefined when the true value is 0)."""
    mask = y_true > 0
    return mean_absolute_percentage_error(y_true[mask], y_pred[mask])


def xgb(cls, seed, **kw):
    return cls(n_estimators=700, learning_rate=0.05, subsample=0.8,
               colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0,
               tree_method="hist", enable_categorical=True, n_jobs=-1,
               random_state=seed, **kw)


def fit(df):
    """Trains both stages (3 seeds each); also returns the typical values of a closed day.
    Strike days are excluded from training: their effect is applied separately."""
    normal = df[df.strike == 0]
    open_days = normal[(normal.closed == 0) & (normal.y > 0) & ~normal.year.isin(LEVEL_EXCLUDED_YEARS)]
    # level: log(1+y) - anchor (the MAPE is a RELATIVE error)
    regs = [xgb(XGBRegressor, s, max_depth=8).fit(open_days[LEVEL_FEATURES],
                                                  np.log1p(open_days["y"]) - open_days["anchor"]) for s in SEEDS]
    clfs = [xgb(XGBClassifier, s, max_depth=4).fit(normal[CLF_FEATURES], normal["closed"]) for s in SEEDS]
    y_closed = np.sort(df[(df.closed == 1) & (df.y > 0)]["y"].values)
    return regs, clfs, y_closed, strike_effects(df)


def predict(models, df):
    """For each row: prediction = mu * k, with k minimising the expected MAPE under the
    mixture  closed (probability p, y ~ closed-day values)  /  open (y = mu * e^eps)."""
    regs, clfs, y_closed, (effect, level_effect, spread) = models
    mu = np.expm1(np.mean([r.predict(df[LEVEL_FEATURES]) for r in regs], axis=0)
                  + df["anchor"].values).clip(min=1)
    # strike days: station-specific effect (fallback: average effect of the level) and
    # higher uncertainty -> the decision rule predicts all the more cautiously
    g = df["strike"].values
    key = pd.MultiIndex.from_arrays([df["station"].astype(str).values, g])
    eff = pd.Series(key.map(effect), dtype=float).fillna(pd.Series(g).map(level_effect)).fillna(0).values
    mu = mu * np.where(g > 0, np.exp(eff) * STRIKE_EXTRA, 1)
    sigma = np.where(g > 0, np.sqrt(SIGMA ** 2 + pd.Series(g).map(spread).fillna(0).values ** 2), SIGMA)
    p = np.mean([c.predict_proba(df[CLF_FEATURES])[:, 1] for c in clfs], axis=0)
    p = np.clip(P_MUL * p, 0, 1)[:, None]

    k = np.arange(0, 1.31, 0.02)
    z = np.random.default_rng(0).normal(0, 1, 4000)
    c_grid = np.r_[0, np.geomspace(1, 2e5, 400)]
    cost_closed = np.array([np.mean(np.abs(y_closed - c) / y_closed) for c in c_grid])

    cand = mu[:, None] * k[None, :]
    cost_open = np.zeros_like(cand)
    for s in np.unique(sigma):
        e = np.exp(s * z)
        cost_open[sigma == s] = np.array([np.mean(np.abs(e - ki) / e) for ki in k])
    cost = p * np.interp(cand, c_grid, cost_closed) + (1 - p) * cost_open
    return cand[np.arange(len(mu)), cost.argmin(axis=1)]


def main():
    test = pd.read_csv(DATA / "test.csv")                 # columns: index,date,station,...
    df = add_features(load_train())
    cal = calendar(df, add_features(test))
    df = flag_closures(df.join(cal, on="date"))
    station_cats = df["station"].cat.categories           # reference for the test set
    df = add_anchor(pd.concat([add_station_history(df[df.year == yr], df, yr)
                               for yr in sorted(df.year.unique())]), df)

    # --- 1) TIME-BASED VALIDATION: train on 2015-2021, validate on January-June 2022
    # (mimics the real task: forecasting the first half of an unseen future year)
    tr, va = df[df.year <= 2021], df[(df.year == 2022) & (df.month <= 6)]
    pred_va = predict(fit(tr), va)
    print(f"[Validation 2022 H1] MAPE = {mape_nonzero(va['y'].values, pred_va):.4f}")

    # Naive reference benchmark: copy the previous year (same date, one year earlier).
    prev = df[df.year == 2021][["station", "month", "day_of_year", "y"]]
    prev = prev.rename(columns={"y": "y_prev"})
    bench = va.merge(prev, on=["station", "month", "day_of_year"], how="left")
    bench["y_prev"] = bench["y_prev"].fillna(bench["y_prev"].median())
    print(f"[Validation 2022 H1] MAPE of the benchmark (copy of 2021) = "
          f"{mape_nonzero(bench['y'].values, bench['y_prev'].values):.4f}")

    # --- 2) RETRAINING on ALL years, then prediction of the 2023 test set.
    models = fit(df)

    test_feat = add_anchor(add_station_history(
        add_features(test, station_categories=station_cats).join(cal, on="date"), df, 2023), df)
    pred_test = pd.Series(predict(models, test_feat), index=test_feat["index"])

    # EXACT format expected by the platform: index,y (78,652 rows, test order)
    sub = pd.DataFrame({"index": test["index"],
                        "y": np.round(pred_test.loc[test["index"]].values).astype(int)})
    out = HERE / "submission.csv"
    sub.to_csv(out, index=False)
    print(f"Submission written: {out}  ({len(sub)} rows)")


if __name__ == "__main__":
    main()
