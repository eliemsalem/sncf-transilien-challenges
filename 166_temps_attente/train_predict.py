#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SNCF-Transilien challenge n°166 — real-time forecasting of platform waiting time.

Target p0q0: difference (whole minutes) between the theoretical and the actual waiting
time, for a train located two stations upstream. Metric: MAE. Public score: 0.6264 (2nd).

Approach: each model predicts the DISTRIBUTION of p0q0 over -5..+5 minutes; the
distributions are averaged and we predict their MEDIAN (the value that minimises the
absolute error). Final ensemble, half and half:
  - multiclass XGBoost: 6 hyperparameter variants (3 seeds each) + 1 variant with the
    train context (3 seeds);
  - neural network (embeddings for station / stop / weekday / station×stop / service
    pattern): 2 × 3 seeds, + 3 seeds with the day context, + 3 seeds with the day and
    train contexts.
Recent months weigh more (60-day half-life): the test period (Nov.-Dec.) comes later.

Features, beyond the 6 past measurements p{i}q{j}:
  - service pattern: stations served by the train at its other test stops
    (timetable information, cf. Coulaud et al., WCRR 2022);
  - calendar: school holidays (zone C), public holidays, bridge days, eves of holidays;
  - day context (grey area, see docs/166_NOTES.md): averages of the measurements over
    the station-day and over the day; train context: averages over the train's other stops.
Contexts are computed "at test density": on the training set, only from the rows at the
test stops (12, 18, 24, 30, 36), subsampled to 80%.

Validation (option --validate): last 20 days of the training set, test stops only,
weighted by the distribution of stops in the test set.

Usage:
    python train_predict.py              # submission (≈ 35 min on a GPU)
    python train_predict.py --validate    # + validation (≈ twice as long)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from xgboost import XGBClassifier

HERE = Path(__file__).parent
DATA = HERE / "data"

P = ["p2q0", "p3q0", "p4q0", "p0q2", "p0q3", "p0q4"]
FEATURES = (["gare", "arret", "wd"] + P
            + ["q_mean", "p_mean", "q_d23", "q_d34", "p_d23", "q_min", "q_max", "p_min", "p_max",
               "q_abs", "p_abs", "n_pos", "n_neg", "q_med", "p_med", "qp"])
CTX = [f"gd_{c}" for c in P] + ["gd_n", "gd_absq", "d_p0q2", "d_p2q0", "d_absq"]
TRAIN_CTX = ["tr_p0q2", "tr_p0q3", "tr_p2q0", "tr_absq", "tr_n"]
TEST_ARRETS = {12, 18, 24, 30, 36}
MISSION_ARRETS = (12, 18, 24, 30)
MISSION = [f"m{a}" for a in MISSION_ARRETS]
CALENDAR = ["school_holiday", "public_holiday", "bridge_day", "holiday_eve"]
LO, HI = -5, 5
CLASSES = np.arange(LO, HI + 1)
HALF_LIFE = 60                     # days
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 2023 calendar, zone C (Île-de-France)
SCHOOL_HOLIDAYS = [("2023-04-22", "2023-05-09"), ("2023-07-08", "2023-09-04"),
            ("2023-10-21", "2023-11-06"), ("2023-12-23", "2024-01-08")]
PUBLIC_HOLIDAYS = {"2023-04-10", "2023-05-01", "2023-05-08", "2023-05-18", "2023-05-29", "2023-07-14",
          "2023-08-15", "2023-11-01", "2023-11-11", "2023-12-25"}
BRIDGE_DAYS = {"2023-05-19", "2023-08-14"}

# XGBoost variants of the ensemble (3 seeds each); the train-context variant is added in ensemble_proba
XGB_VARIANTS = [(dict(), (0, 1, 2)), (dict(max_depth=11), (0, 1, 2)), (dict(min_child_weight=60), (0, 1, 2)),
                (dict(colsample_bytree=0.6), (0, 1, 2)), (dict(learning_rate=0.05, n_estimators=700), (0, 1, 2)),
                (dict(), (3, 4, 5))]


def load():
    X = pd.read_csv(DATA / "x_train_final.csv", index_col=0).drop(columns=["Unnamed: 0"])
    X["y"] = pd.read_csv(DATA / "y_train_final_j5KGWWK.csv", index_col=0)["p0q0"].values
    T = pd.read_csv(DATA / "x_test_final.csv", index_col=0)
    return X, T


def add_features(df, gares, ga_cats):
    """p{i}q0: previous trains at the same station; p0q{j}: same train, upstream stations."""
    df = df.copy()
    df["wd"] = pd.to_datetime(df["date"]).dt.day_of_week
    df["gare"] = pd.Categorical(df["gare"], categories=gares)
    q, p = df[["p0q2", "p0q3", "p0q4"]], df[["p2q0", "p3q0", "p4q0"]]
    df["q_mean"], df["p_mean"] = q.mean(axis=1), p.mean(axis=1)
    df["q_d23"], df["q_d34"], df["p_d23"] = df.p0q2 - df.p0q3, df.p0q3 - df.p0q4, df.p2q0 - df.p3q0
    df["q_min"], df["q_max"], df["p_min"], df["p_max"] = q.min(axis=1), q.max(axis=1), p.min(axis=1), p.max(axis=1)
    df["q_abs"], df["p_abs"] = q.abs().sum(axis=1), p.abs().sum(axis=1)
    df["n_pos"], df["n_neg"] = (df[P] > 0).sum(axis=1), (df[P] < 0).sum(axis=1)
    df["q_med"], df["p_med"] = q.median(axis=1), p.median(axis=1)
    df["qp"] = df.p0q2 * df.p2q0
    ga = df["gare"].astype(str) + "_" + df["arret"].astype(str)
    df["gare_arret"] = pd.Categorical(ga, categories=ga_cats)
    return df


def add_calendar(df):
    d = pd.to_datetime(df["date"])
    df = df.copy()
    df["school_holiday"] = 0
    df["holiday_eve"] = 0
    for a, b in SCHOOL_HOLIDAYS:
        df.loc[(d >= a) & (d <= b), "school_holiday"] = 1
        df.loc[(d >= pd.Timestamp(a) - pd.Timedelta(days=3)) & (d < a), "holiday_eve"] = 1
    df["public_holiday"] = df.date.isin(PUBLIC_HOLIDAYS).astype(int)
    df["bridge_day"] = df.date.isin(BRIDGE_DAYS).astype(int)
    return df


def context_rows(df, is_test, seed=0):
    """Rows "like the test set": on the training set, only the rows at the test stops,
    subsampled to 80% (test density); on the test set, all of its rows."""
    if is_test:
        return df
    src = df[df.arret.isin(TEST_ARRETS)]
    return src[np.random.default_rng(seed).random(len(src)) < 0.8]


def add_context(df, is_test, seed=0):
    """Day context (grey area: includes trains that ran later in the day)."""
    src = context_rows(df, is_test, seed)
    src = src.assign(absq=src.p0q2.abs())
    gd = src.groupby(["gare", "date"], observed=True).agg(
        **{f"gd_{c}": (c, "mean") for c in P}, gd_n=("p0q2", "size"), gd_absq=("absq", "mean"))
    d = src.groupby("date").agg(d_p0q2=("p0q2", "mean"), d_p2q0=("p2q0", "mean"), d_absq=("absq", "mean"))
    return df.join(gd, on=["gare", "date"]).join(d, on="date")


def add_mission(df, is_test, seed=0):
    """Service pattern of the train: stations it serves at the other test stops (12, 18,
    24, 30). Timetable information, known in advance (cf. the "timetable patterns" of
    the reference paper, Coulaud et al., WCRR 2022)."""
    src = context_rows(df, is_test, seed)
    piv = src.assign(gare=src.gare.astype(str)).pivot_table(
        index=["date", "train"], columns="arret", values="gare", aggfunc="first")
    key = pd.MultiIndex.from_frame(df[["date", "train"]])
    df = df.copy()
    for a in MISSION_ARRETS:
        served = piv[a].to_dict() if a in piv.columns else {}
        df[f"m{a}"] = pd.Categorical(key.map(served.get), categories=df.gare.cat.categories)
        df.loc[df.arret == a, f"m{a}"] = np.nan                  # not its own station
    return df


def add_train_context(df, is_test, seed=0):
    """Train context: averages of the measurements over its OTHER stops (current row excluded)."""
    src = context_rows(df, is_test, seed)
    src = src.assign(absq=src.p0q2.abs())
    agg = src.groupby(["date", "train"])[["p0q2", "p0q3", "p2q0", "absq"]].agg(["sum", "count"])
    key = pd.MultiIndex.from_frame(df[["date", "train"]])
    in_src = df.index.isin(src.index)
    df = df.copy()
    own = {"p0q2": df.p0q2.values, "p0q3": df.p0q3.values, "p2q0": df.p2q0.values, "absq": df.p0q2.abs().values}
    for c in ["p0q2", "p0q3", "p2q0", "absq"]:
        tot = key.map(agg[(c, "sum")].to_dict().get).astype(float)
        n = key.map(agg[(c, "count")].to_dict().get).astype(float)
        tot, n = np.where(in_src, tot - own[c], tot), np.where(in_src, n - 1, n)
        with np.errstate(invalid="ignore", divide="ignore"):
            df[f"tr_{c}"] = np.where(n > 0, tot / n, np.nan)
    df["tr_n"] = np.nan_to_num(n)
    return df


def enrich(df, is_test):
    return add_calendar(add_train_context(add_mission(add_context(df, is_test), is_test), is_test))


def recency_weight(dates, ref):
    age = (pd.Timestamp(ref) - pd.to_datetime(dates)).dt.days.values
    return 0.5 ** (age / HALF_LIFE)


# ---------------------------------------------------------------- XGBoost
def xgb_proba(tr, te, feats, ref, seeds, **kw):
    out = np.zeros((len(te), len(CLASSES)))
    for s in seeds:
        params = dict(n_estimators=400, learning_rate=0.08, max_depth=9, subsample=0.8, colsample_bytree=0.8,
                      min_child_weight=20, tree_method="hist", device=DEVICE, enable_categorical=True,
                      max_cat_to_onehot=1, objective="multi:softprob", random_state=s)
        params.update(kw)
        m = XGBClassifier(**params)
        m.fit(tr[feats], tr.y.clip(LO, HI).astype(int) - LO, sample_weight=recency_weight(tr.date, ref))
        out += m.predict_proba(te[feats]) / len(seeds)
    return out


# ---------------------------------------------------------------- neural network
class Net(nn.Module):
    def __init__(self, card, n_num, h=2048, drop=0.2):
        super().__init__()
        # station, stop, weekday, station×stop, then one served station per pattern stop
        dims = [16, 8, 3, 24] + [8] * (len(card) - 4)
        self.emb = nn.ModuleList([nn.Embedding(k + 1, e) for k, e in zip(card, dims)])
        block = lambda i, o: [nn.Linear(i, o), nn.BatchNorm1d(o), nn.SiLU(), nn.Dropout(drop)]
        self.mlp = nn.Sequential(*block(n_num + sum(dims), h), *block(h, h), *block(h, h // 2),
                                 nn.Linear(h // 2, len(CLASSES)))

    def forward(self, x, c):
        return self.mlp(torch.cat([x] + [m(c[:, i]) for i, m in enumerate(self.emb)], 1))


def nn_inputs(d, num, stats=None):
    x = d[num].values.astype(np.float32)
    if stats is None:
        med = np.nanmedian(x, axis=0)
        x = np.where(np.isnan(x), med, x)
        stats = (x.mean(0), x.std(0) + 1e-6, med)
    else:
        x = np.where(np.isnan(x), stats[2], x)                  # missing values: training median
    x = np.clip((x - stats[0]) / stats[1], -6, 6)
    cols = [d.gare.cat.codes, d.arret.clip(0, 45), d.wd, d.gare_arret.cat.codes]
    cols += [d[m].cat.codes + 1 for m in MISSION]                # 0 = station missing
    c = np.stack(cols, 1).clip(min=0)
    return torch.tensor(x), torch.tensor(c.astype(np.int64)), stats


def nn_proba(tr, te, num, ref, seeds, epochs=12, lr=2e-3, bs=4096):
    xt, ct, stats = nn_inputs(tr, num)
    xv, cv, _ = nn_inputs(te, num, stats)
    yt = torch.tensor(tr.y.clip(LO, HI).astype(int).values - LO)
    wt = torch.tensor(recency_weight(tr.date, ref), dtype=torch.float32)
    card = [len(tr.gare.cat.categories), 46, 7, len(tr.gare_arret.cat.categories)]
    card += [len(tr.gare.cat.categories) + 1] * len(MISSION)
    X_, C_, Y_, W_ = xt.to(DEVICE), ct.to(DEVICE), yt.to(DEVICE), wt.to(DEVICE)
    out = np.zeros((len(te), len(CLASSES)))
    for s in seeds:
        torch.manual_seed(s)
        net = Net(card, len(num)).to(DEVICE)
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
        n_steps = len(xt) // bs
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * n_steps)
        lossf = nn.CrossEntropyLoss(reduction="none")
        net.train()
        for _ in range(epochs):
            perm = torch.randperm(len(xt), device=DEVICE)
            for b in range(n_steps):
                i = perm[b * bs:(b + 1) * bs]
                loss = (lossf(net(X_[i], C_[i]), Y_[i]) * W_[i]).sum() / W_[i].sum()
                opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        net.eval()
        with torch.no_grad():
            pr = torch.cat([torch.softmax(net(xv[j:j + 65536].to(DEVICE), cv[j:j + 65536].to(DEVICE)), 1).cpu()
                            for j in range(0, len(xv), 65536)]).numpy()
        out += pr / len(seeds)
    return out


# ---------------------------------------------------------------- ensemble
def ensemble_proba(tr, te, ref):
    feats = FEATURES + CTX + MISSION + CALENDAR
    xgb = [xgb_proba(tr, te, feats, ref, seeds, **kw) for kw, seeds in XGB_VARIANTS]
    xgb.append(xgb_proba(tr, te, feats + TRAIN_CTX, ref, (0, 1, 2)))
    print(f"  XGBoost: {len(xgb)} variants", flush=True)
    num = [f for f in FEATURES if f not in ("gare", "arret", "wd")] + CALENDAR
    nets = [nn_proba(tr, te, num, ref, (0, 1, 2)), nn_proba(tr, te, num, ref, (3, 4, 5)),
            nn_proba(tr, te, num + CTX, ref, (0, 1, 2)), nn_proba(tr, te, num + CTX + TRAIN_CTX, ref, (0, 1, 2))]
    print(f"  networks: {len(nets)} variants", flush=True)
    return 0.5 * np.mean(xgb, axis=0) + 0.5 * np.mean(nets, axis=0)


def median(proba):
    """Median of the predicted distribution = minimiser of the expected absolute error."""
    return CLASSES[(proba.cumsum(axis=1) >= 0.5).argmax(axis=1)]


def main():
    X, T = load()
    gares = sorted(set(X.gare) | set(T.gare))
    ga_cats = sorted(set(X.gare + "_" + X.arret.astype(str)) | set(T.gare + "_" + T.arret.astype(str)))
    X, T = add_features(X, gares, ga_cats), add_features(T, gares, ga_cats)
    X, T = enrich(X, is_test=False), enrich(T, is_test=True)

    if "--validate" in sys.argv:
        # last 20 days of the training set, test stops, weighted as in the test set
        cut = np.sort(X.date.unique())[-20]
        share = T.arret.value_counts(normalize=True)
        tr, va = X[X.date < cut], X[(X.date >= cut) & X.arret.isin(TEST_ARRETS)]
        w = va.arret.map(share / va.arret.value_counts(normalize=True)).values
        print(f"[Validation] predict 0: MAE = {np.average(np.abs(va.y), weights=w):.4f}", flush=True)
        err = np.abs(va.y - median(ensemble_proba(tr, va, cut)))
        print(f"[Validation] ensemble: MAE = {np.average(err, weights=w):.4f}", flush=True)

    # Retraining on the whole training set, prediction of the test set (13 Nov. -> 22 Dec. 2023)
    sub = pd.DataFrame({"p0q0": median(ensemble_proba(X, T, T.date.min()))}, index=T.index)
    out = HERE / "submission.csv"
    sub.to_csv(out)                                        # same format as y_sample: ,p0q0
    print(f"Submission written: {out}  ({len(sub)} rows)", flush=True)


if __name__ == "__main__":
    main()
