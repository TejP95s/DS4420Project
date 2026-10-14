import numpy as np
import pandas as pd
from src.data import (VAL_START, VAL_END, TEST_START, TEST_END, val_fit_train, eval_mask, load_hourly)
import os
import matplotlib 
import matplotlib.pyplot as plt

SEASONAL_LAG = 168
HORIZON = 24
P_GRID = [1, 2, 3, 4, 6, 12, 24, 48, 72, 96, 168]
HOUR = pd.Timedelta(hours=1)

def _shift(a, k):
    out = np.full(len(a), np.nan)
    out[k:] = a[:-k]
    return out


def metrics(err):
    err = np.asarray(err, float)
    return {"MAE": float(np.mean(np.abs(err))),
            "RMSE": float(np.sqrt(np.mean(err ** 2))),
            "n": int(len(err))}

"""Basline model that essentially predicts crowdedness will be exactly the same as it was 1 week ago. 
1 week has 168 hours."""
class SeasonalNaive:
    def __init__(self, lag=SEASONAL_LAG):
        self.lag = lag
        self.name = f"Seasonal naive ({lag}h)"

    def fit(self, hourly, fit_start=None, fit_end=None):
        return self

    def predict_one_step(self, hourly):
        return _shift(hourly["y"].to_numpy(float), self.lag)

    def forecast_recursive(self, hourly, origin_pos, H=HORIZON):
        y = hourly["y"].to_numpy(float)
        T = origin_pos[:, None] + np.arange(1, H + 1)[None, :]
        src = T - self.lag
        out = np.full(T.shape, np.nan)
        known = (src >= 0) & (src <= origin_pos[:, None])
        out[known] = y[src[known]]
        return out

"""This is like SeasonalNaive but it averages the past 4 weeks, so it smooths out any spikes in
crowdedness caused by one-off events like holidays."""
class SeasonalMean:
    def __init__(self, n_weeks=4):
        self.n_weeks = n_weeks
        self.name = f"Seasonal mean ({n_weeks}wk)"

    def fit(self, hourly, fit_start=None, fit_end=None):
        return self

    def predict_one_step(self, hourly):
        y = hourly["y"].to_numpy(float)
        stack = np.column_stack([_shift(y, SEASONAL_LAG * k) for k in range(1, self.n_weeks + 1)])
        valid = ~np.isnan(stack)
        cnt = valid.sum(axis=1)
        tot = np.where(valid, stack, 0.0).sum(axis=1)
        return np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)

    def forecast_recursive(self, hourly, origin_pos, H=HORIZON):
        pred = self.predict_one_step(hourly)
        N = len(pred)
        T = origin_pos[:, None] + np.arange(1, H + 1)[None, :]
        out = pred[np.minimum(T, N - 1)]
        out[T >= N] = np.nan
        return out

"""Main AR model which uses P grid for different values of previous history used for forecasting"""
class ARModel:
    def __init__(self, p, seasonal_lags=(), calendar=False, flags=(), name=None):
        self.p = p
        self.lags = sorted(set(range(1, p + 1)) | set(seasonal_lags))
        self.calendar = calendar
        self.flags = tuple(flags)
        self.name = name or f"AR({p})"
    
    def _exog(self, hourly):
        n = len(hourly)
        cols, names = [], []
        if self.calendar:
            hr = hourly["hour"].to_numpy()
            we = hourly["is_weekend"].to_numpy()
            for w, label in ((0, "wkday"), (1, "wkend")):
                for h in range(24):
                    cols.append(((hr == h) & (we == w)).astype(float))
                    names.append(f"{label}_h{h:02d}")
        else:
            cols.append(np.ones(n))
            names.append("const")
        for f in self.flags:
            cols.append(hourly[f].to_numpy(float))
            names.append(f)
        return np.column_stack(cols), names
    
    def _lag_source(self, hourly):
        y = hourly["y"].to_numpy(float)
        key = hourly["is_weekend"].to_numpy(int) * 24 + hourly["hour"].to_numpy(int)
        return np.where(np.isnan(y), self.profile_[key], y)
    
    def _design(self, hourly):
        E, names = self._exog(hourly)
        y = hourly["y"].to_numpy(float)
        ys = self._lag_source(hourly)
        L = np.column_stack([_shift(ys, k) for k in self.lags])
        return np.hstack([E, L]), y, names + [f"lag{k}" for k in self.lags]

    def fit(self, hourly, fit_start=None, fit_end=None):
        idx = hourly.index
        m = np.ones(len(idx), bool)
        if fit_start is not None:
            m &= np.asarray(idx >= pd.Timestamp(fit_start))
        if fit_end is not None:
            m &= np.asarray(idx <= pd.Timestamp(fit_end))
        win = hourly.loc[m & hourly["y"].notna().to_numpy()]
        prof = win.groupby(win["is_weekend"] * 24 + win["hour"])["y"].mean()
        self.profile_ = prof.reindex(range(48)).fillna(win["y"].mean()).to_numpy()
        X, y, self.names_ = self._design(hourly)
        m &= ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        self.coef_, *_ = np.linalg.lstsq(X[m], y[m], rcond=None)
        self.n_fit_ = int(m.sum())
        return self

    def coef_table(self):
        return pd.Series(self.coef_, index=self.names_, name="coef")

    def predict_one_step(self, hourly):
        X, _, _ = self._design(hourly)
        return np.maximum(X @ self.coef_, 0)

    def forecast_recursive(self, hourly, origin_pos, H=HORIZON):
        E, _ = self._exog(hourly)
        y = self._lag_source(hourly)
        N, ne = len(y), E.shape[1]
        w_e, w_l = self.coef_[:ne], self.coef_[ne:]
        preds = np.full((len(origin_pos), H), np.nan)
        for h in range(1, H + 1):
            z = E[np.minimum(origin_pos + h, N - 1)] @ w_e
            for k, w in zip(self.lags, w_l):
                j = h - k
                if j <= 0:
                    src = origin_pos + j
                    v = np.where(src >= 0, y[np.clip(src, 0, N - 1)], np.nan)
                else:
                    v = preds[:, j - 1]
                z = z + w * v
            preds[:, h - 1] = np.maximum(z, 0)
        return preds


def score_one_step(hourly, preds_by_model, start, end):
    idx = hourly.index
    m = eval_mask(hourly).to_numpy() & np.asarray((idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end)))
    for p in preds_by_model.values():
        m &= ~np.isnan(p)
    y = hourly["y"].to_numpy(float)
    return {k: metrics(p[m] - y[m]) for k, p in preds_by_model.items()}

def origins_for(hourly, start, end):
    rng = pd.date_range(pd.Timestamp(start) - HOUR, pd.Timestamp(end) - HOUR, freq="h")
    pos = hourly.index.get_indexer(rng)
    return pos[pos >= 0]

def multistep_targets(hourly, origin_pos, start, end, H=HORIZON):
    N = len(hourly)
    T = origin_pos[:, None] + np.arange(1, H + 1)[None, :]
    Tc = np.minimum(T, N - 1)
    idx = hourly.index
    in_win = np.asarray((idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end)))
    ok = (T < N) & (eval_mask(hourly).to_numpy() & in_win)[Tc]
    return hourly["y"].to_numpy(float)[Tc], ok

def score_multistep(truth, ok, preds_by_model):
    ok = ok.copy()
    for p in preds_by_model.values():
        ok &= ~np.isnan(p)
    overall = {k: metrics((p - truth)[ok]) for k, p in preds_by_model.items()}
    by_h = {k: np.array([np.abs(p - truth)[:, h][ok[:, h]].mean() for h in range(truth.shape[1])])
            for k, p in preds_by_model.items()}
    return overall, pd.DataFrame(by_h, index=np.arange(1, truth.shape[1] + 1))


def tune_p(hourly, make_model, p_grid=P_GRID, criterion="recursive"):
    fit_end = val_fit_train(hourly).index[-1]
    models = {p: make_model(p).fit(hourly, fit_end=fit_end) for p in p_grid}
    if criterion == "one_step":
        preds = {p: m.predict_one_step(hourly) for p, m in models.items()}
        sc = score_one_step(hourly, preds, VAL_START, VAL_END)
    else:
        pos = origins_for(hourly, VAL_START, VAL_END)
        truth, ok = multistep_targets(hourly, pos, VAL_START, VAL_END)
        preds = {p: m.forecast_recursive(hourly, pos) for p, m in models.items()}
        sc, _ = score_multistep(truth, ok, preds)
    if all(v["n"] == 0 for v in sc.values()):
        raise ValueError("No validation rows could be scored; check NaNs in lags/targets.")
    best = min(sc, key=lambda p: sc[p]["MAE"])
    return best, pd.DataFrame(sc).T[["MAE", "RMSE", "n"]]


def plot_sample_week(hourly, models, week_start, path, H=HORIZON):
    matplotlib.use("Agg")
    ws = pd.Timestamp(week_start)
    pos = hourly.index.get_indexer(pd.date_range(ws - HOUR, periods=7, freq="24h"))
    times = pd.date_range(ws, periods=168, freq="h")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(times, hourly["y"].reindex(times), "k", lw=2, label="Actual")
    for name, m in models.items():
        ax.plot(times, m.forecast_recursive(hourly, pos, H).reshape(-1), lw=1.2, label=name)
    ax.set_ylabel("Occupancy (people)")
    ax.set_title(f"Test week of {ws.date()}: 24h-ahead forecasts issued each night")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def run(hourly, out_dir="results", week_start="2016-10-10"):
    os.makedirs(out_dir, exist_ok=True)
    train_end = pd.Timestamp(TEST_START) - HOUR
    flags = ("is_during_semester",)

    SPARSE_SEASONAL = (24, 48, 168, 336)
    SPARSE_GRID = [1, 2, 3, 6, 12]
    makers = {
        "AR(p)": (lambda p: ARModel(p, name=f"AR({p})"), P_GRID),
        "AR(p)+cal+sem": (lambda p: ARModel(p, calendar=True, flags=flags, name=f"AR({p})+cal+sem"), P_GRID),
        "Sparse AR (p short lags + 24,48,168,336) + sem": (lambda p: ARModel(p, seasonal_lags=SPARSE_SEASONAL, flags=flags, name=f"SparseAR({p}+24,48,168,336)+sem"), SPARSE_GRID),
    }

    baselines = [SeasonalNaive(), SeasonalMean(4)]
    models = {b.name: b for b in baselines}
    for label, (mk, grid) in makers.items():
        best_p, table = tune_p(hourly, mk, p_grid=grid)
        print(f"\n[{label}] validation recursive-24h MAE by p:\n{table.round(3)}")
        print(f"chosen p = {best_p}")
        models[mk(best_p).name] = mk(best_p).fit(hourly, fit_end=train_end)

    sparse = [m for m in models.values() if isinstance(m, ARModel) and m.name.startswith("SparseAR")][0]
    print(f"\nSparse AR coefficients ({sparse.name}):\n{sparse.coef_table().round(4)}")

    one = {k: m.predict_one_step(hourly) for k, m in models.items()}
    t1 = pd.DataFrame(score_one_step(hourly, one, TEST_START, TEST_END)).T

    pos = origins_for(hourly, TEST_START, TEST_END)
    truth, ok = multistep_targets(hourly, pos, TEST_START, TEST_END)
    rec = {k: m.forecast_recursive(hourly, pos) for k, m in models.items()}
    t24, by_h = score_multistep(truth, ok, rec)
    t24 = pd.DataFrame(t24).T

    print("\nTEST one-step-ahead (hours 8-23, real readings):\n", t1.round(3))
    print("\nTEST recursive 24h-ahead (hours 8-23, real readings):\n", t24.round(3))

    summary = pd.concat({"one_step": t1, "recursive_24h": t24})
    summary.to_csv(f"{out_dir}/ar_test_errors.csv")
    by_h.to_csv(f"{out_dir}/ar_mae_by_horizon.csv")
    for k, m in models.items():
        if isinstance(m, ARModel): #add files in the reuslts folder
            m.coef_table().to_csv(f"{out_dir}/coef_{k.replace(' ', '_').replace('/', '')}.csv")

    best_ar = min((k for k, m in models.items() if isinstance(m, ARModel)),
                  key=lambda k: t24.loc[k, "MAE"])
    plot_sample_week(hourly, {"Seasonal mean (4wk)": models["Seasonal mean (4wk)"],
                       best_ar: models[best_ar]}, week_start, f"{out_dir}/ar_forecast_week.png")
    return models, summary, by_h


def main():
    run(load_hourly())
if __name__ == "__main__":
    main()