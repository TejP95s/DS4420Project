import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
from src.data import TRAIN_END, TEST_START, TEST_END, EVAL_HOURS, eval_mask, load_hourly
from src import ar

HOUR = pd.Timedelta(hours=1)
FEATURES = ["temperature", "is_during_semester", "is_start_of_semester", "lvl"]
SEED = 4420
EVAL_H = list(EVAL_HOURS)
LEVEL_WINDOW = 168      
LEVEL_LAG = 24


def add_level(hourly):
    roll = hourly["y"].rolling(LEVEL_WINDOW, min_periods=LEVEL_WINDOW // 2).mean().shift(LEVEL_LAG)
    return hourly.assign(lvl=np.log1p(roll))


class BayesRegression:
    def __init__(self, draws=1000, tune=1000, chains=4, target_accept=0.9, seed=SEED):
        self.draws, self.tune, self.chains = draws, tune, chains
        self.target_accept, self.seed = target_accept, seed
        self.name = "Bayesian regression"

    def _design(self, df):
        hr = df["hour"].to_numpy(int)
        we = df["is_weekend"].to_numpy(float)
        H = np.eye(24)[hr]
        Hw = H * we[:, None]
        t = (df["temperature"].to_numpy(float) - self.t_mean_) / self.t_std_
        l = (df["lvl"].to_numpy(float) - self.l_mean_) / self.l_std_
        X = np.column_stack([t, df["is_during_semester"].to_numpy(float),
                             df["is_start_of_semester"].to_numpy(float), l])
        return H, Hw, X

    def fit(self, hourly, fit_end=TRAIN_END):
        train = hourly.loc[:fit_end]
        train = train[train["hour"].isin(EVAL_H) & train["y"].notna() & train[FEATURES].notna().all(axis=1)]
        self.t_mean_, self.t_std_ = float(train["temperature"].mean()), float(train["temperature"].std())
        self.l_mean_, self.l_std_ = float(train["lvl"].mean()), float(train["lvl"].std())
        H, Hw, X = self._design(train)
        y = np.round(train["y"].to_numpy(float)).astype(int)
        self.n_fit_ = len(y)

        with pm.Model() as self.model:
            icpt = pm.Normal("intercept", mu=np.log(y.mean() + 1), sigma=1)
            sig_h = pm.HalfNormal("sigma_hour", 3)
            sig_w = pm.HalfNormal("sigma_wkend", 1)
            h_raw = sig_h * pm.Normal("hour_z", 0, 1, shape=24)
            hour_eff = pm.Deterministic("hour_eff", h_raw - h_raw[EVAL_H].mean())
            wk_eff = pm.Deterministic("wkend_eff", sig_w * pm.Normal("wkend_z", 0, 1, shape=24))
            beta = pm.Normal("beta", mu=0, sigma=np.array([0.5, 1.0, 1.0, 1.0]), shape=4)
            pm.Deterministic("b_temp", beta[0])
            pm.Deterministic("b_sem", beta[1])
            pm.Deterministic("b_start", beta[2])
            pm.Deterministic("b_lvl", beta[3])
            alpha = pm.HalfNormal("alpha", 10)
            pm.Deterministic("weekend_effect", wk_eff[EVAL_H].mean())

            eta = icpt + pm.math.dot(H, hour_eff) + pm.math.dot(Hw, wk_eff) + pm.math.dot(X, beta)
            pm.NegativeBinomial("obs", mu=pm.math.exp(eta), alpha=alpha, observed=y)

            self.idata = pm.sample(draws=self.draws, tune=self.tune, chains=self.chains,
                                   target_accept=self.target_accept, random_seed=self.seed,
                                   nuts_sampler="nutpie", progressbar=True)

        post = self.idata.posterior
        flat = lambda v: post[v].values.reshape(-1, *post[v].shape[2:])
        self.post_ = {v: flat(v) for v in ["intercept", "hour_eff", "wkend_eff", "beta",
                                           "alpha", "weekend_effect"]}
        for v in ["b_temp", "b_sem", "b_start", "b_lvl"]:
            self.post_[v] = flat(v)
        return self

    def mu_samples(self, df):
        P = self.post_
        H, Hw, X = self._design(df)
        eta = (P["intercept"][:, None] + P["hour_eff"] @ H.T + P["wkend_eff"] @ Hw.T
               + P["beta"] @ X.T)
        return np.exp(eta)

    def predict_with_interval(self, df, level=0.90, seed=SEED):
        mu = self.mu_samples(df)
        rng = np.random.default_rng(seed)
        al = self.post_["alpha"][:, None]
        valid = ~np.isnan(mu[0]) & df["hour"].isin(EVAL_H).to_numpy()
        y_rep = np.full(mu.shape, np.nan)
        y_rep[:, valid] = rng.negative_binomial(al, al / (al + mu[:, valid]))
        a = (1 - level) / 2 * 100
        lo, hi = np.nanpercentile(y_rep, [a, 100 - a], axis=0) if valid.any() else (mu[0], mu[0])
        pred = np.where(valid, mu.mean(axis=0), np.nan)
        return pred, lo, hi

    def summary(self):
        names = ["intercept", "b_temp", "b_sem", "b_start", "b_lvl", "weekend_effect",
                 "sigma_hour", "sigma_wkend", "alpha"]
        try:
            return az.summary(self.idata, var_names=names, ci_prob=0.94)
        except TypeError:
            return az.summary(self.idata, var_names=names, hdi_prob=0.94)


def full_length(hourly, df_part, values):
    out = np.full(len(hourly), np.nan)
    out[hourly.index.get_indexer(df_part.index)] = values
    return out


def compare_with_ar(hourly, bayes_pred, out_dir):
    train_end = pd.Timestamp(TEST_START) - HOUR
    sparse = ar.ARModel(3, seasonal_lags=(24, 48, 168, 336), flags=("is_during_semester",),
                        name="Sparse AR").fit(hourly, fit_end=train_end)
    models = {"Seasonal naive": ar.SeasonalNaive(), "Seasonal mean (4wk)": ar.SeasonalMean(4),
              "Sparse AR": sparse,
              "AR(168)": ar.ARModel(168, name="AR(168)").fit(hourly, fit_end=train_end)}

    one = {k: m.predict_one_step(hourly) for k, m in models.items()}
    one["Bayesian regression"] = bayes_pred
    t1 = pd.DataFrame(ar.score_one_step(hourly, one, TEST_START, TEST_END)).T

    pos = ar.origins_for(hourly, TEST_START, TEST_END)
    truth, ok = ar.multistep_targets(hourly, pos, TEST_START, TEST_END)
    rec = {k: m.forecast_recursive(hourly, pos) for k, m in models.items()}
    N = len(hourly)
    T = pos[:, None] + np.arange(1, ar.HORIZON + 1)[None, :]
    b = bayes_pred[np.minimum(T, N - 1)]
    b[T >= N] = np.nan
    rec["Bayesian regression"] = b
    t24, by_h = ar.score_multistep(truth, ok, rec)
    t24 = pd.DataFrame(t24).T

    summary = pd.concat({"one_step": t1, "recursive_24h": t24})
    summary.to_csv(f"{out_dir}/bayes_vs_ar_test_errors.csv")
    by_h.to_csv(f"{out_dir}/bayes_vs_ar_mae_by_horizon.csv")
    return summary


def plot_effects(model, path):
    matplotlib.use("Agg")
    P = model.post_
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1, 1.4]})

    pct = lambda v: (np.exp(v) - 1) * 100
    eff = {"Weekend vs weekday\n(avg hrs 8-23)": pct(P["weekend_effect"]),
           "Temperature\n(per +1 SD)": pct(P["b_temp"]),
           "Last-week level\n(per +1 SD)": pct(P["b_lvl"]),
           "During semester": pct(P["b_sem"]),
           "Start of semester": pct(P["b_start"])}
    for i, (k, v) in enumerate(eff.items()):
        lo, hi = np.percentile(v, [3, 97])
        a1.plot([lo, hi], [i, i], "k", lw=2)
        a1.plot(v.mean(), i, "o", color="C3")
    a1.axvline(0, color="gray", ls="--", lw=1)
    a1.set_yticks(range(len(eff)))
    a1.set_yticklabels(eff.keys(), fontsize=8)
    a1.invert_yaxis()
    a1.set_xlabel("Effect on occupancy (% change)")
    a1.set_title("Posterior effects (mean, 94% interval)", fontsize=10)

    hrs = np.array(EVAL_H)
    base = np.exp(P["intercept"][:, None] + P["hour_eff"][:, EVAL_H] + P["b_sem"][:, None])
    for label, prof, c in (("Weekday", base, "C0"),("Weekend", base * np.exp(P["wkend_eff"][:, EVAL_H]), "C1")):
        lo, hi = np.percentile(prof, [3, 97], axis=0)
        a2.plot(hrs, prof.mean(axis=0), color=c, label=label)
        a2.fill_between(hrs, lo, hi, color=c, alpha=0.25)
    a2.set_xlabel("Hour of day")
    a2.set_ylabel("Expected occupancy")
    a2.set_title("Hourly profile (in semester, average temp and level)", fontsize=10)
    a2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_test_week(hourly, test, pred, lo, hi, week_start, path, level=0.90):
    matplotlib.use("Agg")
    times = pd.date_range(week_start, periods=168, freq="h")
    loc = test.index.get_indexer(times)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(times, hourly["y"].reindex(times), "k", lw=2, label="Actual")
    ax.plot(times, pred[loc], "C0", lw=1.5, label="Bayesian posterior mean")
    ax.fill_between(times, lo[loc], hi[loc], color="C0", alpha=0.25,
                    label=f"{int(level * 100)}% predictive interval")
    ax.set_ylabel("Occupancy (people)")
    ax.set_title(f"Test week of {pd.Timestamp(week_start).date()}: Bayesian regression")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def run(hourly, out_dir="results", week_start="2016-10-11"):
    os.makedirs(out_dir, exist_ok=True)
    hourly = add_level(hourly)
    model = BayesRegression().fit(hourly, fit_end=TRAIN_END)
    print(f"Fit on {model.n_fit_} hourly rows")

    summ = model.summary()
    summ.to_csv(f"{out_dir}/bayes_posterior_summary.csv")
    print("\nPosterior summary (94% interval; effects on log scale):\n", summ.round(3))
    print(f"Temperature: {100 * (np.exp(model.post_['b_temp'].mean() / model.t_std_) - 1):.2f}% per degree "
          f"(1 SD = {model.t_std_:.2f} degrees)")
    div = int(model.idata.sample_stats["diverging"].sum())
    rcol = [c for c in summ.columns if "r_hat" in c.lower() or "rhat" in c.lower()]
    rmax = f"{summ[rcol[0]].max():.3f}" if rcol else "n/a"
    print(f"max r_hat = {rmax}, divergences = {div}")

    test = hourly.loc[TEST_START:TEST_END]
    pred, lo, hi = model.predict_with_interval(test)
    bayes_pred = full_length(hourly, test, pred)

    m = (eval_mask(test).to_numpy() & ~np.isnan(pred))
    y = test["y"].to_numpy(float)
    cov = float(np.mean((y[m] >= lo[m]) & (y[m] <= hi[m])))
    print(f"\n90% predictive interval coverage on test (hours 8-23): {cov:.3f}")

    summary = compare_with_ar(hourly, bayes_pred, out_dir)
    print("\nTEST one-step / 24h-ahead (identical rows, hours 8-23):\n", summary.round(3))

    plot_effects(model, f"{out_dir}/bayes_effects.png")
    plot_test_week(hourly, test, pred, lo, hi, week_start, f"{out_dir}/bayes_forecast_week.png")

    d = test.assign(pred=pred)[m]
    d["res"] = d["pred"] - d["y"]
    print("\nBias by hour:\n", d.groupby("hour")["res"].mean().round(1).to_string())
    print("\nBias by week:\n", d.groupby(d.index.to_period("W"))["res"].mean().round(1).to_string())
    return model, summary


def main():
    run(load_hourly())


if __name__ == "__main__":
    main()