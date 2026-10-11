import pandas as pd
import numpy as np

TRAIN_END   = "2016-08-28 23:00"
VAL_START   = "2016-01-18 00:00"
VAL_END     = "2016-04-10 23:00"
TEST_START  = "2016-08-29 00:00"
TEST_END    = "2016-12-11 23:00"
EVAL_HOURS  = range(8, 24)
FLAGS = ["is_during_semester", "is_start_of_semester"]


def split(hourly):
    """Chrono split and val overlaps train by design."""
    train = hourly.loc[:TRAIN_END]
    val   = hourly.loc[VAL_START:VAL_END]
    test  = hourly.loc[TEST_START:TEST_END]
    return train, val, test


def val_fit_train(hourly):
    """Data strictly before the validation window"""
    return hourly.loc[:pd.Timestamp(VAL_START) - pd.Timedelta(hours=1)]


def eval_mask(df):
    return df["hour"].isin(EVAL_HOURS) & df["number_people"].notna()


def mask_outages(df, min_hours=3, day_start=10, day_end=20, break_min=30):
    """Set long daytime zero-runs to NaN (prob. sensor outages)."""
    df = df.copy()
    is_zero = df["number_people"].eq(0)
    gap = df.index.to_series().diff() > pd.Timedelta(minutes=break_min)
    run_id = ((is_zero != is_zero.shift()) | gap).cumsum()
    for _, run in df[is_zero].groupby(run_id[is_zero]):
        hours = (run.index[-1] - run.index[0]).total_seconds() / 3600
        in_day = ((run.index.hour >= day_start) & (run.index.hour < day_end)).any()
        if hours > min_hours and in_day:
            df.loc[run.index, "number_people"] = np.nan
    return df


def interp_short_gaps(s, max_gap):
    """Linearly interpolate NaN gaps of <= max_gap steps but leave longer gaps NaN."""
    filled = s.interpolate(limit_area="inside")
    is_na = s.isna()
    run_id = (is_na != is_na.shift()).cumsum()
    run_len = is_na.groupby(run_id).transform("sum")
    return filled.where(~(is_na & (run_len > max_gap)), np.nan)


def load_hourly(path="data/gymData.csv", max_interp_hours=2):
    df = pd.read_csv(path)
    df["dt"] = pd.to_datetime(df["date"].str[:19])
    df = df.sort_values("dt").set_index("dt")

    df = mask_outages(df)
    hourly = df.resample("h").agg({
        "number_people": "mean",
        "temperature": "mean",
        "is_during_semester": "max",
        "is_start_of_semester": "max",
    })

    hourly[FLAGS] = hourly[FLAGS].ffill()
    hourly["temperature"] = interp_short_gaps(hourly["temperature"], 6)
    hourly["y"] = interp_short_gaps(hourly["number_people"], max_interp_hours)

    hourly["hour"] = hourly.index.hour
    hourly["dow"] = hourly.index.dayofweek
    hourly["is_weekend"] = (hourly["dow"] >= 5).astype(int)
    return hourly