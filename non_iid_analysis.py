"""Computes Augmented Dickey-Fuller (stationarity) and Kolmogorov-Smirnov
(distributional heterogeneity) statistics on the `consumption` series of every
Pecan Street client, to support the Non-IID characterization of the dataset
used in the FL evaluation.

Reads the already-processed per-client train/test CSVs (produced by
ParticipantData.preprocess_readings), not the raw per-client exports.
"""
import glob
import itertools
import os

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from statsmodels.tsa.stattools import adfuller

REGIONS = ["austin", "california", "newyork", "puertorico"]
DATA_ROOT = "dataset/pecanstreet/15min"
OUT_DIR = os.path.join(DATA_ROOT, "non_iid_stats")


def client_ids(region: str):
    train_files = glob.glob(os.path.join(DATA_ROOT, region, "train", "*.csv"))
    ids = sorted(
        os.path.splitext(os.path.basename(f))[0]
        for f in train_files
        if os.path.basename(f) != "full_dataset.csv"
    )
    return ids


def load_client_series(region: str, cid: str) -> pd.Series:
    frames = []
    for split in ("train", "test"):
        path = os.path.join(DATA_ROOT, region, split, f"{cid}.csv")
        if os.path.exists(path):
            frames.append(pd.read_csv(path, usecols=["Date", "consumption"]))
    df = pd.concat(frames, ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"], utc=True)
    df = df.sort_values("Date").drop_duplicates("Date").set_index("Date")
    series = df["consumption"].resample("15min").mean()
    series = series.interpolate(method="time", limit_direction="both")
    return series.dropna()


def run_adf(series: pd.Series) -> dict:
    stat, pvalue, usedlag, nobs, crit, _ = adfuller(series.values, autolag="AIC")
    return {"statistic": stat, "pvalue": pvalue, "usedlag": usedlag, "nobs": nobs, **{f"crit_{k}": v for k, v in crit.items()}}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    adf_rows = []
    ks_intra_rows = []
    region_pool = {}

    for region in REGIONS:
        ids = client_ids(region)
        print(f"[{region}] {len(ids)} clients")
        series_by_client = {}
        for cid in ids:
            series_by_client[cid] = load_client_series(region, cid)

        for cid, series in series_by_client.items():
            res = run_adf(series)
            res.update({"region": region, "cid": cid, "n_obs": len(series)})
            adf_rows.append(res)

        for cid_a, cid_b in itertools.combinations(ids, 2):
            stat, pvalue = ks_2samp(series_by_client[cid_a].values, series_by_client[cid_b].values)
            ks_intra_rows.append({"region": region, "cid_a": cid_a, "cid_b": cid_b, "statistic": stat, "pvalue": pvalue})

        region_pool[region] = np.concatenate([s.values for s in series_by_client.values()])

    adf_df = pd.DataFrame(adf_rows)
    ks_intra_df = pd.DataFrame(ks_intra_rows)

    adf_summary = adf_df.groupby("region").apply(
        lambda g: pd.Series({
            "n_clients": len(g),
            "pct_stationary": 100 * (g["pvalue"] < 0.05).mean(),
            "mean_statistic": g["statistic"].mean(),
            "median_statistic": g["statistic"].median(),
            "mean_pvalue": g["pvalue"].mean(),
            "median_pvalue": g["pvalue"].median(),
        }),
        include_groups=False,
    ).reset_index()

    ks_intra_summary = ks_intra_df.groupby("region").apply(
        lambda g: pd.Series({
            "n_pairs": len(g),
            "pct_significant": 100 * (g["pvalue"] < 0.05).mean(),
            "mean_statistic": g["statistic"].mean(),
            "median_statistic": g["statistic"].median(),
        }),
        include_groups=False,
    ).reset_index()

    ks_inter_rows = []
    for region_a, region_b in itertools.combinations(REGIONS, 2):
        stat, pvalue = ks_2samp(region_pool[region_a], region_pool[region_b])
        ks_inter_rows.append({"region_a": region_a, "region_b": region_b, "statistic": stat, "pvalue": pvalue})
    ks_inter_df = pd.DataFrame(ks_inter_rows)

    adf_df.to_csv(os.path.join(OUT_DIR, "adf_raw.csv"), index=False)
    ks_intra_df.to_csv(os.path.join(OUT_DIR, "ks_intra_raw.csv"), index=False)
    adf_summary.to_csv(os.path.join(OUT_DIR, "adf_summary.csv"), index=False)
    ks_intra_summary.to_csv(os.path.join(OUT_DIR, "ks_intra_summary.csv"), index=False)
    ks_inter_df.to_csv(os.path.join(OUT_DIR, "ks_inter_summary.csv"), index=False)

    print("\nADF summary:\n", adf_summary)
    print("\nKS intra-region summary:\n", ks_intra_summary)
    print("\nKS inter-region summary:\n", ks_inter_df)


if __name__ == "__main__":
    main()
