from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "sp500_regimes.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "regime_summary.csv"

df = pd.read_csv(INPUT_PATH, index_col=0, parse_dates=True)

volatility_by_state = (
    df.groupby("State")["Volatility20"]
    .mean()
    .sort_values()
)

ordered_states = volatility_by_state.index.tolist()

labels = {
    ordered_states[0]: "Low volatility",
    ordered_states[1]: "Normal",
    ordered_states[2]: "High volatility"
}

df["Regime"] = df["State"].map(labels)

summary = df.groupby(["State", "Regime"]).agg(
    Observations=("LogReturn", "size"),
    Mean_daily_return=("LogReturn", "mean"),
    Daily_return_std=("LogReturn", "std"),
    Mean_annualized_volatility=("Volatility20", "mean")
)

summary["Fraction"] = summary["Observations"] / len(df)
summary["Annualized_mean_return"] = summary["Mean_daily_return"] * 252

run_id = (df["State"] != df["State"].shift()).cumsum()
run_id.name = "Run"

runs = (
    df.groupby(run_id)
    .agg(
        State=("State", "first"),
        Regime=("Regime", "first"),
        Start=("State", lambda x: x.index.min()),
        End=("State", lambda x: x.index.max()),
        Duration=("State", "size")
    )
)

duration_summary = runs.groupby(["State", "Regime"])["Duration"].agg(
    Average_duration="mean",
    Median_duration="median",
    Maximum_duration="max"
)

summary = summary.join(duration_summary)

print(summary.round(4))

print("\nLongest regime periods:")
print(
    runs.sort_values("Duration", ascending=False)
    .head(15)
    .to_string(index=False)
)

summary.to_csv(OUTPUT_PATH)
df.to_csv(INPUT_PATH)

print(f"\nSaved summary to: {OUTPUT_PATH}")