from pathlib import Path
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

root = Path(__file__).resolve().parents[1]

input_file = root / "data" / "processed" / "sp500_returns.csv"
output_file = root / "data" / "processed" / "sp500_hmm.csv"
summary_file = root / "data" / "processed" / "regime_summary.csv"
transition_file = root / "data" / "processed" / "transition_matrix.csv"

df = pd.read_csv(
    input_file,
    index_col="Date",
    parse_dates=["Date"]
)

X = df[["Return_scaled"]].values

best_model = None
best_score = -np.inf
best_seed = None

for seed in range(20):
    model = GaussianHMM(
        n_components=3,
        covariance_type="full",
        n_iter=1000,
        random_state=seed
    )

    model.fit(X)
    score = model.score(X)

    if score > best_score:
        best_score = score
        best_model = model
        best_seed = seed

df["State"] = best_model.predict(X)

state_order = (
    df.groupby("State")["LogReturn"]
    .std()
    .sort_values()
    .index
    .tolist()
)

labels = {
    state_order[0]: "Low volatility",
    state_order[1]: "Medium volatility",
    state_order[2]: "High volatility"
}

df["Regime"] = df["State"].map(labels)

posterior = best_model.predict_proba(X)

for state, label in labels.items():
    column_name = "P_" + label.lower().replace(" ", "_")
    df[column_name] = posterior[:, state]

summary = df.groupby(["State", "Regime"]).agg(
    Observations=("LogReturn", "size"),
    Mean_daily_return=("LogReturn", "mean"),
    Daily_volatility=("LogReturn", "std")
)

summary["Fraction"] = summary["Observations"] / len(df)
summary["Annualized_mean_return"] = summary["Mean_daily_return"] * 252
summary["Annualized_volatility"] = (
    summary["Daily_volatility"] * np.sqrt(252)
)

for state in state_order:
    summary.loc[
        (state, labels[state]),
        "Expected_duration"
    ] = 1 / (1 - best_model.transmat_[state, state])

transition = best_model.transmat_[np.ix_(state_order, state_order)]

transition_df = pd.DataFrame(
    transition,
    index=[labels[state] for state in state_order],
    columns=[labels[state] for state in state_order]
)

df.to_csv(output_file)
summary.to_csv(summary_file)
transition_df.to_csv(transition_file)

print(f"Best seed: {best_seed}")
print(f"Log-likelihood: {best_score:.2f}")

print("\nRegime summary:")
print(summary.round(4))

print("\nTransition matrix:")
print(transition_df.round(4))