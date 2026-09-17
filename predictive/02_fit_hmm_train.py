from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

root = Path(__file__).resolve().parents[1]

train_file = root / "data" / "predictive" / "train.csv"
model_file = root / "data" / "predictive" / "hmm_model.pkl"
state_file = root / "data" / "predictive" / "hmm_state_order.csv"

train = pd.read_csv(
    train_file,
    index_col="Date",
    parse_dates=["Date"]
)

X = train[["Return_scaled"]].values

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

states = best_model.predict(X)
train["State"] = states

state_order = (
    train.groupby("State")["LogReturn"]
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

state_df = pd.DataFrame({
    "State": state_order,
    "Regime": [labels[s] for s in state_order]
})

with open(model_file, "wb") as f:
    pickle.dump(best_model, f)

state_df.to_csv(state_file, index=False)

print(f"Best seed: {best_seed}")
print(f"Training log-likelihood: {best_score:.2f}")

print("\nState order:")
print(state_df)

print("\nTransition matrix:")
print(np.round(best_model.transmat_, 4))