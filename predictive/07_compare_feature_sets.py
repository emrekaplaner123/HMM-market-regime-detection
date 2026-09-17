from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss
)

root = Path(__file__).resolve().parents[1]
data_dir = root / "data" / "predictive"


def load(name):
    return pd.read_csv(
        data_dir / name,
        index_col="Date",
        parse_dates=["Date"]
    )


train = load("train_transition.csv")
validation = load("validation_transition.csv")
test = load("test_transition.csv")

train_filtered = load("train_filtered.csv")
validation_filtered = load("validation_filtered.csv")
test_filtered = load("test_filtered.csv")


def get_episode_ids(filtered):
    medium = filtered["Regime"] == "Medium volatility"

    starts = medium & ~medium.shift(fill_value=False)
    ids = starts.cumsum()

    result = pd.Series(
        pd.NA,
        index=filtered.index,
        dtype="Int64"
    )

    result.loc[medium] = ids.loc[medium]

    return result


def make_episodes(data, filtered):
    data = data.copy()

    ids = get_episode_ids(filtered)

    data["Episode"] = ids.reindex(data.index)
    data = data.dropna(subset=["Episode"])

    data["Episode"] = data["Episode"].astype(int)

    data = data.sort_index()

    return (
        data.groupby("Episode")
        .first()
    )


train_ep = make_episodes(train, train_filtered)
validation_ep = make_episodes(validation, validation_filtered)
test_ep = make_episodes(test, test_filtered)


market_features = [
    "LogReturn",
    "Return_5",
    "Return_20",
    "Volatility_5",
    "Volatility_20",
    "Volatility_change",
    "Drawdown_20"
]

hmm_features = [
    "P_low_volatility",
    "P_high_volatility"
]

combined_features = market_features + hmm_features

feature_sets = {
    "Market only": market_features,
    "HMM only": hmm_features,
    "Combined": combined_features
}


def fit_and_evaluate(name, features):
    scaler = StandardScaler()

    X_train = scaler.fit_transform(train_ep[features])
    X_validation = scaler.transform(validation_ep[features])
    X_test = scaler.transform(test_ep[features])

    y_train = train_ep["Target"].astype(int)
    y_validation = validation_ep["Target"].astype(int)
    y_test = test_ep["Target"].astype(int)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    p_validation = model.predict_proba(X_validation)[:, 1]
    p_test = model.predict_proba(X_test)[:, 1]

    print(f"\n{name}")

    print("Validation")
    print(
        f"ROC-AUC: {roc_auc_score(y_validation, p_validation):.4f}"
    )
    print(
        f"PR-AUC:  {average_precision_score(y_validation, p_validation):.4f}"
    )
    print(
        f"Brier:   {brier_score_loss(y_validation, p_validation):.4f}"
    )

    print("Test")
    print(
        f"ROC-AUC: {roc_auc_score(y_test, p_test):.4f}"
    )
    print(
        f"PR-AUC:  {average_precision_score(y_test, p_test):.4f}"
    )
    print(
        f"Brier:   {brier_score_loss(y_test, p_test):.4f}"
    )


print(f"Training episodes:   {len(train_ep)}")
print(f"Validation episodes: {len(validation_ep)}")
print(f"Test episodes:       {len(test_ep)}")

baseline = train_ep["Target"].mean()

print(f"\nTraining positive rate: {baseline:.4f}")

for name, features in feature_sets.items():
    fit_and_evaluate(name, features)