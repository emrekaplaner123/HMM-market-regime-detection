from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix
)

root = Path(__file__).resolve().parents[1]
data_dir = root / "data" / "predictive"

features = [
    "LogReturn",
    "Return_5",
    "Return_20",
    "Volatility_5",
    "Volatility_20",
    "Volatility_change",
    "Drawdown_20",
    "Days_in_medium",
    "P_low_volatility",
    "P_high_volatility"
]

train = pd.read_csv(
    data_dir / "train_transition.csv",
    index_col="Date",
    parse_dates=["Date"]
)

validation = pd.read_csv(
    data_dir / "validation_transition.csv",
    index_col="Date",
    parse_dates=["Date"]
)

test = pd.read_csv(
    data_dir / "test_transition.csv",
    index_col="Date",
    parse_dates=["Date"]
)

train_filtered = pd.read_csv(
    data_dir / "train_filtered.csv",
    index_col="Date",
    parse_dates=["Date"]
)

validation_filtered = pd.read_csv(
    data_dir / "validation_filtered.csv",
    index_col="Date",
    parse_dates=["Date"]
)

test_filtered = pd.read_csv(
    data_dir / "test_filtered.csv",
    index_col="Date",
    parse_dates=["Date"]
)

scaler = StandardScaler()

X_train = scaler.fit_transform(train[features])
y_train = train["Target"].astype(int)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


def episode_ids(data):
    medium = data["Regime"] == "Medium volatility"

    starts = medium & ~medium.shift(
        fill_value=False
    )

    ids = starts.cumsum()

    result = pd.Series(
        pd.NA,
        index=data.index,
        dtype="Int64"
    )

    result.loc[medium] = ids.loc[medium]

    return result


def make_episode_data(transition_data, filtered_data):
    data = transition_data.copy()

    probabilities = model.predict_proba(
        scaler.transform(data[features])
    )[:, 1]

    data["Probability"] = probabilities

    ids = episode_ids(filtered_data)

    data["Episode"] = ids.reindex(data.index)

    data = data.dropna(subset=["Episode"])
    data["Episode"] = data["Episode"].astype(int)

    data = data.sort_index().reset_index()

    episodes = (
        data.groupby("Episode", as_index=False)
        .first()
    )

    return episodes


train_episodes = make_episode_data(
    train,
    train_filtered
)

validation_episodes = make_episode_data(
    validation,
    validation_filtered
)

test_episodes = make_episode_data(
    test,
    test_filtered
)

baseline_probability = train_episodes["Target"].mean()


def evaluate(name, data):
    y = data["Target"].astype(int).values
    p = data["Probability"].values

    prediction = (p >= 0.5).astype(int)

    baseline = np.full(
        len(y),
        baseline_probability
    )

    print(f"\n{name}")
    print(f"Episodes: {len(data)}")
    print(
        "Targets:",
        dict(pd.Series(y).value_counts().sort_index())
    )

    if len(np.unique(y)) == 2:
        print(f"ROC-AUC:     {roc_auc_score(y, p):.4f}")
        print(
            f"PR-AUC:      "
            f"{average_precision_score(y, p):.4f}"
        )

    print(
        f"Brier score:  "
        f"{brier_score_loss(y, p):.4f}"
    )

    print(
        f"Baseline Brier: "
        f"{brier_score_loss(y, baseline):.4f}"
    )

    print(
        f"Accuracy:     "
        f"{accuracy_score(y, prediction):.4f}"
    )

    print("\nConfusion matrix:")
    print(confusion_matrix(y, prediction))


print(
    f"Training episode baseline probability: "
    f"{baseline_probability:.4f}"
)

evaluate(
    "Validation - first day of medium episode",
    validation_episodes
)

evaluate(
    "Test - first day of medium episode",
    test_episodes
)

validation_episodes.to_csv(
    data_dir / "validation_episodes.csv",
    index=False
)

test_episodes.to_csv(
    data_dir / "test_episodes.csv",
    index=False
)