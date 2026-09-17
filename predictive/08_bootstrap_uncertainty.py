from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss

root = Path(__file__).resolve().parents[1]
data_dir = root / "data" / "predictive"


def load(name):
    return pd.read_csv(
        data_dir / name,
        index_col="Date",
        parse_dates=["Date"]
    )


train = load("train_transition.csv")
test = load("test_transition.csv")

train_filtered = load("train_filtered.csv")
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

    return data.groupby("Episode").first()


train_ep = make_episodes(train, train_filtered)
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


def fit_model(features):
    scaler = StandardScaler()

    X_train = scaler.fit_transform(train_ep[features])
    y_train = train_ep["Target"].astype(int)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    X_test = scaler.transform(test_ep[features])
    probability = model.predict_proba(X_test)[:, 1]

    return probability


p_market = fit_model(market_features)
p_hmm = fit_model(hmm_features)
p_combined = fit_model(combined_features)

y = test_ep["Target"].astype(int).to_numpy()

print("Original test results:")
print(
    f"Market AUC:   {roc_auc_score(y, p_market):.4f}"
)
print(
    f"HMM AUC:      {roc_auc_score(y, p_hmm):.4f}"
)
print(
    f"Combined AUC: {roc_auc_score(y, p_combined):.4f}"
)

rng = np.random.default_rng(42)

n_bootstrap = 5000

auc_market = []
auc_hmm = []
auc_combined = []

brier_market = []
brier_hmm = []
brier_combined = []

n = len(y)

for _ in range(n_bootstrap):
    idx = rng.integers(0, n, size=n)

    y_boot = y[idx]

    if len(np.unique(y_boot)) < 2:
        continue

    auc_market.append(
        roc_auc_score(y_boot, p_market[idx])
    )

    auc_hmm.append(
        roc_auc_score(y_boot, p_hmm[idx])
    )

    auc_combined.append(
        roc_auc_score(y_boot, p_combined[idx])
    )

    brier_market.append(
        brier_score_loss(y_boot, p_market[idx])
    )

    brier_hmm.append(
        brier_score_loss(y_boot, p_hmm[idx])
    )

    brier_combined.append(
        brier_score_loss(y_boot, p_combined[idx])
    )

auc_market = np.array(auc_market)
auc_hmm = np.array(auc_hmm)
auc_combined = np.array(auc_combined)

brier_market = np.array(brier_market)
brier_hmm = np.array(brier_hmm)
brier_combined = np.array(brier_combined)

auc_difference = auc_combined - auc_market
brier_difference = brier_market - brier_combined


def interval(values):
    return np.percentile(values, [2.5, 50, 97.5])


print("\n95% bootstrap intervals:")

print(
    "Market AUC:",
    np.round(interval(auc_market), 4)
)

print(
    "HMM AUC:",
    np.round(interval(auc_hmm), 4)
)

print(
    "Combined AUC:",
    np.round(interval(auc_combined), 4)
)

print(
    "\nCombined - Market AUC difference:",
    np.round(interval(auc_difference), 4)
)

print(
    "Probability Combined AUC > Market AUC:",
    round(np.mean(auc_difference > 0), 4)
)

print(
    "\nMarket Brier:",
    np.round(interval(brier_market), 4)
)

print(
    "HMM Brier:",
    np.round(interval(brier_hmm), 4)
)

print(
    "Combined Brier:",
    np.round(interval(brier_combined), 4)
)

print(
    "\nMarket - Combined Brier difference:",
    np.round(interval(brier_difference), 4)
)

print(
    "Probability Combined Brier < Market Brier:",
    round(np.mean(brier_difference > 0), 4)
)