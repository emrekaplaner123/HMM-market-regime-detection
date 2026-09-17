from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss

root = Path(__file__).resolve().parents[1]
data_dir = root / "data" / "predictive"
figure_dir = root / "figures" / "predictive"

figure_dir.mkdir(parents=True, exist_ok=True)


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


def episode_ids(filtered):
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

    ids = episode_ids(filtered)
    data["Episode"] = ids.reindex(data.index)

    data = data.dropna(subset=["Episode"])
    data["Episode"] = data["Episode"].astype(int)

    data = data.sort_index().reset_index()

    return data.groupby("Episode", as_index=False).first()


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


def fit_model(features):
    scaler = StandardScaler()

    X_train = scaler.fit_transform(train_ep[features])
    X_validation = scaler.transform(validation_ep[features])
    X_test = scaler.transform(test_ep[features])

    y_train = train_ep["Target"].astype(int)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    p_validation = model.predict_proba(X_validation)[:, 1]
    p_test = model.predict_proba(X_test)[:, 1]

    return p_validation, p_test


p_val_market, p_test_market = fit_model(market_features)
p_val_hmm, p_test_hmm = fit_model(hmm_features)
p_val_combined, p_test_combined = fit_model(combined_features)

y_validation = validation_ep["Target"].astype(int).to_numpy()
y_test = test_ep["Target"].astype(int).to_numpy()


models = ["Market only", "HMM only", "Combined"]

validation_auc = [
    roc_auc_score(y_validation, p_val_market),
    roc_auc_score(y_validation, p_val_hmm),
    roc_auc_score(y_validation, p_val_combined)
]

test_auc = [
    roc_auc_score(y_test, p_test_market),
    roc_auc_score(y_test, p_test_hmm),
    roc_auc_score(y_test, p_test_combined)
]


x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))

ax.bar(
    x - width / 2,
    validation_auc,
    width,
    label="Validation"
)

ax.bar(
    x + width / 2,
    test_auc,
    width,
    label="Test"
)

ax.set_xticks(x)
ax.set_xticklabels(models)

ax.set_ylim(0.5, 1.0)
ax.set_ylabel("ROC-AUC")
ax.set_title("Predictive Performance by Feature Set")
ax.legend()

for i, value in enumerate(validation_auc):
    ax.text(
        i - width / 2,
        value + 0.01,
        f"{value:.3f}",
        ha="center"
    )

for i, value in enumerate(test_auc):
    ax.text(
        i + width / 2,
        value + 0.01,
        f"{value:.3f}",
        ha="center"
    )

plt.tight_layout()
plt.savefig(
    figure_dir / "feature_set_auc.png",
    dpi=300
)
plt.close()


fig, ax = plt.subplots(figsize=(12, 5))

negative = y_test == 0
positive = y_test == 1

ax.scatter(
    test_ep.loc[negative, "Date"],
    p_test_combined[negative],
    label="Next regime: Low",
    marker="o"
)

ax.scatter(
    test_ep.loc[positive, "Date"],
    p_test_combined[positive],
    label="Next regime: High",
    marker="x"
)

ax.axhline(
    0.5,
    linestyle="--",
    linewidth=1,
    label="0.5 threshold"
)

ax.set_ylim(0, 1)
ax.set_xlabel("Start of medium-volatility episode")
ax.set_ylabel("Predicted probability of high-volatility exit")
ax.set_title("Predictions for Test-Period Medium-Volatility Episodes")
ax.legend()

plt.tight_layout()
plt.savefig(
    figure_dir / "test_episode_predictions.png",
    dpi=300
)
plt.close()


rng = np.random.default_rng(42)

auc_difference = []
brier_difference = []

n = len(y_test)

for _ in range(5000):
    idx = rng.integers(0, n, size=n)

    y_boot = y_test[idx]

    if len(np.unique(y_boot)) < 2:
        continue

    market_auc = roc_auc_score(
        y_boot,
        p_test_market[idx]
    )

    combined_auc = roc_auc_score(
        y_boot,
        p_test_combined[idx]
    )

    auc_difference.append(
        combined_auc - market_auc
    )

    market_brier = brier_score_loss(
        y_boot,
        p_test_market[idx]
    )

    combined_brier = brier_score_loss(
        y_boot,
        p_test_combined[idx]
    )

    brier_difference.append(
        market_brier - combined_brier
    )


auc_difference = np.array(auc_difference)

lower, median, upper = np.percentile(
    auc_difference,
    [2.5, 50, 97.5]
)

fig, ax = plt.subplots(figsize=(9, 5))

ax.hist(
    auc_difference,
    bins=40,
    density=True,
    alpha=0.7
)

ax.axvline(
    0,
    linestyle="--",
    linewidth=1.5,
    label="No improvement"
)

ax.axvline(
    lower,
    linestyle=":",
    linewidth=1
)

ax.axvline(
    upper,
    linestyle=":",
    linewidth=1
)

ax.set_xlabel(
    "Combined AUC - Market-only AUC"
)

ax.set_ylabel("Bootstrap density")

ax.set_title(
    "Bootstrap Distribution of HMM Incremental Predictive Value"
)

ax.legend()

plt.tight_layout()
plt.savefig(
    figure_dir / "bootstrap_auc_difference.png",
    dpi=300
)
plt.close()


print(f"Figures saved to {figure_dir}")
print()
print(
    f"AUC difference 95% interval: "
    f"[{lower:.4f}, {upper:.4f}]"
)