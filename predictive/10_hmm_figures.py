from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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


train = load("train_filtered.csv")
validation = load("validation_filtered.csv")
test = load("test_filtered.csv")

df = pd.concat([train, validation, test]).sort_index()

with open(data_dir / "hmm_model.pkl", "rb") as f:
    model = pickle.load(f)

state_order = pd.read_csv(data_dir / "hmm_state_order.csv")

labels = {
    int(row["State"]): row["Regime"]
    for _, row in state_order.iterrows()
}

ordered_states = state_order["State"].astype(int).tolist()
ordered_labels = state_order["Regime"].tolist()

colors = {
    "Low volatility": "green",
    "Medium volatility": "orange",
    "High volatility": "red"
}

train_end = pd.Timestamp("2015-12-31")
validation_end = pd.Timestamp("2020-12-31")


fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(
    df.index,
    df["Close"],
    linewidth=1,
    color="black"
)

regime_change = df["Regime"].ne(df["Regime"].shift())
groups = regime_change.cumsum()

used_labels = set()

for _, block in df.groupby(groups):
    regime = block["Regime"].iloc[0]

    label = regime if regime not in used_labels else None

    ax.axvspan(
        block.index[0],
        block.index[-1],
        color=colors[regime],
        alpha=0.18,
        label=label
    )

    used_labels.add(regime)

ax.axvline(
    train_end,
    color="black",
    linestyle="--",
    linewidth=1,
    label="End of training"
)

ax.axvline(
    validation_end,
    color="black",
    linestyle=":",
    linewidth=1,
    label="End of validation"
)

ax.set_yscale("log")
ax.set_xlabel("Date")
ax.set_ylabel("S&P 500")
ax.set_title("Filtered Market Regimes")
ax.legend()

plt.tight_layout()
plt.savefig(
    figure_dir / "predictive_market_regimes.png",
    dpi=300
)
plt.close()


fig, ax = plt.subplots(figsize=(10, 6))

for regime in ordered_labels:
    values = df.loc[
        df["Regime"] == regime,
        "LogReturn"
    ]

    ax.hist(
        values,
        bins=80,
        density=True,
        alpha=0.45,
        label=regime,
        color=colors[regime]
    )

ax.set_xlabel("Daily log return")
ax.set_ylabel("Density")
ax.set_title("Return Distributions by Filtered Regime")
ax.legend()

plt.tight_layout()
plt.savefig(
    figure_dir / "predictive_return_distributions.png",
    dpi=300
)
plt.close()


transition = model.transmat_[
    np.ix_(ordered_states, ordered_states)
]

fig, ax = plt.subplots(figsize=(7, 6))

image = ax.imshow(
    transition,
    vmin=0,
    vmax=1,
    cmap="Blues"
)

ax.set_xticks(range(len(ordered_labels)))
ax.set_yticks(range(len(ordered_labels)))

ax.set_xticklabels(
    ordered_labels,
    rotation=30,
    ha="right"
)

ax.set_yticklabels(ordered_labels)

ax.set_xlabel("Next regime")
ax.set_ylabel("Current regime")
ax.set_title("Training-Period HMM Transition Matrix")

for i in range(len(ordered_labels)):
    for j in range(len(ordered_labels)):
        value = transition[i, j]

        ax.text(
            j,
            i,
            f"{value:.3f}",
            ha="center",
            va="center"
        )

fig.colorbar(image, ax=ax)

plt.tight_layout()
plt.savefig(
    figure_dir / "predictive_transition_matrix.png",
    dpi=300
)
plt.close()


fig, ax = plt.subplots(figsize=(14, 5))

ax.plot(
    df.index,
    df["P_high_volatility"],
    linewidth=1
)

ax.axvline(
    train_end,
    color="black",
    linestyle="--",
    linewidth=1,
    label="End of training"
)

ax.axvline(
    validation_end,
    color="black",
    linestyle=":",
    linewidth=1,
    label="End of validation"
)

ax.set_ylim(0, 1)
ax.set_xlabel("Date")
ax.set_ylabel("Filtered probability")
ax.set_title("Filtered Probability of High-Volatility Regime")
ax.legend()

plt.tight_layout()
plt.savefig(
    figure_dir / "predictive_high_volatility_probability.png",
    dpi=300
)
plt.close()


print(f"Figures saved to {figure_dir}")