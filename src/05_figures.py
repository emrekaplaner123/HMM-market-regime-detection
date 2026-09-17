from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "sp500_regimes.csv"
FIGURE_PATH = PROJECT_ROOT / "figures"

FIGURE_PATH.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_PATH, index_col=0, parse_dates=True)

state_order = (
    df.groupby("State")["Volatility20"]
    .mean()
    .sort_values()
    .index
    .tolist()
)

labels = {
    state_order[0]: "Low volatility",
    state_order[1]: "Normal",
    state_order[2]: "High volatility"
}

df["Regime"] = df["State"].map(labels)

colors = {
    state_order[0]: "green",
    state_order[1]: "orange",
    state_order[2]: "red"
}

fig, ax = plt.subplots(figsize=(14, 6))

for state in sorted(df["State"].unique()):
    mask = df["State"] == state

    ax.scatter(
        df.index[mask],
        df.loc[mask, "Close"],
        s=4,
        color=colors[state],
        label=labels[state]
    )

ax.set_yscale("log")
ax.set_title("S&P 500 Market Regimes Identified by the HMM")
ax.set_xlabel("Date")
ax.set_ylabel("S&P 500")
ax.legend()

plt.tight_layout()
plt.savefig(FIGURE_PATH / "sp500_regimes.png", dpi=300)
plt.show()


fig, ax = plt.subplots(figsize=(10, 6))

for state in sorted(df["State"].unique()):
    subset = df[df["State"] == state]

    ax.scatter(
        subset["Volatility20"],
        subset["LogReturn"],
        s=8,
        alpha=0.4,
        color=colors[state],
        label=labels[state]
    )

ax.set_title("Returns and Volatility by Hidden Market Regime")
ax.set_xlabel("20-Day Annualized Volatility")
ax.set_ylabel("Daily Log Return")
ax.legend()

plt.tight_layout()
plt.savefig(FIGURE_PATH / "regime_scatter.png", dpi=300)
plt.show()


states = df["State"].values
n_states = len(np.unique(states))

transition_counts = np.zeros((n_states, n_states))

for current_state, next_state in zip(states[:-1], states[1:]):
    transition_counts[current_state, next_state] += 1

transition_matrix = (
    transition_counts
    / transition_counts.sum(axis=1, keepdims=True)
)

fig, ax = plt.subplots(figsize=(7, 6))

image = ax.imshow(transition_matrix)

for i in range(n_states):
    for j in range(n_states):
        ax.text(
            j,
            i,
            f"{transition_matrix[i, j]:.3f}",
            ha="center",
            va="center"
        )

ax.set_xticks(range(n_states))
ax.set_yticks(range(n_states))

ax.set_xticklabels([labels[i] for i in range(n_states)])
ax.set_yticklabels([labels[i] for i in range(n_states)])

ax.set_xlabel("Next Regime")
ax.set_ylabel("Current Regime")
ax.set_title("Empirical Regime Transition Probabilities")

fig.colorbar(image)

plt.tight_layout()
plt.savefig(FIGURE_PATH / "transition_matrix.png", dpi=300)
plt.show()