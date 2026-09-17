from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

root = Path(__file__).resolve().parents[1]

data_file = root / "data" / "processed" / "sp500_hmm.csv"
transition_file = root / "data" / "processed" / "transition_matrix.csv"
figure_dir = root / "figures"

figure_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(
    data_file,
    index_col="Date",
    parse_dates=["Date"]
)

transition = pd.read_csv(
    transition_file,
    index_col=0
)

colors = {
    "Low volatility": "green",
    "Medium volatility": "orange",
    "High volatility": "red"
}

fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(
    df.index,
    df["Close"],
    color="black",
    linewidth=0.8
)

blocks = (df["Regime"] != df["Regime"].shift()).cumsum()

for _, block in df.groupby(blocks):
    regime = block["Regime"].iloc[0]

    ax.axvspan(
        block.index[0],
        block.index[-1],
        color=colors[regime],
        alpha=0.2
    )

handles = [
    plt.Line2D(
        [0],
        [0],
        color=colors[regime],
        linewidth=6,
        alpha=0.4
    )
    for regime in colors
]

ax.legend(
    handles,
    list(colors.keys())
)

ax.set_yscale("log")
ax.set_title("S&P 500 Hidden Market Regimes, 1990–2026")
ax.set_xlabel("Date")
ax.set_ylabel("S&P 500")

plt.tight_layout()
plt.savefig(
    figure_dir / "market_regimes.png",
    dpi=300
)
plt.close()


fig, ax = plt.subplots(figsize=(10, 6))

for regime in colors:
    returns = df.loc[
        df["Regime"] == regime,
        "LogReturn"
    ]

    ax.hist(
        returns,
        bins=80,
        density=True,
        alpha=0.45,
        label=regime
    )

ax.set_title("Daily Return Distributions by Hidden Regime")
ax.set_xlabel("Daily Log Return")
ax.set_ylabel("Density")
ax.legend()

plt.tight_layout()
plt.savefig(
    figure_dir / "return_distributions.png",
    dpi=300
)
plt.close()


fig, ax = plt.subplots(figsize=(7, 6))

image = ax.imshow(
    transition.values,
    vmin=0,
    vmax=1
)

for i in range(len(transition)):
    for j in range(len(transition)):
        ax.text(
            j,
            i,
            f"{transition.iloc[i, j]:.3f}",
            ha="center",
            va="center"
        )

ax.set_xticks(range(len(transition)))
ax.set_yticks(range(len(transition)))

ax.set_xticklabels(transition.columns)
ax.set_yticklabels(transition.index)

ax.set_xlabel("Next regime")
ax.set_ylabel("Current regime")
ax.set_title("Fitted HMM Transition Probabilities")

fig.colorbar(image)

plt.tight_layout()
plt.savefig(
    figure_dir / "transition_matrix.png",
    dpi=300
)
plt.close()


fig, ax = plt.subplots(figsize=(14, 5))

p = df["P_high_volatility"]

ax.plot(
    df.index,
    p,
    linewidth=1
)

ax.fill_between(
    df.index,
    p,
    alpha=0.25
)

ax.set_title(
    "Posterior Probability of the High-Volatility Regime"
)
ax.set_xlabel("Date")
ax.set_ylabel("Probability")
ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig(
    figure_dir / "high_volatility_probability.png",
    dpi=300
)
plt.close()

print(f"Figures saved to {figure_dir}")