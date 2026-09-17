from pathlib import Path
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[1]
data_dir = root / "data" / "predictive"

train = pd.read_csv(
    data_dir / "train_filtered.csv",
    index_col="Date",
    parse_dates=["Date"]
)

validation = pd.read_csv(
    data_dir / "validation_filtered.csv",
    index_col="Date",
    parse_dates=["Date"]
)

test = pd.read_csv(
    data_dir / "test_filtered.csv",
    index_col="Date",
    parse_dates=["Date"]
)

train["Split"] = "train"
validation["Split"] = "validation"
test["Split"] = "test"

df = pd.concat([train, validation, test]).sort_index()

df["Return_5"] = df["LogReturn"].rolling(5).sum()
df["Return_20"] = df["LogReturn"].rolling(20).sum()

df["Volatility_5"] = (
    df["LogReturn"].rolling(5).std(ddof=0) * np.sqrt(252)
)

df["Volatility_20"] = (
    df["LogReturn"].rolling(20).std(ddof=0) * np.sqrt(252)
)

df["Volatility_change"] = (
    df["Volatility_5"] - df["Volatility_20"]
)

rolling_high = df["Close"].rolling(20).max()

df["Drawdown_20"] = (
    df["Close"] / rolling_high - 1
)

medium = df["Regime"] == "Medium volatility"

groups = (~medium).cumsum()

df["Days_in_medium"] = (
    medium.groupby(groups).cumsum()
)


def add_target(data):
    data = data.copy()

    next_regime = None
    targets = []

    for regime in reversed(data["Regime"].tolist()):
        if regime != "Medium volatility":
            next_regime = regime
            targets.append(np.nan)
        else:
            if next_regime == "High volatility":
                targets.append(1)
            elif next_regime == "Low volatility":
                targets.append(0)
            else:
                targets.append(np.nan)

    data["Target"] = list(reversed(targets))

    return data


train_part = add_target(df[df["Split"] == "train"])
validation_part = add_target(df[df["Split"] == "validation"])
test_part = add_target(df[df["Split"] == "test"])

df = pd.concat([
    train_part,
    validation_part,
    test_part
]).sort_index()

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
    "P_medium_volatility",
    "P_high_volatility"
]

df = df[df["Regime"] == "Medium volatility"]

df = df.dropna(
    subset=features + ["Target"]
)

train_ml = df[df["Split"] == "train"].copy()
validation_ml = df[df["Split"] == "validation"].copy()
test_ml = df[df["Split"] == "test"].copy()

train_ml.to_csv(data_dir / "train_transition.csv")
validation_ml.to_csv(data_dir / "validation_transition.csv")
test_ml.to_csv(data_dir / "test_transition.csv")

for name, data in [
    ("Train", train_ml),
    ("Validation", validation_ml),
    ("Test", test_ml)
]:
    print(f"\n{name}:")
    print(f"Observations: {len(data)}")
    print(data["Target"].value_counts().sort_index())
    print()
    print(
        data["Target"]
        .value_counts(normalize=True)
        .sort_index()
        .round(4)
    )