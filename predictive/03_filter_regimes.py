from pathlib import Path
import pickle
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[1]

data_dir = root / "data" / "predictive"

train = pd.read_csv(
    data_dir / "train.csv",
    index_col="Date",
    parse_dates=["Date"]
)

validation = pd.read_csv(
    data_dir / "validation.csv",
    index_col="Date",
    parse_dates=["Date"]
)

test = pd.read_csv(
    data_dir / "test.csv",
    index_col="Date",
    parse_dates=["Date"]
)

with open(data_dir / "hmm_model.pkl", "rb") as f:
    model = pickle.load(f)

state_info = pd.read_csv(data_dir / "hmm_state_order.csv")

labels = dict(
    zip(state_info["State"], state_info["Regime"])
)

means = model.means_[:, 0]
variances = model.covars_[:, 0, 0]
transition = model.transmat_
start = model.startprob_

def emission_prob(x):
    return (
        1 / np.sqrt(2 * np.pi * variances)
        * np.exp(
            -0.5 * ((x - means) ** 2) / variances
        )
    )

def filter_series(values, previous=None):
    probabilities = []

    for i, x in enumerate(values):
        likelihood = emission_prob(x)

        if previous is None and i == 0:
            current = start * likelihood
        else:
            predicted = previous @ transition
            current = predicted * likelihood

        current = current / current.sum()

        probabilities.append(current)
        previous = current

    return np.array(probabilities), previous

train_prob, last_prob = filter_series(
    train["Return_scaled"].values
)

validation_prob, last_prob = filter_series(
    validation["Return_scaled"].values,
    last_prob
)

test_prob, last_prob = filter_series(
    test["Return_scaled"].values,
    last_prob
)

def add_probabilities(df, probabilities):
    result = df.copy()

    for state in range(model.n_components):
        label = labels[state]
        column = "P_" + label.lower().replace(" ", "_")
        result[column] = probabilities[:, state]

    result["State"] = probabilities.argmax(axis=1)
    result["Regime"] = result["State"].map(labels)

    return result

train = add_probabilities(train, train_prob)
validation = add_probabilities(validation, validation_prob)
test = add_probabilities(test, test_prob)

train.to_csv(data_dir / "train_filtered.csv")
validation.to_csv(data_dir / "validation_filtered.csv")
test.to_csv(data_dir / "test_filtered.csv")

print("Last training probabilities:")
print(train.iloc[-1][[
    "P_low_volatility",
    "P_medium_volatility",
    "P_high_volatility"
]])

print("\nValidation regime counts:")
print(validation["Regime"].value_counts())

print("\nTest regime counts:")
print(test["Regime"].value_counts())