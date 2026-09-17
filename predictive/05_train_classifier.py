from pathlib import Path

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix
)

root = Path(__file__).resolve().parents[1]
data_dir = root / "data" / "predictive"

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

X_train = train[features]
y_train = train["Target"].astype(int)

X_validation = validation[features]
y_validation = validation["Target"].astype(int)

X_test = test[features]
y_test = test["Target"].astype(int)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_validation_scaled = scaler.transform(X_validation)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(
    max_iter=1000
)

model.fit(X_train_scaled, y_train)


def evaluate(name, X, y):
    probability = model.predict_proba(X)[:, 1]
    prediction = (probability >= 0.5).astype(int)

    print(f"\n{name}")
    print(f"Accuracy:          {accuracy_score(y, prediction):.4f}")
    print(f"Balanced accuracy: {balanced_accuracy_score(y, prediction):.4f}")
    print(f"ROC-AUC:           {roc_auc_score(y, probability):.4f}")
    print(f"PR-AUC:            {average_precision_score(y, probability):.4f}")
    print(f"Brier score:       {brier_score_loss(y, probability):.4f}")

    print("\nConfusion matrix:")
    print(confusion_matrix(y, prediction))


evaluate(
    "Validation",
    X_validation_scaled,
    y_validation
)

evaluate(
    "Test",
    X_test_scaled,
    y_test
)

coefficients = pd.DataFrame({
    "Feature": features,
    "Coefficient": model.coef_[0]
})

coefficients["AbsCoefficient"] = (
    coefficients["Coefficient"].abs()
)

coefficients = coefficients.sort_values(
    "AbsCoefficient",
    ascending=False
)

print("\nCoefficients:")
print(
    coefficients[
        ["Feature", "Coefficient"]
    ].to_string(index=False)
)