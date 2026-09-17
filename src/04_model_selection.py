from pathlib import Path
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

root = Path(__file__).resolve().parents[1]

input_file = root / "data" / "processed" / "sp500_returns.csv"
output_file = root / "data" / "processed" / "model_selection.csv"

df = pd.read_csv(
    input_file,
    index_col="Date",
    parse_dates=["Date"]
)

X = df[["Return_scaled"]].values

n = len(X)
d = X.shape[1]

results = []

for k in range(2, 6):
    best_score = -np.inf

    for seed in range(10):
        model = GaussianHMM(
            n_components=k,
            covariance_type="full",
            n_iter=1000,
            random_state=seed
        )

        model.fit(X)
        score = model.score(X)

        if score > best_score:
            best_score = score

    parameters = (
        (k - 1)
        + k * (k - 1)
        + k * d
        + k * d * (d + 1) / 2
    )

    aic = -2 * best_score + 2 * parameters
    bic = -2 * best_score + parameters * np.log(n)

    results.append({
        "States": k,
        "LogLikelihood": best_score,
        "Parameters": int(parameters),
        "AIC": aic,
        "BIC": bic
    })

results = pd.DataFrame(results)

results.to_csv(output_file, index=False)

print(results.round(2))