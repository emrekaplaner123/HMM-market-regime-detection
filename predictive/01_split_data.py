from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parents[1]

input_file = root / "data" / "processed" / "sp500_returns.csv"
output_dir = root / "data" / "predictive"

output_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(
    input_file,
    index_col="Date",
    parse_dates=["Date"]
)

train = df.loc["1990-01-01":"2015-12-31"].copy()
validation = df.loc["2016-01-01":"2020-12-31"].copy()
test = df.loc["2021-01-01":].copy()

mean_return = train["LogReturn"].mean()
std_return = train["LogReturn"].std(ddof=0)

for data in [train, validation, test]:
    data["Return_scaled"] = (
        (data["LogReturn"] - mean_return) / std_return
    )

train.to_csv(output_dir / "train.csv")
validation.to_csv(output_dir / "validation.csv")
test.to_csv(output_dir / "test.csv")

pd.DataFrame({
    "mean": [mean_return],
    "std": [std_return]
}).to_csv(
    output_dir / "return_scaler.csv",
    index=False
)

print("Train:")
print(train.index.min(), "to", train.index.max(), len(train))

print("\nValidation:")
print(validation.index.min(), "to", validation.index.max(), len(validation))

print("\nTest:")
print(test.index.min(), "to", test.index.max(), len(test))

print(f"\nTraining mean: {mean_return:.8f}")
print(f"Training standard deviation: {std_return:.8f}")