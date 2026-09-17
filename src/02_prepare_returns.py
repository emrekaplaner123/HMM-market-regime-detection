from pathlib import Path
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[1]

input_file = root / "data" / "raw" / "sp500_raw.csv"
output_file = root / "data" / "processed" / "sp500_returns.csv"

output_file.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(input_file, index_col="Date", parse_dates=["Date"])

df["LogReturn"] = np.log(df["Close"]).diff()
df = df.dropna()

mean_return = df["LogReturn"].mean()
std_return = df["LogReturn"].std(ddof=0)

df["Return_scaled"] = (
    (df["LogReturn"] - mean_return) / std_return
)

df.to_csv(output_file)

print(df.head())
print()
print(df["LogReturn"].describe())
print(f"\nSaved to {output_file}")