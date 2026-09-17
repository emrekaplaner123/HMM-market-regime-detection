from pathlib import Path
import pandas as pd
import yfinance as yf

root = Path(__file__).resolve().parents[1]
output = root / "data" / "raw" / "sp500_raw.csv"
output.parent.mkdir(parents=True, exist_ok=True)

data = yf.download(
    "^GSPC",
    start="1990-01-01",
    auto_adjust=False,
    progress=False
)

if data.empty:
    raise RuntimeError("No data downloaded")

close = data["Close"]

if isinstance(close, pd.DataFrame):
    close = close.iloc[:, 0]

df = pd.DataFrame({"Close": close})
df.index.name = "Date"

df.to_csv(output)

print(df.head())
print(df.tail())
print(f"\nObservations: {len(df)}")
print(f"Saved to {output}")