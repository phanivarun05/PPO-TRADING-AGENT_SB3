import yfinance as yf
import pandas as pd
import os

def download_data(symbol="AAPL", start="2015-01-01", end="2024-01-01"):
    os.makedirs("data", exist_ok=True)

    raw = yf.download(symbol, start=start, end=end)

    
    df = pd.DataFrame(raw)

    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]]

    df.to_csv("data/raw_data.csv", index=False)
    print("Data saved successfully")

if __name__ == "__main__":
    download_data()
