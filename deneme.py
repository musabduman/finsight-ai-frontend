import yfinance as yf
df = yf.download("THYAO.IS", period="5d", progress=False)
print(df)