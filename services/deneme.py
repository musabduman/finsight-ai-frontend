import yfinance as yf
t = yf.Ticker("THYAO.IS")
print(t.financials)
print(t.balance_sheet)