import pandas as pd
import yfinance as yf
import streamlit as st
import requests

# ---------------------------
# Session Fix (stabil veri çekme)
# ---------------------------
session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0"
})
yf.shared._requests = session


# ---------------------------
# Symbol Normalize
# ---------------------------
def normalize_symbol(symbol: str):
    tr_to_en = str.maketrans("ıiğüşöçIİĞÜŞÖÇ", "IIGUSOCIIGUSOC")
    clean = str(symbol).translate(tr_to_en).upper().strip()
    
    if not clean.endswith(".IS"):
        clean += ".IS"
    
    return clean


# ---------------------------
# PRICE DATA
# ---------------------------
@st.cache_data(ttl=3600)
def get_price_data(symbol):
    return yf.download(symbol, period="3y", progress=False)


# ---------------------------
# FAST INFO
# ---------------------------
@st.cache_data(ttl=3600)
def get_fast_info(symbol):
    return yf.Ticker(symbol).fast_info


# ---------------------------
# MAIN STOCK FETCH
# ---------------------------
def get_stock(symbol):
    clean = normalize_symbol(symbol)

    try:
        df = get_price_data(clean)
        info = get_fast_info(clean)

        if df is None or df.empty:
            return clean, None, None

        return clean, df, info

    except Exception as e:
        print("get_stock error:", e)
        return clean, None, None


# Alias - app.py ve watchlist.py get_stock_data adıyla çağırıyor
get_stock_data = get_stock


# ---------------------------
# BULK STOCK FETCH
# ---------------------------
@st.cache_data(ttl=3600)
def get_bulk_stocks(symbols: list):
    """Tüm sembolleri tek seferde çeker. {clean_symbol: df} dict döner."""
    try:
        symbols_str = " ".join([normalize_symbol(s) for s in symbols])
        raw = yf.download(symbols_str, period="3y", progress=False, group_by="ticker")

        if raw is None or raw.empty:
            return None

        result = {}
        for s in symbols:
            clean = normalize_symbol(s)
            try:
                df = raw[clean].copy() if clean in raw.columns.get_level_values(0) else pd.DataFrame()
                if not df.empty:
                    result[clean] = df
            except Exception:
                continue

        return result if result else None

    except Exception as e:
        print("get_bulk_stocks error:", e)
        return None


# ---------------------------
# FUNDAMENTAL CALC
# ---------------------------
def get_temel_hesapla(symbol):
    ticker = yf.Ticker(symbol)

    try:
        income = ticker.financials
        balance = ticker.balance_sheet
        fast = ticker.fast_info

        net_kar = income.loc['Net Income'].iloc[0]
        ozkaynak = balance.loc['Stockholders Equity'].iloc[0]
        piyasa_degeri = fast['market_cap']

        fk = piyasa_degeri / net_kar if net_kar > 0 else "Zararda"
        pd_dd = piyasa_degeri / ozkaynak if ozkaynak > 0 else "Yok"

        return {
            "FK": round(fk, 2) if isinstance(fk, (int, float)) else fk,
            "PD/DD": round(pd_dd, 2) if isinstance(pd_dd, (int, float)) else pd_dd,
        }

    except Exception as e:
        print("temel hesap hata:", e)
        return {"FK": "Yok", "PD/DD": "Yok"}