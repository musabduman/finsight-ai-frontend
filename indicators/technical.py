import numpy as np
import pandas as pd
class TechnicalAnalyzer:
    def __init__(self,df):
        self.df=df.copy()

    def bollinger(self, window=20):
        self.df['SMA'] = self.df['Close'].rolling(window=20).mean()
        
        std = self.df['Close'].rolling(window=window).std()
        self.df['Upper'] = self.df['SMA'] + 2 * std
        self.df['Lower'] = self.df['SMA'] - 2 * std
        self.df['Width'] = (self.df['Upper'] - self.df['Lower']) / self.df['SMA']
        self.df['BOLL_signal'] = np.select(
            [self.df['Close'] > self.df['Upper'], self.df['Close'] < self.df['Lower']],
            [1, -1],
            default=0
        )
        return self.df

    def volume_trend(self, window=10):
        self.df['Volume_signal'] = np.where(
            self.df['Volume'] > self.df['Volume'].rolling(window=window).mean(), 1, 0
        )
        return self.df['Volume_signal']

    def calcu_volatility(self,window=20):
        self.df['Returns'] = self.df['Close'].pct_change()
        self.df['Volatility'] = self.df['Returns'].rolling(window=window).std()
        return self.df['Volatility']

    def calcu_macd(self):
        exp1 = self.df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = self.df['Close'].ewm(span=26, adjust=False).mean()
        self.df['MACD'] = exp1 - exp2
        self.df['Signal_line'] = self.df['MACD'].ewm(span=9, adjust=False).mean()
        self.df['MACD_signal'] = np.where(
            (self.df['MACD'] > self.df['Signal_line']) & 
            (self.df['MACD'].shift(1) <= self.df['Signal_line'].shift(1)),1,
        np.where(
            (self.df['MACD'] < self.df['Signal_line']) &
            (self.df['MACD'].shift(1) >= self.df['Signal_line'].shift(1)),-1,
            0
        ))
        return self.df

    def calcu_pivot(self):
        self.df['Pivot'] = (self.df['High'] + self.df['Low'] + self.df['Close']) / 3
        self.df['R1'] = (2 * self.df['Pivot']) - self.df['Low']
        self.df['S1'] = (2 * self.df['Pivot']) - self.df['High']
        return self.df

    def calculate_fibonacci_levels(self, period=20):
        recent_df = self.df.tail(period)
        high = recent_df['High'].max()
        low = recent_df['Low'].min()
        diff = high - low
        
        return {
            "fib_high": round(high, 2),
            "fib_low": round(low, 2),
            "fib_618": round(high - 0.382 * diff, 2), 
            "fib_382": round(high - 0.618 * diff, 2)
        }

    # 2. Vektörel SBS Hesaplama (Satır satır değil, tek seferde hesaplar!)
    def calculate_sbs_vectorized(self):
        # Yüzdelik değişimi tam sayı formatında al (Örn: 0.03 yerine 3.0)
        self.df['Percent_Change'] = self.df['Close'].pct_change() * 100
        
        # Fiyat Skoru (FS)
        fs = 50 + (self.df['Percent_Change'] * 7.14)
        fs = fs.clip(0, 100) # max ve min sınırlandırması
        
        # Hacim Skoru (VS)
        vol_avg = self.df['Volume'].rolling(window=20).mean()
        hacim_orani = self.df['Volume'] / vol_avg
        yon = np.where(self.df['Percent_Change'] >= 0, 1, -1)
        
        vs = 50 + ((hacim_orani - 1) * 25 * yon)
        vs = vs.clip(0, 100)
        
        # MFI (Para Akışı). Eğer kodunda MFI yoksa, RSI'ı proxy (vekil) olarak kullanabiliriz.
        # Daha sonra gerçek MFI eklersen burayı self.df['MFI'] yaparsın.
        mfi_proxy = self.df['RSI'] 
        
        # Final SBS Skoru
        self.df['SBS'] = (mfi_proxy * 0.40) + (fs * 0.40) + (vs * 0.20)
        self.df['SBS'] = self.df['SBS'].round(2)
        
        return self.df
    def teknik_baslat(self):
        
        delta = self.df['Close'].diff()
        gain = (delta.where(delta > 0, 0))
        lose = (-delta.where(delta < 0, 0))
        avg_gain = gain.ewm(com=13, adjust=False).mean()
        avg_lose = lose.ewm(com=13, adjust=False).mean()
        rs = avg_gain / avg_lose
        self.df['RSI'] = 100 - (100 / (1 + rs))
        
        self.df['SMA_50'] = self.df['Close'].rolling(window=50).mean()
        self.df['SMA_200'] = self.df['Close'].rolling(window=200).mean()
        self.df['SMA_20'] = self.df['Close'].rolling(window=20).mean() 
        
        self.df['Volume_signal'] = self.volume_trend(window=60)
        self.df['Volatility'] = self.calcu_volatility(window=20)
        
        self.bollinger(window=20)
        self.calcu_macd()
        self.calcu_pivot()
        
        self.calculate_sbs_vectorized()

        # NaN verileri temizle
        self.df = self.df.dropna()
        
        # Fibonacci seviyelerini (son güncel duruma göre) hesapla
        fib_20 = self.calculate_fibonacci_levels(period=20)
        fib_200 = self.calculate_fibonacci_levels(period=200)

        return self.df.dropna(), fib_20, fib_200
    
def teknik_analiz(df):
    analizor = TechnicalAnalyzer(df)
    return analizor.teknik_baslat()