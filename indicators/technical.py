import numpy as np
import pandas as pd
class TechnicalAnalyzer:
    def __init__(self,df):
        self.df=df.copy()

    def bollinger(self, window):
        self.df['SMA'] = self.df['Close'].rolling(window=20).mean()
        std = self.df['Close'].rolling(window=window).std()
        self.df['Upper'] = self.df['SMA'] + 2 * std
        self.df['Lower'] = self.df['SMA'] - 2 * std
        self.df['Width'] = (self.df['Upper'] - self.df['Lower']) / self.df['SMA']
        self.df['Signal'] = np.select(
            [self.df['Close'] > self.df['Upper'], self.df['Close'] < self.df['Lower']],
            [1, -1],
            default=0
        )
        return self.df

    def volume_trend(self, window=10):
        self.df['volume_signal'] = np.where(
            self.df['Volume'] > self.df['Volume'].rolling(window=window).mean(), 1, 0
        )
        return self.df['volume_signal']

    def calcu_volatility(self,window=20):
        self.df['Returns'] = self.df['Close'].pct_change()
        self.df['Volatility'] = self.df['Returns'].rolling(window=window).std()
        return self.df['Volatility']

    def calcu_macd(self):
        exp1 = self.df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = self.df['Close'].ewm(span=26, adjust=False).mean()
        self.df['MACD'] = exp1 - exp2
        self.df['Signal_line'] = self.df['MACD'].ewm(span=9, adjust=False).mean()
        self.df['MACD_signal'] = np.where(self.df['MACD'] > self.df['Signal_line'], 1, -1)
        return self.df

    def calcu_pivot(self):
        self.df['Pivot'] = (self.df['High'] + self.df['Low'] + self.df['Close']) / 3
        self.df['R1'] = (2 * self.df['Pivot']) - self.df['Low']
        self.df['S1'] = (2 * self.df['Pivot']) - self.df['High']
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
        self.df['Volume_signal'] = self.volume_trend(window=60)
        self.df['Volatility'] = self.calcu_volatility(window=20)
        self.df = self.bollinger(window=20)
        self.df = self.calcu_macd()
        self.df = self.calcu_pivot()
        
        return self.df.dropna()
    
def teknik_analiz(df):
    analizor = TechnicalAnalyzer(df)
    return analizor.teknik_baslat()