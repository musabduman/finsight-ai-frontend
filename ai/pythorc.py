import math
import torch
import warnings

import numpy as np
import torch.nn as nn
import pandas as pd
import torch.optim as optim

from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

class deeplearning:
    def __init__(self):
        self.model=nn.Sequential(nn.Linear(6,32),
                                 nn.ReLU(),
                                 nn.Linear(32,1)
                                 )
    
    def analiz_et(self,df):
        if df is None or df.empty:
                return {"yön": "VERİ YOK", "güven": "0"}
        data=df.copy()

        df = df.sort_index(ascending=True)
        
        df['Getiri'] = df['Close'].pct_change()
        df['Hacim_degisimi'] = df['Volume'].pct_change()
        df['Oynaklık'] = (df['High'] - df['Low']) / df['Close']

        #macd
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Fark'] = df['MACD'] - df['Signal_Line'] # Histogram
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0))
        lose = (-delta.where(delta < 0, 0))
        avg_gain = gain.ewm(com=13, adjust=False).mean()
        avg_lose = lose.ewm(com=13, adjust=False).mean()
        rs = avg_gain / avg_lose
        df['RSI'] = 100 - (100 / (1 + rs))
        
        #bollinger
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['STD_20'] = df['Close'].rolling(window=20).std()
        df['Bollinger_Upper'] = df['SMA_20'] + (df['STD_20'] * 2)
        df['Bollinger_Lower'] = df['SMA_20'] - (df['STD_20'] * 2)
        df['Bollinger_Konum'] = (df['Close'] - df['Bollinger_Lower']) / (df['Bollinger_Upper'] - df['Bollinger_Lower'])
        
        #SMA_50,SMA_200
        df['SMA_50'] = df['Close'] / df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'] / df['Close'].rolling(window=200).mean()
        
        features_to_lag = ['Getiri', 'RSI', 'Hacim_degisimi', 'MACD_Fark']
        for feature in features_to_lag:
            df[f'{feature}_Lag1'] = df[feature].shift(1)
            df[f'{feature}_Lag2'] = df[feature].shift(2)
        
        df['Gun'] = df.index.dayofweek
        df['Momentum'] = df['Close'] / df['Close'].shift(10)

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df = df.astype('float32')

        df['Target']=df['Close'].shift(-1)
        
        features = ['Close', 'RSI', 'MACD', 'Bollinger_Konum', 'Hacim_degisimi', 'Momentum']

        son_gün_verisi=df[features].iloc[[-1]].values

        df_clean=df.dropna(subset=features+['Target'])

        x_data=df_clean[features].values
        y_data=df_clean[['Target']].values

        x_scaler=MinMaxScaler()
        y_scaler=MinMaxScaler()
        
        x_scaled=x_scaler.fit_transform(x_data)
        y_scaled=y_scaler.fit_transform(y_data)

        x=torch.tensor(x_scaled,dtype=torch.float32)
        y=torch.tensor(y_scaled,dtype=torch.float32)
        
        hata=nn.MSELoss()
        optime=optim.Adam(self.model.parameters(), lr=0.005)

        for epoch in range(1000):
            optime.zero_grad()

            y_pred=self.model(x)
            loss=hata(y_pred,y)
            
            loss.backward()

            optime.step()

        son_gün=x_scaler.transform(son_gün_verisi)
        son_gün_tensor=torch.tensor(son_gün,dtype=torch.float32)
        
        sonuc=self.model(son_gün_tensor)
        sonuc=y_scaler.inverse_transform(sonuc.detach().numpy())

        # --- ESKİ RETURN SATIRINI SİL VE BURAYI YAPIŞTIR ---
        
        tahmin_degeri = sonuc[0][0]
        suanki_fiyat = df['Close'].iloc[-1]
        
        # Gelecekteki fiyat şu ankinden büyükse YÜKSELİŞ, değilse DÜŞÜŞ
        yon = "YÜKSELİŞ" if tahmin_degeri > suanki_fiyat else "DÜŞÜŞ"
        
        # Tahminle şu anki fiyat arasındaki makasa göre basit bir AI güven skoru
        fark_orani = abs(tahmin_degeri - suanki_fiyat) / suanki_fiyat
        guven_skoru = min(99, int(60 + (fark_orani * 1000))) 
        # 🛡️ 1. NAN KORUMASI: Hedef fiyat NaN gelirse 0.0 yap
        if pd.isna(tahmin_degeri) or math.isnan(tahmin_degeri):
            tahmin_degeri = 0.0
            yon = "Veri Yetersiz"
            
        # 🛡️ 2. NAN KORUMASI: Güven oranı NaN gelirse 0.0 yap
        if pd.isna(guven_skoru) or math.isnan(guven_skoru):
            guven_skoru = 0.0
        # main.py dosyasının beklediği SÖZLÜK (Dictionary) formatında cevap veriyoruz:
        return {
            "suanki_fiyat": round(float(suanki_fiyat), 2),
            "tahmin": round(float(tahmin_degeri), 2),
            "yön": yon,
            "güven": guven_skoru
        }
        
        