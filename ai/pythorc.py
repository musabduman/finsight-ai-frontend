import math
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib # .gz dosyalarını açmak için
import os

class deeplearning:
    def __init__(self):
        # 1. Eğittiğimiz modelin BİREBİR aynısını burada da tanımlıyoruz (Şablon)
        self.model = nn.Sequential(
            nn.Linear(6, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 1)
        )
        
        # 2. Dosya yollarını belirliyoruz (Streamlit'te hata almamak için os.path kullanmak iyidir)
        klasör = os.path.dirname(__file__) # Şu anki dosyanın olduğu klasör
        model_yolu = os.path.join(klasör, "kahin_model.pth")
        x_scaler_yolu = os.path.join(klasör, "x_scaler.gz")
        y_scaler_yolu = os.path.join(klasör, "y_scaler.gz")

        # 3. Eğitilmiş beyni ve sözlükleri yüklüyoruz
        try:
            # Model ağırlıklarını yükle
            state_dict=torch.load(model_yolu, map_location=torch.device('cpu'), weights_only=True)
            new_state_dict={k.replace('model.',''): v for k, v in state_dict.items()}
            self.model.load_state_dict(new_state_dict)
            self.model.eval() # ÇOK KRİTİK: Modeli tahmin moduna aldık (Dropout kapandı)
            
            # Scaler'ları yükle
            self.x_scaler = joblib.load(x_scaler_yolu)
            self.y_scaler = joblib.load(y_scaler_yolu)
            
            self.hazir_mi = True
            print("✅ Model ve Scaler'lar başarıyla yüklendi!")
        except Exception as e:
            self.hazir_mi = False
            print(f"❌ Yükleme hatası: {e}")
    
    def analiz_et(self, df):
        # 1. GÜVENLİK KONTROLÜ
        if df is None or df.empty or not self.hazir_mi:
            return {
                "suanki_fiyat": 0.0, # Test kodunun çökmemesi için ekledik
                "tahmin": 0.0, 
                "yön": "HATA", 
                "güven": 0
            }
        
        # --- KRİTİK DÜZELTME: Sütun karmaşasını bitiriyoruz ---
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # 2. VERİ HAZIRLAMA (Eğitimdekiyle birebir aynı indikatörler olmalı!)
        df = df.sort_index(ascending=True)
        
        # Temel teknik indikatörlerin hesaplanması
        df['Getiri'] = df['Close'].pct_change()
        df['Hacim_degisimi'] = df['Volume'].pct_change()
        
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).ewm(com=13, adjust=False).mean()
        lose = (-delta.where(delta < 0, 0)).ewm(com=13, adjust=False).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / lose)))
        
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['STD_20'] = df['Close'].rolling(window=20).std()
        df['Bollinger_Upper'] = df['SMA_20'] + (df['STD_20'] * 2)
        df['Bollinger_Lower'] = df['SMA_20'] - (df['STD_20'] * 2)
        df['Bollinger_Konum'] = (df['Close'] - df['Bollinger_Lower']) / (df['Bollinger_Upper'] - df['Bollinger_Lower'])
        df['Momentum'] = df['Close'] / df['Close'].shift(10)

        # 3. VERİYİ MODELE UYGUN HALE GETİRME
        features = ['Close', 'RSI', 'MACD', 'Bollinger_Konum', 'Hacim_degisimi', 'Momentum']
        df_clean = df.dropna(subset=features)
        
        if len(df_clean) < 1:
            return {"yön": "YETERSİZ VERİ", "tahmin": 0, "güven": 0}

        # Sadece en son günün verisini (bugünü) tahmin için ayırıyoruz
        son_gün_verisi = df_clean[features].iloc[[-1]].values

        # --- ÇOK KRİTİK: Kendi eğittiğin ölçekleyicileri (Scaler) kullanıyoruz ---
        son_gün_scaled = self.x_scaler.transform(son_gün_verisi)
        son_gün_tensor = torch.tensor(son_gün_scaled, dtype=torch.float32)

        # 4. TAHMİN (INFERENCE)
        with torch.no_grad(): # Türev hesaplamayı kapat, hızlansın
            cikti = self.model(son_gün_tensor)
            
        # 0-1 arasındaki sonucu tekrar TL bazında fiyata çeviriyoruz
        tahmin_fiyat = self.y_scaler.inverse_transform(cikti.numpy())[0][0]
        suanki_fiyat = float(df_clean['Close'].iloc[-1])

        # 5. SONUÇLARI ANALİZ ETME
        yon = "YÜKSELİŞ" if tahmin_fiyat > suanki_fiyat else "DÜŞÜŞ"
        
        # Basit bir güven skoru (Tahmin ne kadar uzaktaysa AI o kadar 'emin' demektir)
        fark_orani = abs(tahmin_fiyat - suanki_fiyat) / suanki_fiyat
        guven_skoru = min(99, int(60 + (fark_orani * 1000)))

        # NaN Koruması
        if math.isnan(tahmin_fiyat):
            return {"suanki_fiyat": suanki_fiyat, "tahmin": 0, "yön": "BELİRSİZ", "güven": 0}

        return {
            "suanki_fiyat": round(suanki_fiyat, 2),
            "tahmin": round(float(tahmin_fiyat), 2),
            "yön": yon,
            "güven": guven_skoru
        }