import torch
import warnings
import yfinance as yf
import joblib
import numpy as np
import torch.nn as nn
import pandas as pd
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")
bist_30=["AKBNK.IS", "ALARK.IS", "ARCLK.IS", "ASELS.IS", "ASTOR.IS", 
    "BIMAS.IS", "BRSAN.IS", "CCOLA.IS", "EKGYO.IS", "ENKAI.IS", 
    "EREGL.IS", "FROTO.IS", "GARAN.IS", "GUBRF.IS", "HEKTS.IS", 
    "ISCTR.IS", "KCHOL.IS", "KONTR.IS", "KRDMD.IS", "OYAKC.IS", 
    "PETKM.IS", "PGSUS.IS", "SAHOL.IS", "SASA.IS", "SISE.IS", 
    "TCELL.IS", "THYAO.IS", "TOASO.IS", "TUPRS.IS", "YKBNK.IS"]

# PyTorch'un veri setlerini anlaması için bir köprü (Dataset) kuruyoruz
class BorsaDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.x)
        
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
class deeplearning(nn.Module):
    def __init__(self):
        super(deeplearning, self).__init__()
        self.model=nn.Sequential(
            nn.Linear(6,16),                    
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16,1)
            )
    def forward(self, x):
        return self.model(x)
    
    # 2. VERİ HAZIRLAMA (Cevap Anahtarını Oluşturma)
    @staticmethod
    def verileri_hazirla(symbol_listesi):
        tum_x=[]
        tum_y=[]
        
        for symbol in symbol_listesi:
            print(f"{symbol} verileri çekiliyor ve teknik altyapı hazırlanıyor...")
        
            try:
                # Modelin bolca pratik yapması için 5 yıllık geniş bir veri seti çekiyoruz
                df = yf.download(symbol, period="730d",interval="1h", progress=False, multi_level_index=False)
                
                # Senin belirlediğin temel teknik analiz metrikleri:
                df['Getiri'] = df['Close'].pct_change()
                df['Hacim_degisimi'] = df['Volume'].pct_change()
                
                # MACD Hesaplaması
                exp1 = df['Close'].ewm(span=12, adjust=False).mean()
                exp2 = df['Close'].ewm(span=26, adjust=False).mean()
                df['MACD'] = exp1 - exp2
                
                # RSI Hesaplaması
                delta = df['Close'].diff()
                gain = (delta.where(delta > 0, 0)).ewm(com=13, adjust=False).mean()
                lose = (-delta.where(delta < 0, 0)).ewm(com=13, adjust=False).mean()
                df['RSI'] = 100 - (100 / (1 + (gain / lose)))
                
                # Bollinger Bantları ve Konumlandırma
                df['SMA_20'] = df['Close'].rolling(window=20).mean()
                df['STD_20'] = df['Close'].rolling(window=20).std()
                df['Bollinger_Upper'] = df['SMA_20'] + (df['STD_20'] * 2)
                df['Bollinger_Lower'] = df['SMA_20'] - (df['STD_20'] * 2)
                
                # Fiyatın bantların neresinde olduğunu (0 ile 1 arası) buluyoruz
                df['Bollinger_Konum'] = (df['Close'] - df['Bollinger_Lower']) / (df['Bollinger_Upper'] - df['Bollinger_Lower'])
                
                # Momentum
                df['Momentum'] = df['Close'] / df['Close'].shift(10)
                
                # İŞTE KİLİT NOKTA (Cevap Anahtarı): Modelin tahmin etmeye çalışacağı şey "yarının kapanış fiyatı"
                df['Target'] = df['Close'].shift(-1) 
                
                # Bozuk (Sonsuz veya NaN) verileri temizle
                df.replace([np.inf, -np.inf], np.nan, inplace=True)
                
                # Sadece modele girecek 6 özelliği (feature) ve hedefi (Target) seçiyoruz
                features = ['Close', 'RSI', 'MACD', 'Bollinger_Konum', 'Hacim_degisimi', 'Momentum']
                df_clean = df.dropna(subset=features + ['Target'])
                
                # x_data: Modelin bakacağı şeyler, y_data: Modelin bulmaya çalışacağı sonuç
                # Temizlenmiş verileri listeye ekle
                tum_x.append(df_clean[features].values)
                tum_y.append(df_clean[['Target']].values)
        
            except Exception as e:
                print("Veri çekme hatası: {e}")
        x_dev_matris = np.vstack(tum_x)
        y_dev_matris = np.vstack(tum_y)

        return x_dev_matris, y_dev_matris 

if __name__=="__main__":
    x_raw, y_raw = deeplearning.verileri_hazirla(bist_30)

    x_sacler = MinMaxScaler()
    y_sacler = MinMaxScaler()
    
    x_scaled = x_sacler.fit_transform(x_raw)
    y_scaled = y_sacler.fit_transform(y_raw)

    dataset = BorsaDataset(x_scaled,y_scaled)

    train_size=int(0.8*len(dataset))
    
    train_dataset = torch.utils.data.Subset(dataset, range(0, train_size))
    val_dataset = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))

    val_size=len(dataset)-train_size

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    model = deeplearning()
    hata = nn.MSELoss()

    optime=optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)

    epochs = 50
    for i in range(epochs):
        model.train()
        toplam_train_loss=0.0

        for batch_x, batch_y in train_loader:
            optime.zero_grad()
            tahmin = model(batch_x)
            
            loss=hata(tahmin, batch_y)
            loss.backward()
            optime.step()
            toplam_train_loss += loss.item()
        
        model.eval()
        toplam_val_loss=0.0
        
        with torch.no_grad():
            for v_batch_x, v_batch_y in val_loader:
                v_tahmin = model(v_batch_x)
                v_loss = hata(v_tahmin, v_batch_y)
                toplam_val_loss += v_loss.item()
        
        avg_train_loss = toplam_train_loss / len(train_loader)
        avg_val_loss = toplam_val_loss / len(val_loader)

        if i%10==0:
            print(f"Epoch {i}/{epochs} | Eğitim Kaybı: {avg_train_loss:.6f} | Sınav Kaybı: {avg_val_loss:.6f}")
    torch.save(model.state_dict(), "kahin_model.pth")

    # 2. Ölçekleyicileri Kaydet (Web sitesinde veri işlerken lazım olacak)
    joblib.dump(x_sacler, "x_scaler.gz")
    joblib.dump(y_sacler, "y_scaler.gz")

    print("---")
    print("✅ İşlem Başarılı!")
    print("1. 'kahin_model.pth' (Modelin Beyni)")
    print("2. 'x_scaler.gz' (Girdi Sözlüğü)")
    print("3. 'y_scaler.gz' (Çıktı Sözlüğü)")
    print("Dosyalar klasöre kazındı. Artık web tarafına geçmeye hazırız.")
