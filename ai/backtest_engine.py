import torch
import numpy as np
import pandas as pd
import yfinance as yf
import os
from ai.pythorc import deeplearning

# BIST 30 Listesi (Güncel)
BIST30_LISTESI = [
    "AKBNK.IS", "ALARK.IS", "ARCLK.IS", "ASELS.IS", "ASTOR.IS", "BIMAS.IS", 
    "BRSAN.IS", "CCOLA.IS", "EKGYO.IS", "ENKAI.IS", "EREGL.IS", "FROTO.IS", 
    "GARAN.IS", "GUBRF.IS", "HEKTS.IS", "ISCTR.IS", "KCHOL.IS", "KONTR.IS", 
    "KRDMD.IS", "OYAKC.IS", "PETKM.IS", "PGSUS.IS", "SAHOL.IS", "SASA.IS", 
    "SISE.IS", "TCELL.IS", "THYAO.IS", "TOASO.IS", "TUPRS.IS", "YKBNK.IS"
]

def tekli_hisse_test(hisse, ai, periyot, esik_deger):
    # --- SERT TEST AYARLARI ---
    komisyon_orani = 0.002 # %0.2 Komisyon
    kayma_orani = 0.001    # %0.1 Slippage (Kötü fiyattan alma)
    stop_loss_orani = 0.05 # %5 Zarar edince KAÇ!
    
    """Belirli bir hisse için simülasyonu çalıştırır ve getiri yüzdesini döner."""
    baslangic_kasa = 100000.0
    kasa = baslangic_kasa
    hisse_adedi = 0
    alim_fiyati = 0.0
    
    try:
        df = yf.download(hisse, period=periyot, interval="1h", progress=False, multi_level_index=False)
        if df.empty: return 0.0
        
        # Teknik Hesaplamalar (Senin kodundan alındı)
        df = df.copy()
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
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)

        test_baslangic = int(len(df) * 0.7)
        for i in range(test_baslangic, len(df) - 1):
            o_anki_fiyat = float(df['Close'].iloc[i])
            df_dilim = df.iloc[max(0, i - 50):i + 1].copy()
            
            sonuc = ai.analiz_et(df_dilim)
            tahmin_fiyat = float(sonuc.get('tahmin', 0))
            if tahmin_fiyat == 0: continue

            # İşlem mantığının içini şöyle güncelle:
            if tahmin_fiyat > o_anki_fiyat * (1 + esik_deger) and hisse_adedi == 0:
                # ALIRKEN: Komisyon ve kötü fiyat farkını öde
                gercek_alim_fiyat = o_anki_fiyat * (1 + kayma_orani)
                hisse_adedi = int(kasa // gercek_alim_fiyat)
                kasa -= (hisse_adedi * gercek_alim_fiyat) * (1 + komisyon_orani)
                alim_fiyati = gercek_alim_fiyat
                
            elif hisse_adedi > 0:
                # STOP LOSS KONTROLÜ (Acil Çıkış)
                if o_anki_fiyat < alim_fiyati * (1 - stop_loss_orani):
                    kasa += (hisse_adedi * o_anki_fiyat) * (1 - komisyon_orani)
                    hisse_adedi = 0
                    print(f"🚨 [STOP LOSS] Fiyat çakıldı, Ghost kaçtı! Zarar: %5")

                # NORMAL SATIŞ
                elif tahmin_fiyat < o_anki_fiyat * 0.995:
                    gercek_satis_fiyat = o_anki_fiyat * (1 - kayma_orani)
                    kasa += (hisse_adedi * gercek_satis_fiyat) * (1 - komisyon_orani)
                    hisse_adedi = 0

        final_deger = kasa + (hisse_adedi * float(df['Close'].iloc[-1]))
        return ((final_deger / baslangic_kasa) - 1) * 100
    except:
        return 0.0

def master_backtest():
    print("\n" + "🚀"*20)
    print(" GHOST OPERATOR: BIST 30 GENEL TAARRUZ ")
    print("🚀"*20)

    periyot = input("📅 Kaç günlük periyot taransın? (Örn: 300d): ").strip().lower()
    esik_deger = float(input("🚀 Yüzde kaç artış eşiği kullanılsın? (Örn: 2): ")) / 100

    ai = deeplearning()
    if not ai.hazir_mi:
        print("❌ Model yüklenemedi!")
        return

    rapor_listesi = []
    
    print(f"\n📡 {len(BIST30_LISTESI)} hisse için tarama başlıyor. Kahveni al Musab, bu biraz sürecek...\n")

    for hisse in BIST30_LISTESI:
        print(f"🔄 Analiz ediliyor: {hisse}...", end="\r")
        getiri = tekli_hisse_test(hisse, ai, periyot, esik_deger)
        rapor_listesi.append({"Hisse": hisse, "Getiri": getiri})

    # Sonuçları Tabloya Dök
    df_final = pd.DataFrame(rapor_listesi)
    df_final = df_final.sort_values(by="Getiri", ascending=False)

    print("\n\n" + "="*45)
    print(" 🏆 BIST 30 PERFORMANS SIRALAMASI ")
    print("="*45)
    
    for i, row in df_final.iterrows():
        emoji = "🔥" if row['Getiri'] > 15 else "📈" if row['Getiri'] > 0 else "📉"
        print(f"{emoji} {row['Hisse']:<10} | Getiri: %{row['Getiri']:>7.2f}")

    print("="*45)
    print(f"📊 BIST 30 Genel Ortalama Getiri: %{df_final['Getiri'].mean():.2f}")
    print(f"💎 En Başarılı Hisse: {df_final.iloc[0]['Hisse']} (%{df_final.iloc[0]['Getiri']:.2f})")
    print("="*45)

if __name__ == "__main__":
    master_backtest()