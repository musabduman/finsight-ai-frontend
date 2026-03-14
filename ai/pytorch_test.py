from pythorc import deeplearning # Kendi yazdığın classı çekiyoruz
import yfinance as yf

# 1. Modelimizi ayağa kaldıralım
ai = deeplearning()

# 2. Test için rastgele bir hisse seçelim (Meselen THYAO)
print("Canlı veri çekiliyor...")
test_df = yf.download("PGSUS.IS", period="1mo", interval="1h")

# 3. Analiz butonuna basalım
print("Tahmin yapılıyor...")
sonuc = ai.analiz_et(test_df)

# 4. Bakalım ne diyor?
print("-" * 30)
print(f"Hisse: THYAO")
print(f"Şu Anki Fiyat: {sonuc['suanki_fiyat']} TL")
print(f"Yapay Zeka Tahmini: {sonuc['tahmin']} TL")
print(f"Beklenen Yön: {sonuc['yön']}")
print(f"Güven Skoru: %{sonuc['güven']}")
print("-" * 30)