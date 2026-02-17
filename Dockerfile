# 1. Temel olarak hafif bir Python sürümü seçiyoruz
FROM python:3.10-slim

# 2. Konteyner içindeki çalışma klasörümüzü belirliyoruz
WORKDIR /app

# 3. Kütüphane listemizi kopyalıyoruz
COPY requirements.txt .

# 4. Gerekli kütüphaneleri kuruyoruz
RUN pip install --no-cache-dir -r requirements.txt

# 5. Projedeki tüm dosyalarımızı (kodlarımızı) kopyalıyoruz
COPY . .

# 6. Projenin nasıl çalıştırılacağını belirliyoruz
# EĞER ANA DOSYANIN ADI main.py DEĞİLSE BURAYI DEĞİŞTİR
CMD ["python", "hisse_bilgi_özel.py"]