import ssl
import feedparser
from datetime import datetime

# SSL sertifika hatalarını aşmak için
if hasattr(ssl, '_create_unverified_context'):
    ssl._create_default_https_context = ssl._create_unverified_context

class StockNewsFetcher:
    def __init__(self):
        # İstediğin kadar yeni kaynak ekleyebilirsin!
        self.rss_kaynaklari = {
            "Investing BİST": "https://tr.investing.com/rss/news_301.rss",
            "Investing Şirketler": "https://tr.investing.com/rss/news_437.rss",
            "Investing Son Dakika": "https://tr.investing.com/rss/news_285.rss",
            "Investing Ekonomi": "https://tr.investing.com/rss/news_14.rss"
        }

    def get_news(self):
        toplanan_haberler = []
        
        for kaynak_adi, rss_url in self.rss_kaynaklari.items():
            print(f"📡 {kaynak_adi} kaynağından haberler çekiliyor...")
            
            try:
                feed = feedparser.parse(rss_url)
                
                if not feed.entries:
                    print(f"  ❌ {kaynak_adi} boş döndü veya ulaşılamadı.")
                    continue

                # Her kaynaktan en yeni 5 haberi alalım (sayıyı artırabilirsin)
                for entry in feed.entries[:5]: 
                    baslik = entry.title
                    ozet = entry.description if hasattr(entry, 'description') else "Özet bulunamadı"
                    
                    # Arayüzde tıklanabilir yapmak için linki de alıyoruz
                    haber_linki = entry.link if hasattr(entry, 'link') else "#" 
                    
                    toplanan_haberler.append({
                        "hisse": "BİST Genel", # İleride metin içinden hisse adı ayıklayan bir NLP eklenebilir
                        "ozet": f"{baslik} - {ozet}",
                        "duygu": "Nötr", 
                        "kaynak": kaynak_adi, # YENİ: Hangi siteden çekildi?
                        "link": haber_linki,  # YENİ: Haberin orijinal linki
                        "tarih": datetime.now().strftime("%Y-%m-%d %H:%M")
                    })
            except Exception as e:
                print(f"  ⚠️ {kaynak_adi} çekilirken hata oluştu: {e}")
            
        print(f"✅ Toplam {len(toplanan_haberler)} adet haber başarıyla toplandı.")
        return toplanan_haberler

# Sadece bu dosyayı test etmek için:
if __name__ == "__main__":
    fetcher = StockNewsFetcher()
    haberler = fetcher.get_news()
    for h in haberler[:3]: # Sadece ilk 3'ünü ekrana yazdırarak görelim
        print(f"\n[{h['kaynak']}] {h['ozet'][:100]}...\nLink: {h['link']}")