import ssl
import feedparser

from datetime import datetime

# SSL bazı makinelerde RSS'i bozuyor
if hasattr(ssl, "_create_unverified_context"):
    ssl._create_default_https_context = ssl._create_unverified_context


class StockNewsFetcher:
    def __init__(self):
        self.rss_kaynaklari = {
            "Investing BİST": "https://tr.investing.com/rss/news_301.rss",
            "Investing Şirketler": "https://tr.investing.com/rss/news_437.rss",
            "Investing Son Dakika": "https://tr.investing.com/rss/news_285.rss",
            "Investing Ekonomi": "https://tr.investing.com/rss/news_14.rss"
        }

    def get_news(self, limit_per_source=5):
        haberler = []

        for kaynak_adi, rss_url in self.rss_kaynaklari.items():
            try:
                feed = feedparser.parse(rss_url)

                if not feed.entries:
                    continue

                for entry in feed.entries[:limit_per_source]:
                    baslik = entry.get("title", "")
                    ozet = entry.get("description", baslik)
                    link = entry.get("link", "")

                    haberler.append({
                        "hisse": "BİST Genel",
                        "ozet": f"{baslik} - {ozet}",
                        "duygu": "Nötr",
                        "kaynak": kaynak_adi,
                        "link": link,
                        "tarih": datetime.now().strftime("%Y-%m-%d %H:%M")
                    })

            except Exception as e:
                print(f"{kaynak_adi} haber çekme hatası: {e}")

        return haberler


def anlik_hisse_haberi_cek(sembol: str, limit: int = 5) -> str:
    """Belirli bir hisse için anlık haber çeker, metin olarak döner."""
    fetcher = StockNewsFetcher()
    haberler = fetcher.get_news(limit_per_source=limit)

    temiz = sembol.replace(".IS", "").upper()
    ilgili = [h for h in haberler if temiz in h.get("ozet", "").upper()]
    kaynak = ilgili if ilgili else haberler[:limit]

    if not kaynak:
        return "Haber bulunamadı."

    return "\n".join(
        f"[{h.get('tarih','')}] {h.get('ozet','')}" for h in kaynak
    )