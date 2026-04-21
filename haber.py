import yfinance as yf
import pandas as pd
import re

class StockNewsFetcher:
    def __init__(self):
        pass
    
    def fetch_news(self, symbol, days_back=None):
        """Belirli bir sembol için haberleri çeker"""
        if not symbol.endswith(".IS"):
            symbol = symbol + ".IS"
        
        try:
            ticker = yf.Ticker(symbol)
            news_data = ticker.news
            
            parsed_articles = []
            for article in news_data:
                title = article.get('title', '').strip()
                publisher = article.get('publisher', '').strip()
                link = article.get('link', '')
                publish_time = article.get('providerPublishTime', '')
                
                if title:
                    cleaned_title = self._clean_text(title)
                    parsed_articles.append({
                        'title': cleaned_title,
                        'publisher': publisher,
                        'link': link,
                        'providerPublishTime': publish_time
                    })
            return parsed_articles
        except Exception as e:
            print(f"Error fetching news for {symbol}: {e}")
            return []
    
    def _clean_text(self, text):
        """Metin temizleme işlemi yapar"""
        # HTML tag'lerini kaldır
        clean = re.sub(r'<.*?>', '', text)
        # Fazladan boşlukları temizle
        clean = re.sub(r'\s+', ' ', clean).strip()
        return clean