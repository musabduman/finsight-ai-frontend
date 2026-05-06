from services.haber import StockNewsFetcher 
from services.hafıza import save_to_memory 

def main():
    print("GitHub Actions: Haber çekme işlemi başlatılıyor...")
    
    # Sınıfı başlat
    fetcher = StockNewsFetcher()
    
    # Haberleri çek (Fonksiyon adını kendi koduna göre düzenle)
    yeni_haberler = fetcher.get_news() 
    
    if yeni_haberler and len(yeni_haberler) > 0:
        print(f"{len(yeni_haberler)} adet haber bulundu. Pinecone'a kaydediliyor...")
        # Hafızaya kaydet
        save_to_memory(yeni_haberler)
        print("İşlem başarıyla tamamlandı!")
    else:
        print("Yeni haber bulunamadı veya çekilemedi.")

# Dosya doğrudan çalıştırıldığında (GitHub Actions tarafından) bu blok tetiklenir
if __name__ == "__main__":
    main()