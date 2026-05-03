import pandas as pd
from datetime import datetime
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import uuid
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()
# --- 1. AYARLAR VE BAŞLATMA ---
# Ücretsiz bir Pinecone hesabı açıp API anahtarını buraya yazacaksın
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "finsight-memory"

if not PINECONE_API_KEY:
    print("⚠️ Pinecone API Anahtarı bulunamadı! Hafıza devre dışı.")
    pc = None
    index = None
    model = None
else:
    # Anahtar varsa normal şekilde bağlan / Pinecone bağlantısı
    pc = Pinecone(api_key=PINECONE_API_KEY)
    # Index (Veritabanı Tablosu) yoksa bulutta oluştur
    if INDEX_NAME not in pc.list_indexes().names():
        print("Bulutta yeni bir vektör hafızası oluşturuluyor...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=384, # Seçtiğimiz embedding modelinin vektör boyutu
            metric="cosine", # Anlamsal benzerlik ölçümü
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

index = pc.Index(INDEX_NAME)

# Metinleri vektöre çevirecek yerel model (PyTorch tabanlıdır, bilgisayarında çalışır)
print("Embedding modeli yükleniyor...")
# Modeli sadece gerektiğinde RAM'e alacak akıllı fonksiyon
@st.cache_resource
def get_embedding_model():
    from sentence_transformers import SentenceTransformer
    print("Embedding modeli RAM'e yükleniyor...")
    return SentenceTransformer('all-MiniLM-L6-v2')

# --- 2. FONKSİYONLAR ---
def save_to_memory(new_data):
    try:
        if isinstance(new_data, dict):
            new_data = [new_data]
            
        vectors_to_upsert = []
        
        for item in new_data:
            hisse = item.get('hisse', 'Bilinmiyor')
            ozet = item.get('ozet', 'Bilinmiyor')
            duygu = item.get('duygu', 'Bilinmiyor')
            tarih = datetime.now().strftime("%Y-%m-%d %H:%M")
            
            # Yapay zekanın anlayacağı ana metni oluştur
            text_to_embed = f"{hisse} hissesi hakkında gelişme: {ozet}"
            
            # Metni vektöre (sayısal dizilere) çevir
            vector = get_embedding_model().encode(text_to_embed).tolist()
            
            # Her habere benzersiz bir ID ver
            doc_id = str(uuid.uuid4())
            
            # Buluta gönderilecek format: (ID, Vektör, Metadata/Ek Bilgiler)
            vectors_to_upsert.append((
                doc_id, 
                vector, 
                {"hisse": hisse, "ozet": ozet, "duygu": duygu, "tarih": tarih, "text": text_to_embed}
            ))
            
        # Vektörleri Pinecone'a fırlat (Upsert işlemi)
        if vectors_to_upsert:
            index.upsert(vectors=vectors_to_upsert)
            print(f"{len(vectors_to_upsert)} adet yeni analiz bulut hafızasına eklendi.")
            
        return pd.DataFrame(new_data) # Akışı bozmamak için dataframe dönüyoruz
        
    except Exception as e:
        print(f"Bulut hafızaya kaydetme hatası: {e}")
        return None

def get_memory_for_llm(query, limit=5,hisse_filtresi=None):
    """LLM'e bağlam vermek için, sorulan soruyla EN ALAKALI geçmiş verileri getirir."""
    try:
        # 1. Önce kullanıcının veya LLM'in sorusunu vektöre çeviriyoruz
        query_vector = get_embedding_model().encode(query)
        
        filtre = None
        if hisse_filtresi:
            filtre = {"hisse": {"$eq": hisse_filtresi}}

        # 2. Pinecone'da bu vektöre en çok benzeyen haberleri arıyoruz (İşte RAG budur!)
        search_results = index.query(
            vector=query_vector,
            top_k=limit,
            include_metadata=True,
            filtre=filtre
        )
        
        matches = search_results.get('matches', [])
        
        if not matches:
            return "İlgili geçmiş veri bulunamadı."
            
        context_text = f"--- '{query}' İLE İLGİLİ GEÇMİŞ HAFIZA ---\n"
        
        for match in matches:
            meta = match['metadata']
            skor = round(match['score'], 2)
            kaynak = meta.get('kaynak', 'Bilinmeyen Kaynak')
            link = meta.get('link', '#')
            
            # LLM'in okuyacağı format (Kaynak bilgisiyle birlikte)
            context_text += f"Tarih: {meta['tarih']} | Hisse: {meta['hisse']} | Kaynak: {kaynak}\n"
            context_text += f"Haber: {meta['ozet']}\n"
            context_text += f"Orijinal Link: {link}\n"
            context_text += f"(Alaka Skoru: {skor})\n"
            context_text += "-" * 30 + "\n"
            
        return context_text
    
    except Exception as e:
        print(f"Hafıza okuma hatası: {e}")
        return "Hafızaya ulaşılamadı."
    
@st.cache_data(ttl=1800) 
def anlik_hisse_haberi_cek(sembol):
    print(f"🌐 Sadece {sembol} için güncel web taraması yapılıyor...") 
    
    arama_url = f"https://news.google.com/rss/search?q={sembol}+hisse+KAP+BİST&hl=tr&gl=TR&ceid=TR:tr"
