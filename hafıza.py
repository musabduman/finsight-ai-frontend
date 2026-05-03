import pandas as pd
import uuid
import streamlit as st
import os
import requests
import feedparser

from dotenv import load_dotenv
from datetime import datetime
from pinecone import Pinecone, ServerlessSpec

load_dotenv("api_keys.env")

# --- 1. AYARLAR VE BAŞLATMA ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "finsight-memory"
EMBEDDING_MODEL = "multilingual-e5-large"  # Türkçe destekli, 1024 boyut
EMBEDDING_DIM = 1024

# Pinecone bağlantısı
pc = Pinecone(api_key=PINECONE_API_KEY)

# Index yoksa oluştur
if INDEX_NAME not in pc.list_indexes().names():
    print("Bulutta yeni bir vektör hafızası oluşturuluyor...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBEDDING_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(INDEX_NAME)


# --- 2. EMBEDDING FONKSİYONU ---
def embed(texts: list[str], input_type: str = "passage") -> list[list[float]]:
    """Pinecone inference API ile metinleri vektöre çevirir. Torch gerekmez."""
    result = pc.inference.embed(
        model=EMBEDDING_MODEL,
        inputs=texts,
        parameters={"input_type": input_type, "truncate": "END"}
    )
    return [item["values"] for item in result]


# --- 3. HAFIZAYA KAYDET ---
def save_to_memory(new_data):
    try:
        if isinstance(new_data, dict):
            new_data = [new_data]

        texts = []
        for item in new_data:
            hisse = item.get("hisse", "Bilinmiyor")
            ozet = item.get("ozet", "Bilinmiyor")
            texts.append(f"{hisse} hissesi hakkında gelişme: {ozet}")

        # Tek API çağrısında tüm metinleri vektöre çevir
        vectors = embed(texts, input_type="passage")

        vectors_to_upsert = []
        for item, text, vector in zip(new_data, texts, vectors):
            vectors_to_upsert.append((
                str(uuid.uuid4()),
                vector,
                {
                    "hisse": item.get("hisse", "Bilinmiyor"),
                    "ozet": item.get("ozet", "Bilinmiyor"),
                    "duygu": item.get("duygu", "Bilinmiyor"),
                    "tarih": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "text": text
                }
            ))

        if vectors_to_upsert:
            index.upsert(vectors=vectors_to_upsert)
            print(f"{len(vectors_to_upsert)} adet yeni analiz bulut hafızasına eklendi.")

        return pd.DataFrame(new_data)

    except Exception as e:
        print(f"Bulut hafızaya kaydetme hatası: {e}")
        return None


# --- 4. HAFIZADAN OKU ---
def get_memory_for_llm(query: str, limit: int = 5, hisse_filtresi: str = None) -> str:
    """Soruyla en alakalı geçmiş verileri getirir (RAG)."""
    try:
        # Soruyu vektöre çevir (query tipinde embed et)
        query_vector = embed([query], input_type="query")[0]

        filtre = None
        if hisse_filtresi:
            filtre = {"hisse": {"$eq": hisse_filtresi}}

        search_results = index.query(
            vector=query_vector,
            top_k=limit,
            include_metadata=True,
            filter=filtre  # Türkçe değil, 'filter' olmalı
        )

        matches = search_results.get("matches", [])

        if not matches:
            return "İlgili geçmiş veri bulunamadı."

        context_text = f"--- '{query}' İLE İLGİLİ GEÇMİŞ HAFIZA ---\n"
        for match in matches:
            meta = match["metadata"]
            skor = round(match["score"], 2)
            context_text += f"Tarih: {meta['tarih']} | Hisse: {meta['hisse']} | Duygu: {meta['duygu']}\n"
            context_text += f"Haber: {meta['ozet']}\n"
            context_text += f"(Alaka Skoru: {skor})\n"
            context_text += "-" * 30 + "\n"

        return context_text

    except Exception as e:
        print(f"Hafıza okuma hatası: {e}")
        return "Hafızaya ulaşılamadı."


# --- 5. HABERLERİ ÇEK ---
@st.cache_data(ttl=1800)
def anlik_hisse_haberi_cek(sembol: str) -> list[dict]:
    """Google News RSS'ten belirtilen hisse için güncel haberleri çeker."""
    print(f"🌐 {sembol} için güncel web taraması yapılıyor...")

    arama_url = (
        f"https://news.google.com/rss/search"
        f"?q={sembol}+hisse+KAP+BİST&hl=tr&gl=TR&ceid=TR:tr"
    )

    try:
        feed = feedparser.parse(arama_url)
        haberler = []

        for entry in feed.entries[:10]:
            haberler.append({
                "baslik": entry.get("title", ""),
                "link": entry.get("link", ""),
                "tarih": entry.get("published", ""),
                "ozet": entry.get("summary", entry.get("title", ""))
            })

        return haberler

    except Exception as e:
        print(f"Haber çekme hatası: {e}")
        return []