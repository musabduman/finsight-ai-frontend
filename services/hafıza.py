import os
import uuid
import hashlib
import pandas as pd

from dotenv import load_dotenv
from datetime import datetime
from pinecone import Pinecone, ServerlessSpec

load_dotenv("api_keys.env")

# -------------------------
# AYARLAR
# -------------------------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "finsight-memory"
EMBEDDING_MODEL = "multilingual-e5-large"
EMBEDDING_DIM = 1024

pc = Pinecone(api_key=PINECONE_API_KEY)

# Index yoksa oluştur
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBEDDING_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(INDEX_NAME)


# -------------------------
# EMBEDDING
# -------------------------
def embed(texts, input_type="passage"):
    result = pc.inference.embed(
        model=EMBEDDING_MODEL,
        inputs=texts,
        parameters={
            "input_type": input_type,
            "truncate": "END"
        }
    )
    return [item["values"] for item in result]


# -------------------------
# TEKRAR KAYDI ENGELLEME
# -------------------------
def create_stable_id(text: str):
    return hashlib.md5(text.encode("utf-8")).hexdigest()


# -------------------------
# HAFIZAYA KAYDET
# -------------------------
def save_to_memory(new_data):
    try:
        if isinstance(new_data, dict):
            new_data = [new_data]

        if not new_data:
            return None

        texts = []
        prepared = []

        for item in new_data:
            hisse = item.get("hisse", "BİST")
            ozet = item.get("ozet", "")
            text = f"{hisse} hissesi hakkında gelişme: {ozet}"

            texts.append(text)
            prepared.append((item, text))

        vectors = embed(texts, input_type="passage")

        vectors_to_upsert = []

        for (item, text), vector in zip(prepared, vectors):
            stable_id = create_stable_id(text)

            vectors_to_upsert.append((
                stable_id,
                vector,
                {
                    "hisse": item.get("hisse", "BİST"),
                    "ozet": item.get("ozet", ""),
                    "duygu": item.get("duygu", "Nötr"),
                    "kaynak": item.get("kaynak", ""),
                    "link": item.get("link", ""),
                    "tarih": item.get(
                        "tarih",
                        datetime.now().strftime("%Y-%m-%d %H:%M")
                    ),
                    "text": text
                }
            ))

        if vectors_to_upsert:
            index.upsert(vectors=vectors_to_upsert)

        return pd.DataFrame(new_data)

    except Exception as e:
        print(f"Hafızaya kaydetme hatası: {e}")
        return None


# -------------------------
# HAFIZADAN OKU
# -------------------------
def get_memory_for_llm(query: str, limit: int = 5, hisse_filtresi: str = None):
    try:
        query_vector = embed([query], input_type="query")[0]

        filtre = None
        if hisse_filtresi:
            filtre = {
                "hisse": {
                    "$eq": hisse_filtresi
                }
            }

        results = index.query(
            vector=query_vector,
            top_k=limit,
            include_metadata=True,
            filter=filtre
        )

        matches = results.get("matches", [])

        if not matches:
            return "İlgili geçmiş veri bulunamadı."

        metin = f"### '{query}' ile ilgili hafıza\n\n"

        for match in matches:
            meta = match["metadata"]
            skor = round(match["score"], 2)

            metin += (
                f"**{meta.get('hisse', '')}** | "
                f"{meta.get('tarih', '')} | "
                f"Skor: {skor}\n\n"
            )

            metin += f"{meta.get('ozet', '')}\n\n"

            link = meta.get("link", "")
            if link:
                metin += f"[Habere git]({link})\n\n"

            metin += "---\n"

        return metin

    except Exception as e:
        print(f"Hafıza okuma hatası: {e}")
        return "Hafızaya ulaşılamadı."