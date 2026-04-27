import pandas as pd
import os
from datetime import datetime

HISTORY_FILE = "analiz_arsivi.csv"

def save_to_memory(new_data):
    """n8n'den gelen veriyi alır ve arşive ekler."""
    try:
        # n8n bazen tek bir dict bazen liste döndürebilir, garantiye alalım
        if isinstance(new_data, dict):
            new_data = [new_data]
            
        new_df = pd.DataFrame(new_data)
        
        # Gerekli kolonlar yoksa hata vermesin diye kontrol
        required_cols = ['hisse', 'duygu', 'ozet']
        for col in required_cols:
            if col not in new_df.columns:
                new_df[col] = "Bilinmiyor"

        new_df['analiz_tarihi'] = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        if os.path.exists(HISTORY_FILE):
            try:
                # utf-8-sig daha güvenlidir
                old_df = pd.read_csv(HISTORY_FILE, encoding='utf-8-sig')
                final_df = pd.concat([new_df, old_df]).drop_duplicates(subset=['hisse', 'ozet']).reset_index(drop=True)
            except Exception:
                final_df = new_df # Eski dosya bozuksa yenisiyle devam et
        else:
            final_df = new_df
            
        # Encoding'i utf-8-sig yaptık
        final_df.to_csv(HISTORY_FILE, index=False, encoding='utf-8-sig')
        return final_df
    except Exception as e:
        print(f"Hafızaya kaydetme hatası: {e}")
        return None

def load_memory():
    """Arşivi okur ve DataFrame döner."""
    if os.path.exists(HISTORY_FILE):
        try:
            return pd.read_csv(HISTORY_FILE, encoding='utf-8-sig')
        except:
            return None
    return None

def get_memory_for_llm(limit=5):
    """LLM'e bağlam (context) olarak vermek için veriyi metne çevirir."""
    df = load_memory()
    if df is not None and not df.empty:
        # En son 'limit' kadar haberi al (head en eskileri, tail en yenileri verir. Genelde tail istenir)
        recent_data = df.tail(limit) 
        context_text = "--- GEÇMİŞ HABER VE ANALİZ HAFIZASI ---\n"
        for _, row in recent_data.iterrows():
            context_text += f"[{row.get('analiz_tarihi', '')}] {row.get('hisse', '')}: {row.get('ozet', '')} (Duygu: {row.get('duygu', '')})\n"
        return context_text
    return "Henüz geçmiş veri hafızası yok."
