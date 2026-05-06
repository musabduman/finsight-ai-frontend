import streamlit as st

from main import tek_hisse_run
from ai.llm import Gemini, OllamaAgresif, OllamaChat
from ai.pythorc import deeplearning
from modules.watchlist import watchlist_sayfasi
from modules.mega_tarama import mega_tarama
from modules.bist30_tarama import bist30_tarama
from modules.haber_akisi import haber_akisi
from services.veri import get_stock_data
from services.config import get_api_keys
    
keys = get_api_keys()

kullanici_api_key = keys["gemini"]
ollama_api_key = keys["ollama"]

st.set_page_config(page_title="AI Borsa", layout="wide")

# --- BOTLAR ---
dl_bot = deeplearning()

gemini_bot = Gemini(api_key=kullanici_api_key)
ollama_bot = OllamaAgresif(api_key=ollama_api_key, model="gpt-oss:120b-cloud")

# --- INPUT ---
sembol_input = st.text_input("Hisse (THYAO, GARAN...)")

bist100_hisseler = [
    "AEFES.IS", "AKBNK.IS", "ASELS.IS", "BIMAS.IS",
    "EREGL.IS", "FROTO.IS", "GARAN.IS", "KCHOL.IS",
    "THYAO.IS", "TUPRS.IS", "YKBNK.IS"
]

secim = st.sidebar.radio(
    "Mod",
    ["Tek Hisse Analizi", "Mega Tarama", "BIST30 Tarama", "Haber Akışı"]
)

# --- TEK HİSSE ---
if secim == "Tek Hisse Analizi":
    if st.button("Analiz Et"):
        result = tek_hisse_run(
            sembol_input,
            dl_bot,
            gemini_bot,
            ollama_bot
        )

        if "error" in result:
            st.error(result["error"])
        else:
            st.subheader(result["symbol"])

            st.metric("Son Fiyat", result["son_fiyat"])
            st.metric("SBS", result["son_sbs"])

            st.write("### Gemini")
            st.markdown(result["gemini"])

            st.write("### Ollama")
            st.markdown(result["ollama"])

# --- MEGA TARAMA ---
elif secim == "Mega Tarama":
    mega_tarama(bist100_hisseler, dl_bot)

# --- BIST30 ---
elif secim == "BIST30 Tarama":
    bist30_tarama()

# --- HABER ---
elif secim == "Haber Akışı":
    haber_akisi()

elif secim == "Watchlist":
    watchlist_sayfasi()

@st.fragment
def chat_bolumu():
    st.markdown("### 💬 Asistan")
    
    # Geçmişi en tepede başlat
    if "chat_gecmisi" not in st.session_state:
        st.session_state.chat_gecmisi = []

    # 1. EKRANA MESAJLARI BAS
    chat_container = st.container(height=350)
    with chat_container:
        for msg in st.session_state.chat_gecmisi:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

    # 2. INPUT ALANI (En altta sabit durur)
    if prompt := st.chat_input("Borsa hakkında bir şey sor..."):
        # Kullanıcı mesajını ekle ve ekrana bas
        st.session_state.chat_gecmisi.append({"role": "user", "content": prompt})
        st.session_state.chat_gecmisi = st.session_state.chat_gecmisi[-10:]
        with chat_container:
            with st.chat_message("user"):
                st.write(prompt)

        # 3. CEVAP ÜRET
        with chat_container:
            with st.chat_message("assistant"):
                with st.spinner("Düşünüyorum..."):
                    # Bağlamları topla
                    aktif_baglam = st.session_state.get("aktif_analiz_baglami", "Aktif analiz yok.")
                    
                    try:
                        chat_bot = OllamaChat(api_key=ollama_api_key)
                        cevap = chat_bot.generate(st.session_state.chat_gecmisi, 
                                                  aktif_baglam=aktif_baglam,)
                        st.write(cevap)
                        st.session_state.chat_gecmisi.append({"role": "assistant", "content": cevap})
                    except Exception as e:
                        st.error(f"AI Hatası: {e}")
