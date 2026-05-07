import streamlit as st

from main import tek_hisse_run
from ai.llm import Gemini, OllamaAgresif, OllamaChat
from ai.pythorc import deeplearning
from modules.watchlist import watchlist_sayfasi
from modules.mega_tarama import mega_tarama
from modules.bist30_tarama import bist30_tarama
from modules.haber_akisi import haber_akisi
from services.veri import get_stock_data, get_temel_hesapla
from services.hafıza import get_memory_for_llm
from services.haber import anlik_hisse_haberi_cek
from services.config import get_api_keys
from indicators.technical import teknik_analiz
from auth_ui import login_sidebar

st.set_page_config(page_title="AI Borsa", layout="wide")

# --- GİRİŞ KONTROLÜ ---
giris_yapildi = login_sidebar()

if not giris_yapildi:
    st.info("Devam etmek için lütfen giriş yapın.")
    st.stop()

# --- BURADAN SONRASI SADECE GİRİŞ YAPILMIŞSA ÇALIŞIR ---
keys = get_api_keys()

kullanici_api_key = keys["gemini"]
ollama_api_key = keys["ollama"]

if not kullanici_api_key:
    st.warning("⚠️ Gemini API key eksik. Lütfen hesabınıza kayıtlı bir API key ekleyin.")
    st.stop()

# --- BOTLAR (login sonrası, key varsa init edilir) ---
dl_bot = deeplearning()
gemini_bot = Gemini(api_key=kullanici_api_key)
ollama_bot = OllamaAgresif(api_key=ollama_api_key, model="gpt-oss:120b-cloud")

bist100_hisseler = [
    "AKBNK.IS", "GARAN.IS", "HALKB.IS", "ISCTR.IS", "SKBNK.IS","ASTOR.IS", "BERA.IS", "ENJSA.IS", "GESAN.IS", "GENIL.IS","RODRG.IS"
    "TSKB.IS", "VAKBN.IS", "YKBNK.IS", "QNBFB.IS","KCHOL.IS", "SAHOL.IS", "DOHOL.IS", "EKGYO.IS", "ENKAI.IS","ULUSE.IS", "UYUM.IS"
    "TTKOM.IS", "TCELL.IS","FROTO.IS", "TOASO.IS", "OTKAR.IS","TUPRS.IS", "PETKM.IS", "AGHOL.IS", "AKSEN.IS", "ZOREN.IS","SNPAM.IS",
    "KONTR.IS", "ODAS.IS","EREGL.IS", "KRDMD.IS", "OYAKC.IS", "CEMTS.IS","AEFES.IS", "CCOLA.IS", "ULKER.IS", "PNSUT.IS", "BANVT.IS",
    "TATGD.IS","BIMAS.IS", "MGROS.IS", "SOKM.IS","AKCNS.IS", "BOLUC.IS", "CIMSA.IS", "NUHCM.IS", "ADANA.IS","SISE.IS","TAVHL.IS", "TKFEN.IS",
    "ASELS.IS", "HAVAS.IS", "LOGO.IS", "INDES.IS","THYAO.IS", "PGSUS.IS", "CLEBI.IS", "MAVI.IS","GUBRF.IS", "ECILC.IS", "DEVA.IS", "SELEC.IS",
    "SASA.IS", "ALARK.IS", "ARCLK.IS", "VESTL.IS", "BRSAN.IS","HEKTS.IS", "KERVT.IS","AGESA.IS", "ANHYT.IS", "RAYSG.IS","TMSN.IS", "TURSG.IS",
    "ISGYO.IS", "TRGYO.IS", "VKGYO.IS","KORDS.IS", "KOZAA.IS", "KOZAL.IS", "LUKSK.IS", "MPARK.IS","NETAS.IS", "PRKAB.IS", "QUAGR.IS", "REEDR.IS",  
]

secim = st.sidebar.radio(
    "Mod",
    ["İzleme Listesi", "Tek Hisse Analizi", "Mega Tarama", "BIST30 Tarama", "Haber Akışı"]
)

if secim == "İzleme Listesi":
    watchlist_sayfasi(get_stock_data=get_stock_data, teknik_analiz=teknik_analiz)

elif secim == "Tek Hisse Analizi":
    sembol_input = st.text_input("Hisse (THYAO, GARAN...)")

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

elif secim == "Mega Tarama":
    mega_tarama(bist100_hisseler, dl_bot)

elif secim == "BIST30 Tarama":
    bist30_tarama(
        get_stock_data=get_stock_data,
        teknik_analiz=teknik_analiz,
        dl_bot=dl_bot,
        Gemini=Gemini,
        OllamaAgresif=OllamaAgresif,
        get_temel_hesapla=get_temel_hesapla,
        anlik_hisse_haberi_cek=anlik_hisse_haberi_cek,
        get_memory_for_llm=get_memory_for_llm,
        gemini_api=kullanici_api_key,
        ollama_api=ollama_api_key,
    )

elif secim == "Haber Akışı":
    haber_akisi()


# --- CHAT (tanımdan sonra çağrılıyor) ---
@st.fragment
def chat_bolumu():
    st.markdown("### 💬 Asistan")

    if "chat_gecmisi" not in st.session_state:
        st.session_state.chat_gecmisi = []

    chat_container = st.container(height=350)
    with chat_container:
        for msg in st.session_state.chat_gecmisi:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

    if prompt := st.chat_input("Borsa hakkında bir şey sor..."):
        st.session_state.chat_gecmisi.append({"role": "user", "content": prompt})
        st.session_state.chat_gecmisi = st.session_state.chat_gecmisi[-10:]
        with chat_container:
            with st.chat_message("user"):
                st.write(prompt)

        with chat_container:
            with st.chat_message("assistant"):
                with st.spinner("Düşünüyorum..."):
                    aktif_baglam = st.session_state.get("aktif_analiz_baglami", "Aktif analiz yok.")
                    try:
                        chat_bot = OllamaChat(api_key=ollama_api_key)
                        cevap = chat_bot.generate(st.session_state.chat_gecmisi,
                                                  aktif_baglam=aktif_baglam)
                        st.write(cevap)
                        st.session_state.chat_gecmisi.append({"role": "assistant", "content": cevap})
                    except Exception as e:
                        st.error(f"AI Hatası: {e}")

chat_bolumu()