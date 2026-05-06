import time
import streamlit as st

from veri import get_stock, normalize_symbol
from indicators.technical import teknik_analiz, get_temel_hesapla
from services.haber import anlik_hisse_haberi_cek
from services.hafıza import get_memory_for_llm

from ai.llm import Gemini, OllamaAgresif


def tek_hisse_analizi(sembol_input, gemini_key, ollama_key, dl_bot):

    if not sembol_input:
        st.warning("Lütfen bir hisse gir.")
        return

    if not gemini_key:
        st.error("Gemini API key gerekli.")
        return

    progress = st.progress(0, text="Sistem hazırlanıyor...")

    try:
        # --- VERİ ---
        clean_symbol = normalize_symbol(sembol_input)
        df = get_stock(clean_symbol)

        if df is None or df.empty:
            st.error("Veri çekilemedi. Biraz bekleyip tekrar dene.")
            return

        progress.progress(20, text="Teknik analiz yapılıyor...")

        df, fib_20, fib_200 = teknik_analiz(df)
        df = df.ffill().bfill()

        son_fiyat = float(df['Close'].iloc[-1])
        son_sbs = float(df['SBS'].iloc[-1])

        # --- AI MODEL ---
        progress.progress(40, text="PyTorch modeli çalışıyor...")
        sonuc_dl = dl_bot.analiz_et(df)

        ai_rapor = f"""
        Yön: {sonuc_dl.get('yön')}
        Hedef: {sonuc_dl.get('tahmin')} TL
        Güven: %{sonuc_dl.get('güven')}
        """

        # --- HABER + RAG ---
        progress.progress(60, text="Haberler çekiliyor...")

        anlik_haber = anlik_hisse_haberi_cek(clean_symbol)

        sorgu = f"{clean_symbol} hissesi güncel haberler"
        rag = get_memory_for_llm(query=sorgu, limit=5, hisse_filtresi=clean_symbol)

        haberler = f"""
        🔴 Güncel Haberler:
        {anlik_haber}

        📚 Hafıza:
        {rag}
        """

        # --- GEMINI ---
        progress.progress(75, text="Gemini analiz yapıyor...")

        gemini = Gemini(api_key=gemini_key)

        df_kisa = df.tail(30)
        temel = get_temel_hesapla(clean_symbol)

        analiz = gemini(
            clean_symbol,
            temel,
            df_kisa,
            haberler,
            ai_rapor,
            fib_200,
            son_sbs
        )

        # --- OLLAMA ---
        if ollama_key:
            progress.progress(90, text="Agresif analiz...")
            ollama = OllamaAgresif(api_key=ollama_key, model="gpt-oss:120b-cloud")
            agresif = ollama.generate(df_kisa, ai_rapor, fib_20, son_sbs)
        else:
            agresif = "Ollama aktif değil."

        progress.progress(100, text="Tamamlandı")
        time.sleep(0.5)
        progress.empty()

        # --- UI ---
        st.subheader(f"📊 {clean_symbol}")

        st.line_chart(df['Close'].tolist())

        c1, c2, c3, c4, c5 = st.columns(5)

        c1.metric("Fiyat", f"{son_fiyat:.2f}₺")

        yon = sonuc_dl.get("yön", "Nötr")
        emoji = "🔼" if yon == "YÜKSELİŞ" else "🔽"
        c2.metric("Tahmin", f"{sonuc_dl.get('tahmin',0):.2f}₺", f"{emoji} {yon}")

        c3.metric("RSI", f"{df['RSI'].iloc[-1]:.1f}")
        c4.metric("MACD", f"{df['MACD'].iloc[-1]:.2f}")

        sbs_text = "Pozitif" if son_sbs > 50 else "Negatif"
        c5.metric("SBS", f"%{son_sbs:.1f}", sbs_text)

        tab1, tab2, tab3 = st.tabs(["📄 Analiz", "🔥 Agresif", "📊 Veri"])

        with tab1:
            st.markdown(analiz)

        with tab2:
            st.markdown(agresif)

        with tab3:
            st.dataframe(df.tail(10), width="stretch")

        # --- CHAT CONTEXT ---
        st.session_state.aktif_analiz_baglami = f"""
        Hisse: {clean_symbol}
        Fiyat: {son_fiyat}
        Tahmin: {sonuc_dl.get('tahmin')}
        Yön: {yon}
        """

    except Exception as e:
        st.error(f"Hata oluştu: {e}")