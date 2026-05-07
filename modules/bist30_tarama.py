import streamlit as st
import time
import pandas as pd

# --- SİNYAL MOTORU ---
def sinyal_kontrol(df):
    try:
        son = df.iloc[-1]

        def sig(key, default=0):
            try:
                return int(float(son.get(key, default)))
            except:
                return default

        def val(key, default=0):
            try:
                return float(son.get(key, default))
            except:
                return default

        sbs = val('SBS')
        macd_sig = sig('MACD_signal')
        boll_sig = sig('BOLL_signal')
        vol_sig = sig('VOLUME_signal')
        rsi = val('RSI', 50)
        width = val('Width', 1)
        macd_val = val('MACD')
        fiyat = val('Close')
        sma50 = val('SMA_50')
        sma200 = val('SMA_200')

        # --- SİNYALLER ---
        if macd_sig and boll_sig and vol_sig and sbs >= 55:
            return True, "🚀 Ralli"

        if width < 0.25 and rsi < 65 and macd_val > 0 and sbs > 45:
            return True, "💎 Sıkışma"

        if fiyat > sma50 and fiyat > sma200 and 50 < rsi < 75 and macd_val > 0 and sbs >= 50:
            return True, "📈 Trend"

        if macd_sig and (boll_sig or vol_sig) and sbs > 40:
            return True, "⚠️ Erken"

        if rsi < 45 and macd_sig and sbs > 35:
            return True, "🔄 Dip Dönüş"

        if sbs >= 70 and vol_sig:
            return True, "🔥 Para Girişi"

        return False, None

    except Exception:
        return False, None


# --- ANA FONKSİYON ---
def bist30_tarama(get_stock_data, teknik_analiz, dl_bot,
                  Gemini, OllamaAgresif,
                  get_temel_hesapla,
                  anlik_hisse_haberi_cek,
                  get_memory_for_llm,
                  gemini_api, ollama_api):

    st.subheader("🎯 BIST30 Sinyal Avcısı")
    st.markdown("Bu bölüm, BIST30 hisselerini tarar ve yapay zeka destekli analizler sunar. Sinyal kriterlerine uyan hisseler detaylı olarak incelenir. **Not:** Bu işlem biraz zaman alabilir, lütfen sabırlı olun.")

    bist30 = [
        "AKBNK","ALARK","ARCLK","ASELS","ASTOR","BIMAS","BRSAN","CCOLA",
        "EKGYO","ENKAI","EREGL","FROTO","GARAN","GUBRF","HEKTS","ISCTR",
        "KCHOL","KONTR","KRDMD","OYAKC","PETKM","PGSUS",
        "SAHOL","SASA","SISE","TCELL","THYAO","TOASO","TUPRS","YKBNK"
    ]

    if st.button("Taramayı Başlat", type="primary"):

        progress = st.progress(0)
        bulunan = []

        # --- 1. FAST SCAN ---
        for i, sembol in enumerate(bist30):
            progress.progress((i+1)/len(bist30), text=f"{sembol} taranıyor...")

            try:
                clean, df, info = get_stock_data(sembol)
                df, fib20, fib200 = teknik_analiz(df)

                sinyal, tip = sinyal_kontrol(df)

                if sinyal:
                    bulunan.append((clean, df, tip, fib20, fib200))

            except:
                continue

            time.sleep(0.4)  # hız + ban dengesi

        progress.empty()

        if not bulunan:
            st.warning("Bugün sinyal yok.")
            return

        st.success(f"{len(bulunan)} hisse bulundu 🚀")

        # --- BOTLAR (TEK SEFER INIT) ---
        gemini = Gemini(api_key=gemini_api)
        ollama = OllamaAgresif(api_key=ollama_api, model="gpt-oss:120b-cloud")

        analiz_bar = st.progress(0)

        # --- 2. DEEP ANALYSIS ---
        for i, (symbol, df, tip, fib20, fib200) in enumerate(bulunan):

            analiz_bar.progress((i+1)/len(bulunan), text=f"{symbol} analiz ediliyor...")

            with st.expander(f"{symbol} | {tip}"):

                try:
                    df = df.ffill().bfill()

                    son_sbs = df['SBS'].iloc[-1]

                    # DL
                    sonuc = dl_bot.analiz_et(df)
                    ai_rapor = f"{sonuc.get('yön')} | {sonuc.get('tahmin')}₺ | %{sonuc.get('güven')}"

                    # Temel
                    temel = get_temel_hesapla(symbol)

                    # Haber
                    haber = anlik_hisse_haberi_cek(symbol)
                    hafiza = get_memory_for_llm(query=symbol, limit=3)

                    haber_blob = f"{haber}\n{hafiza}"

                    # LLM
                    yorum = gemini(symbol, temel, df.tail(50), haber_blob, ai_rapor, fib200, son_sbs)
                    agresif = ollama.generate(df, ai_rapor, fib20, son_sbs)

                    # METRİK
                    c1,c2,c3,c4 = st.columns(4)
                    c1.metric("Fiyat", f"{df['Close'].iloc[-1]:.2f}₺")
                    c2.metric("Sinyal", tip)
                    c3.metric("Tahmin", f"{sonuc.get('tahmin',0)}₺")
                    c4.metric("SBS", f"%{son_sbs:.1f}")

                    tab1, tab2 = st.tabs(["Gemini", "Ollama"])
                    with tab1:
                        st.markdown(yorum)
                    with tab2:
                        st.markdown(agresif)

                except Exception as e:
                    st.error(f"Hata: {e}")

            time.sleep(1.5)

        analiz_bar.empty()
        st.success("Analiz tamamlandı ✅")