import pandas as pd
import yfinance as yf
import time
import os
import streamlit as st

from ddgs import DDGS
from indicators.technical import teknik_analiz
from ai.pythorc import deeplearning
from ai.llm import Gemini, OllamaLLM

st.set_page_config(page_title="AI Borsa Asistanı", page_icon="📈", layout="wide")

GOOGLE_API_KEY= "AIzaSyBUqEdgpxuvePLAD2fl2t9J7Vtc8I4Lmws"

st.sidebar.title("🤖 Kontrol Paneli")
secim = st.sidebar.radio("Mod Seçiniz", ["Tek Hisse Analizi", "BIST30 Tarama", "Mega Tarama"])

st.sidebar.info("""
**Aktif Ajanlar:**
* 🧠 Gemini 1.5 Flash (Analist)
* 🛡️ Ollama (Denetçi)
* 🧮 PyTorch (Kahin) 
""")

def get_stock_data(symbol):
    tr_to_en = str.maketrans("ıiğüşöçIİĞÜŞÖÇ", "IIGUSOCIIGUSOC")
    clean_symbol = str(symbol).translate(tr_to_en).upper().strip()
    if not clean_symbol.endswith(".IS"):
        clean_symbol+=".IS"
    try:
        hisse=yf.Ticker(clean_symbol)
        df=hisse.history(period="3y")
        return hisse ,clean_symbol,df
    except Exception as e:
        print(f"⚠️ DİKKAT: Yahoo Finance '{clean_symbol}' için BOŞ veri gönderdi. (Hisse adını yanlış yazmış veya ban yemiş olabiliriz)")
        st.stop()
        return None,None,None
    
def haber_cek_web(symbol):
    haberler_listesi = []
    try:
        with DDGS() as ddgs:
            query = f"{symbol.replace('.IS','')} hisse haberleri"
            result = ddgs.news(keywords=query, region="tr-tr", safesearch="off", max_results=5)
            for r in result:
                tarih = r.get('date', '')[:10]
                baslik = r.get('title', 'Başlık yok')
                kaynak = r.get('source', 'Bilinmiyor')
                haberler_listesi.append(f"-[{tarih}]{kaynak}:{baslik}")
    except:
        return "Haber verisi cekilemedi"
    return haberler_listesi

st.title("🚀 Borsa İstanbul Yapay Zeka Analisti")
st.markdown("---")

if secim== "Tek Hisse Analizi":
    col1,col2=st.columns([3,1])
    with col1:
        sembol_input=st.text_input("Hisse ismini giriniz (Örn: THYAO, GARAN)")
    
    if st.button("Analizi Başlat", type="primary"):
        if not sembol_input:
            st.warning("Lütfen bir Hisse ismi girin!")
        else:
            progress_text="Yapay zekalar göreve çağrılıyor..."
            my_bar=st.progress(0, text=progress_text)

            hisse, sembol, df= get_stock_data(sembol_input)

            if df is None or df.empty:
                st.error("Hisse bulunamadı ya da veri çekilemedi!")
            else:
                my_bar.progress(20, text="Veriler çekildi, teknik analiz yapılıyor...")
                df=teknik_analiz(df)

                dl_bot=deeplearning()
                gemini_bot=Gemini(api_key=GOOGLE_API_KEY)

                try:
                    ollama_bot=OllamaLLM(model="gemma3:4b")
                except:
                    st.error("Ollamaya ulaşılamadı!")
                    st.stop()
                
                my_bar.progress(50, text="Pythorc sayısal tahmin yapıyor...")
                df_muhasebeci=df[['Open','High','Low','Close','Volume']].dropna()
                sonuc_dl=dl_bot.analiz_et(df_muhasebeci)
                ai_rapor=f"Yön: {sonuc_dl['yön']}, hedef: {sonuc_dl['tahmin']} TL, güven: %{sonuc_dl['güven']}"
                my_bar.progress(70, text="Gemini yorumunu hazırlıyor...")

                info=hisse.info
                temel={
                    "FK": info.get('trailingPE', 'Yok'),
                    "PD/DD": info.get('priceToBook', 'Yok'),
                    "Sektor": info.get('sector', 'Bilinmiyor')
                }

                haberler_listesi=haber_cek_web(sembol)
                analiz_sonucu=gemini_bot(sembol,temel,df,haberler_listesi,ai_rapor)

                denetleme=ollama_bot(df,analiz_sonucu)

                my_bar.progress(100, text="Yorum Tamamlandı!")
                time.sleep(0.5)
                my_bar.empty()

                st.subheader(f"📊 {sembol} Analiz Paneli")

                c1,c2,c3,c4=st.columns(4)
                son_fiyat=df['Close'].iloc[-1]
                c1.metric("Son Fiyat",f"{son_fiyat:.2f}₺")
                c2.metric("PyThorc hedefi", f"Yön: {sonuc_dl['yön']}, hedef: {sonuc_dl['tahmin']}₺, güven: %{sonuc_dl['güven']}")
                c3.metric("RSI",f"{df['RSI'].iloc[-1]:.1f}")
                c4.metric("MACD Sinyali", f"{df['MACD'].iloc[-1]}")

                st.line_chart(df['Close'])

                tab1,tab2,tab3=st.tabs(["📄 Gemini Raporu", "🛡️ Ollama Denetimi", "🧮 Veri Tablosu"])
                with tab1:
                    st.markdown(analiz_sonucu)
                with tab2:
                    if "HATA" in denetleme or "⚠️" in denetleme:
                        st.error(denetleme)
                    else:
                        st.success(denetleme)
                with tab3:
                    st.dataframe(df.tail(10))

elif secim=="BIST30 Tarama":
    st.info("bu modül yakında eklenecek...")
elif secim=="Mega Tarama":
    st.info("bu modül yakında eklenecek...")
                