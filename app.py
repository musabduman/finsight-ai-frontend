import pandas as pd
import yfinance as yf
import time
import streamlit as st
import matplotlib.pyplot as plt

from duckduckgo_search import DDGS
from indicators.technical import teknik_analiz
from ai.pythorc import deeplearning
from ai.llm import Gemini, GroqDenetci

st.set_page_config(page_title="AI Borsa Asistanı", page_icon="📈", layout="wide")

st.sidebar.title("🤖 Kontrol Paneli")
secim = st.sidebar.radio("Mod Seçiniz", ["Tek Hisse Analizi", "BIST30 Tarama", "Mega Tarama"])

st.sidebar.info("""
**Aktif Ajanlar:**
* 🧠 Gemini 1.5 Flash (Analist)
* 🛡️ Groq (Denetçi)
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

st.sidebar.markdown("---")
st.sidebar.subheader("🔑 API Ayarları")
# Kullanıcıdan API anahtarını şifreli (yıldızlı) şekilde alıyoruz
kullanici_api_key = st.sidebar.text_input("Gemini API Key", type="password", help="Google AI Studio'dan alabilirsiniz.", key="gemini_hafıza")
groq_api_key = st.sidebar.text_input("Groq API Key (Denetçi)", type="password",help="Groq'un kendi sitesinden alabilirsiniz.", key="groq hafıza") 

if not kullanici_api_key or not groq_api_key:
    st.sidebar.warning("Sistemi kullanmak için her iki API Key'i de giriniz.")
    st.stop() # Anahtarlar yoksa kod aşağıya inmez

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
                gemini_bot=Gemini(api_key=kullanici_api_key)

                try:
                    groq_bot=GroqDenetci(api_key=groq_api_key,model="llama-3.1-8b-instant")
                except Exception as e:
                    st.error(f"Groq'a ulaşılamadı! {e}")
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

                denetleme=groq_bot(df,analiz_sonucu)

                my_bar.progress(100, text="Yorum Tamamlandı!")
                time.sleep(0.5)
                my_bar.empty()

                # --- ANALİZ ÖNCESİ VERİ TEMİZLİK ZIRHI ---
                # Tüm NaN değerleri temizleyelim ki o meşhur hatayı bir daha görme
                df_temiz = df.copy()
                df_temiz = df_temiz.ffill().bfill().fillna(0) # NaN'ları doldur

                # --- GRAFİK KISMI ---
                st.subheader(f"📊 {sembol} Analiz Paneli")
                # LargeUtf8 hatasından kurtulmak için veriyi saf listeye çeviriyoruz
                # Bu sayede Arrow paketleme sistemini tamamen devre dışı bırakırız
                grafik_listesi = df_temiz['Close'].tolist() 
                st.line_chart(grafik_listesi)

                # --- METRİKLER (NaN Korumalı) ---
                son_fiyat = float(df_temiz['Close'].iloc[-1])
                rsi_deger = float(df_temiz['RSI'].iloc[-1])
                                
                c1,c2,c3,c4=st.columns(4)
                son_fiyat=df['Close'].iloc[-1]
                c1.metric("Son Fiyat",f"{son_fiyat:.2f}₺")
                c2.metric("PyThorc hedefi", f"Yön: {sonuc_dl['yön']}, hedef: {sonuc_dl['tahmin']}₺, güven: %{sonuc_dl['güven']}")
                c3.metric("RSI",f"{df['RSI'].iloc[-1]:.1f}")
                c4.metric("MACD Sinyali", f"{df['MACD'].iloc[-1]:.2f}")
            
                tab1,tab2,tab3=st.tabs(["📄 Gemini Raporu", "🛡️ Groq Denetimi", "🧮 Veri Tablosu"])
                with tab1:
                    st.markdown(analiz_sonucu)
                with tab2:
                    if "HATA" in denetleme or "⚠️" in denetleme:
                        st.error(denetleme)
                    else:
                        st.success(denetleme)
                with tab3:
                    st.dataframe(df.tail(10))

elif secim == "Mega Tarama":
    st.subheader("📊 BIST100 Hızlı Yapay Zeka Taraması")
    st.markdown("Bu modül, BIST30 hisselerinin teknik göstergelerini ve PyTorch tahminlerini hesaplayarak fırsatları listeler.")

    # Güncel BIST100 Hisseleri (Gerektiğinde güncelleyebilirsiniz)
    bist100_hisseler = [
        "AEFES.IS", "AGHOL.IS", "AKBNK.IS", "AKCNS.IS", "AKSA.IS", "AKSEN.IS", "ALARK.IS", "ALBRK.IS", "ALGYO.IS", "ALKIM.IS",
            "ARCLK.IS", "ASELS.IS", "ASTOR.IS", "BERA.IS", "BIMAS.IS", "BRSAN.IS", "BRYAT.IS", "BUCIM.IS", "CANTE.IS", "CCOLA.IS",
            "CEMTS.IS", "CIMSA.IS", "DOAS.IS", "DOHOL.IS", "ECILC.IS", "EGEEN.IS", "EKGYO.IS", "ENJSA.IS", "ENKAI.IS", "EREGL.IS",
            "EUREN.IS", "FROTO.IS", "GARAN.IS", "GENIL.IS", "GESAN.IS", "GLYHO.IS", "GUBRF.IS", "HALKB.IS", "HEKTS.IS", "IPEKE.IS",
            "ISCTR.IS", "ISDMR.IS", "ISGYO.IS", "ISMEN.IS", "IZMDC.IS", "KARSN.IS", "KCAER.IS", "KCHOL.IS", "KONTR.IS", "KORDS.IS",
            "KOZAL.IS", "KOZAA.IS", "KRDMD.IS", "MGROS.IS", "ODAS.IS", "OTKAR.IS", "OYAKC.IS", "PETKM.IS", "PGSUS.IS", "SAHOL.IS",
            "SASA.IS", "SISE.IS", "SKBNK.IS", "SMRTG.IS", "SNGYO.IS", "SOKM.IS", "TAVHL.IS", "TCELL.IS", "THYAO.IS", "TKFEN.IS",
            "TOASO.IS", "TSKB.IS", "TTKOM.IS", "TTRAK.IS", "TUKAS.IS", "TUPRS.IS", "ULKER.IS", "VAKBN.IS", "VESBE.IS", "VESTL.IS",
            "YKBNK.IS", "YYLGD.IS", "ZOREN.IS"
    ]

    if st.button("Mega Taramayı Başlat", type="primary"):
        progress_bar = st.progress(0, text="BIST100 hisseleri taranıyor, lütfen bekleyin...")
        tarama_sonuclari = []
        
        # PyTorch modelini bir kez başlatıyoruz
        dl_bot = deeplearning() 

        for i, sembol in enumerate(bist100_hisseler):
            # İlerleme çubuğunu güncelle
            progress_bar.progress((i + 1) / len(bist100_hisseler), text=f"({i+1}/{len(bist100_hisseler)}) {sembol} analiz ediliyor...")
            
            hisse, clean_symbol, df = get_stock_data(sembol)
            
            if df is not None and not df.empty:
                # Teknik analiz verilerini hesapla
                df = teknik_analiz(df)
                df_muhasebeci = df[['Open','High','Low','Close','Volume']].dropna()
                
                # PyTorch (Kahin) Tahmini
                try:
                    sonuc_dl = dl_bot.analiz_et(df_muhasebeci)
                    yon = sonuc_dl.get('yön', 'Nötr')
                    hedef = sonuc_dl.get('tahmin', 0.0)
                    guven = sonuc_dl.get('güven', 0.0)
                except Exception:
                    yon, hedef, guven = "Hata", 0.0, 0.0
                
                # Son verileri çek
                son_fiyat = df['Close'].iloc[-1]
                rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns else 0
                macd = df['MACD'].iloc[-1] if 'MACD' in df.columns else 0
                
                
                # Tablo için sözlüğe ekle
                tarama_sonuclari.append({
                    "Hisse": clean_symbol.replace(".IS", ""),
                    "Son Fiyat (₺)": round(son_fiyat, 2),
                    "PyTorch Yön": yon,
                    "Hedef Fiyat (₺)": round(hedef, 2),
                    "Güven Skor (%)": round(guven, 2),
                    "RSI": round(rsi, 2),
                    "MACD": round(macd, 2)
                })
                
            # Yahoo Finance ban yememek için ufak bir bekleme süresi
            time.sleep(0.1) 
                
        progress_bar.empty()
        
        if tarama_sonuclari:
            st.success("Tarama başarıyla tamamlandı!")
            
            # Verileri Pandas DataFrame'e çevir ve Streamlit'te göster
            sonuc_df = pd.DataFrame(tarama_sonuclari).round(2)
            def yon_ikonu_ekle(yon):
                yon_str = str(yon).upper()
                if "YÜKSELİŞ" in yon_str or "AL" in yon_str:
                    return f"🟢 📈 {yon}"
                elif "DÜŞÜŞ" in yon_str or "SAT" in yon_str:
                    return f"🔴 📉 {yon}"
                else:
                    return f"🟡 ➖ {yon}"

            # 3. Güven Skoruna emoji ekleme fonksiyonu
            def guven_ikonu_ekle(skor):
                try:
                    skor_val = float(skor)
                    if skor_val >= 80:
                        return f"🔥 %{skor_val}"
                    elif skor_val >= 60:
                        return f"👍 %{skor_val}"
                    else:
                        return f"⚠️ %{skor_val}"
                except:
                    return skor

            # 4. Fonksiyonları DataFrame'e uygula
            sonuc_df['PyTorch Yön'] = sonuc_df['PyTorch Yön'].apply(yon_ikonu_ekle)
            sonuc_df['Güven Skor (%)'] = sonuc_df['Güven Skor (%)'].apply(guven_ikonu_ekle)
            
            # Tabloyu Streamlit'in interaktif dataframe bileşeni ile gösteriyoruz
            # Kullanıcılar sütun başlıklarına tıklayarak RSI, Güven Skoru vb. filtrelemeler yapabilir
            st.dataframe(
                sonuc_df.style.apply(lambda x: ['background: #1e3d2f' if v == 'Al' else 'background: #3d1e1e' if v == 'Sat' else '' for v in x], subset=['PyTorch Yön']),
                use_container_width=True,
                hide_index=True
            )
            
            st.info("💡 **İpucu:** Detaylı Gemini ve Groq raporu almak istediğiniz hisseyi soldaki 'Tek Hisse Analizi' menüsünden aratabilirsiniz.")

elif secim == "BIST30 Tarama":
    st.subheader("🎯 BIST30 Sinyal Avcısı")
    st.markdown("Sadece özel indikatör sinyalleri (Ralli, Wonderkid, Erken Uyarı) üreten hisseler filtrelenir ve yapay zeka heyeti tarafından derin analize sokulur.")

    def sinyal_kontrol(df):
        
        try:
            son = df.iloc[-1]
            # Not: Bu sütunların (Width, Signal, MACD_signal vb.) teknik_analiz(df) içinde hesaplandığından emin olun!
            wonderkid = (son.get('Width', 1) < 0.15) and (son.get('RSI', 50) < 60)
            erken_uyari = (son.get('MACD_signal', 0) == 1) and (son.get('Signal', 0) == 1)
            ralli = (son.get('MACD_signal', 0) == 1) and (son.get('Signal', 0) == 1) and (son.get('Volume_signal', 0) == 1)

            if ralli:
                return True, "🚀 Ralli Modu"
            elif wonderkid:
                return True, "💎 Wonderkid Modu"
            elif erken_uyari:
                return True, "⚠️ Erken Uyarı"
            return False, "Temiz"
        except KeyError as e:
            return False, f"Eksik bilgi indikatör verisi: {e}"

    bist30_hisseler = [
        "AKBNK", "ALARK", "ARCLK", "ASELS", "ASTOR", "BIMAS", "BRSAN", "CCOLA", 
        "EKGYO", "ENKAI", "EREGL", "FROTO", "GARAN", "GUBRF", "HEKTS", "ISCTR", 
        "KCHOL", "KONTR", "KOZAA", "KOZAL", "KRDMD", "OYAKC", "PETKM", "PGSUS", 
        "SAHOL", "SASA", "SISE", "TCELL", "THYAO", "TOASO", "TUPRS", "YKBNK"
    ]

    if st.button("Sinyal Taramasını Başlat", type="primary"):
        progress_bar = st.progress(0, text="BIST30 hisselerinde sinyal aranıyor...")
        bulunan_hisseler = []
        
        # Sadece sinyal çıkarsa kullanılacak olan botları baştan tanımlıyoruz
        dl_bot = deeplearning()
        gemini_bot = Gemini(api_key=kullanici_api_key)
        try:
            groq_bot = GroqDenetci(api_key=groq_api_key, model="llama-3.1-8b-instant")
        except Exception as e:
            st.error(f"Groq'a ulaşılamadı! {e}")
            st.stop()

        # 1. AŞAMA: Hızlı Tarama ve Filtreleme
        for i, sembol in enumerate(bist30_hisseler):
            progress_bar.progress((i + 1) / len(bist30_hisseler), text=f"Radar devrede: {sembol} taranıyor...")
            
            hisse, clean_symbol, df = get_stock_data(sembol)
            
            if df is not None and not df.empty:
                df = teknik_analiz(df)
                sinyal_var_mi, mesaj = sinyal_kontrol(df)
                
                # Eğer sinyal varsa, derin analiz için listeye ekle
                if sinyal_var_mi:
                    bulunan_hisseler.append((hisse, clean_symbol, df, mesaj))
            
            time.sleep(0.1) # Yahoo Finance ban koruması
            
        progress_bar.empty()
        
        # 2. AŞAMA: Bulunan Hisseler İçin Derin Analiz
        if not bulunan_hisseler:
            st.warning("Bugün hiçbir BIST30 hissesinde belirlediğiniz sinyaller (Ralli, Wonderkid, Erken Uyarı) bulunamadı. Nakitte kalmak da bir pozisyondur!")
        else:
            st.success(f"Tebrikler! Radara {len(bulunan_hisseler)} adet hisse takıldı. Derin yapay zeka analizi başlıyor...")
            analiz_bar = st.progress(0, text="Yapay zeka ajanları çalışıyor, lütfen API yanıtlarını bekleyin...")
            # Bulunan her hisse için expander (açılır kapanır kutu) oluştur
           
            for idx,(hisse, clean_symbol, df, mesaj) in enumerate(bulunan_hisseler):
                analiz_bar.progress((idx + 1) / len(bulunan_hisseler), text=f"({idx+1}/{len(bulunan_hisseler)}) {clean_symbol} için derin analiz yapılıyor...")
                
                with st.expander(f"📌 {clean_symbol} | Yakalanan Sinyal: {mesaj}", expanded=False):
                    try:
                        df = df.ffill().bfill().fillna(0)    
                        st.info(f"{clean_symbol} için yapay zeka ajanları çalışıyor, lütfen bekleyin...")
                        
                        # PyTorch Sayısal Tahmin
                        df_muhasebeci = df[['Open','High','Low','Close','Volume']].dropna()
                        if len(df_muhasebeci) < 50:
                            sonuc_dl = {'yön': 'Veri Yetersiz', 'tahmin': 0, 'güven': 0}
                        else:
                            sonuc_dl = dl_bot.analiz_et(df_muhasebeci)
                        
                        dl_tahmin = 0 if pd.isna(sonuc_dl.get('tahmin', 0)) else sonuc_dl.get('tahmin', 0)
                        dl_guven = 0 if pd.isna(sonuc_dl.get('güven', 0)) else sonuc_dl.get('güven', 0)
                        
                        ai_rapor = f"Yön: {sonuc_dl.get('yön', 'Nötr')}, hedef: {sonuc_dl.get('tahmin', 0)} TL, güven: %{sonuc_dl.get('güven', 0)}"
                        
                        # Temel Analiz Verileri
                        info = hisse.info
                        temel = {
                            "FK": info.get('trailingPE', 'Yok'),
                            "PD/DD": info.get('priceToBook', 'Yok'),
                            "Sektor": info.get('sector', 'Bilinmiyor')
                        }
                        
                        # Haberler ve LLM Raporları
                        haberler_listesi = haber_cek_web(clean_symbol)
                        analiz_sonucu = gemini_bot(clean_symbol, temel, df, haberler_listesi, ai_rapor)
                        denetleme = groq_bot(df, analiz_sonucu)
                        
                        # Metrikleri Göster
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Son Fiyat", f"{df['Close'].iloc[-1]:.2f}₺")
                        c2.metric("Durum", mesaj)
                        c3.metric("Kahin (DL) Hedef", f"{sonuc_dl.get('tahmin', 0)}₺")
                        c4.metric("Kahin Güveni", f"%{sonuc_dl.get('güven', 0)}")
                        
                        # Raporları Sekmeler Halinde Göster
                        tab1, tab2 = st.tabs(["📄 Analist (Gemini)", "🛡️ Denetçi (Groq)"])
                        with tab1:
                            st.markdown(analiz_sonucu)
                        with tab2:
                            if "HATA" in denetleme or "⚠️" in denetleme:
                                st.error(denetleme)
                            else:
                                st.success(denetleme)
                    except Exception as e:
                        st.error(f"⚠️ {clean_symbol} analizi sırasında bir hata oluştu (Muhtemelen API limiti aşıldı). Detay: {e}")
                time.sleep(7)
            analiz_bar.empty()
            st.balloons()
            st.success("✅ Tüm hisselerin derin yapay zeka analizi başarıyla tamamlandı! Yukarıdaki sekmeleri açarak raporları okuyabilirsiniz.")