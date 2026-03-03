import pandas as pd
import yfinance as yf
import time
import streamlit as st
import matplotlib.pyplot as plt

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

def normalize_symbol(symbol: str):
    tr_to_en = str.maketrans("ıiğüşöçIİĞÜŞÖÇ", "IIGUSOCIIGUSOC")
    clean_symbol = str(symbol).translate(tr_to_en).upper().strip()
    if not clean_symbol.endswith(".IS"):
        clean_symbol += ".IS"
    return clean_symbol

@st.cache_data(ttl=1800)
def get_price_data(symbol):
    df=yf.download(symbol, period="3y", progress=False, multi_level_index=False)
    
    if df.empty:
        raise ValueError("Boş veri döndü (muhtemelen Yahoo limiti veya sembol hatası)")
    
    return df

@st.cache_data(ttl=1800)
def get_fast_info(symbol):
    ticker=yf.Ticker(symbol).fast_info
    return dict(ticker)

def get_temel_info(symbol):
    """Temel analiz verisi — FK, PD/DD, Sektör için ticker.info kullanır"""
    try:
        info = yf.Ticker(symbol).info
        return {
            "FK": info.get("trailingPE", "Yok"),
            "PD/DD": info.get("priceToBook", "Yok"),
             "Sektor": info.get("sector", "Bilinmiyor"),
            "Kar Marji": info.get("profitMargins", "Yok"),
            "Piyasa Degeri": info.get("marketCap", "Yok"),
        }
    except Exception:
        return {"FK": "Yok", "PD/DD": "Yok", "Sektor": "Bilinmiyor",
                "Kar Marji": "Yok", "Piyasa Degeri": "Yok"}

def get_stock_data(symbol):
    try:
        clean_symbol = normalize_symbol(symbol)
        df = get_price_data(clean_symbol)
        info = get_fast_info(clean_symbol)   # sadece fiyat için
        return clean_symbol, df, info
    except Exception as e:
        st.error(f"⚠️ '{symbol}' için veri alınamadı: {e}")
        return None, None, None
    
def haber_cek_web(symbol):
    haberler_listesi = []
    try:
        ticker = yf.Ticker(symbol)
        news = ticker.news

        if not news:
            return ["Son güncel haber bulunamadı."]

        for n in news[:5]:
            try:
                # ── YENİ FORMAT: yfinance >= 0.2.50 ──
                if "content" in n and isinstance(n["content"], dict):
                    icerik = n["content"]
                    baslik = icerik.get("title", "Başlık yok")
                    kaynak = (icerik.get("provider") or {}).get("displayName", "Bilinmiyor")
                    tarih_raw = icerik.get("pubDate", "")
                    # "2024-01-15T10:30:00Z" → "2024-01-15"
                    tarih = tarih_raw[:10] if tarih_raw else "Tarih Yok"

                # ── ESKİ FORMAT: yfinance < 0.2.50 ──
                else:
                    baslik = n.get("title", "Başlık yok")
                    kaynak = n.get("publisher", "Bilinmiyor")
                    tarih_unix = n.get("providerPublishTime")
                    tarih = time.strftime("%Y-%m-%d", time.localtime(tarih_unix)) \
                            if tarih_unix else "Tarih Yok"

                haberler_listesi.append(f"- [{tarih}] {kaynak}: {baslik}")

            except Exception:
                continue  # Tek bir haber parse edilemezse diğerine geç

    except Exception as e:
        return [f"Haber verisi çekilemedi. Detay: {e}"]

    return haberler_listesi if haberler_listesi else ["Haber bulunamadı."]

st.title("🚀 Borsa İstanbul Analisti")
st.markdown("---")

st.sidebar.markdown("---")
st.sidebar.subheader("🔑 API Ayarları")
# Kullanıcıdan API anahtarını şifreli (yıldızlı) şekilde alıyoruz
kullanici_api_key = st.sidebar.text_input("Gemini API Key", type="password", help="Google AI Studio'dan alabilirsiniz.", key="gemini_hafıza")
groq_api_key = st.sidebar.text_input("Groq API Key (Agresif Yorumcu)", type="password",help="Groq'un kendi sitesinden alabilirsiniz.", key="groq hafıza") 
agresif_yorum=""
if not kullanici_api_key:
    st.sidebar.warning("⚠️ Gemini API Key eksik!")
else:
    st.sidebar.success("✅ Gemini Hazır")

if not groq_api_key:
    st.sidebar.warning("ℹ️ Groq anahtarı yok: Agresif yorumcu modu pasif.")

if secim== "Tek Hisse Analizi":
    sembol_input=st.text_input("Hisse ismini giriniz (Örn: THYAO, GARAN)")
    analiz_button=st.button("Analizi Başlat", type="primary")

    col1,col2=st.columns([3,1])

    with col1:    
        if analiz_button:
            if not sembol_input:
                st.warning("Lütfen bir Hisse ismi girin!")
                if not sembol_input:
                    st.warning("Lütfen bir Hisse ismi girin!")
                elif not kullanici_api_key:
                    st.error("Analiz için önce Gemini API anahtarını girmelisiniz!")
            elif not kullanici_api_key: # <-- BURASI KRİTİK: Anahtar yoksa içeri sokma!
                st.error("Analiz için önce Gemini API anahtarını girmelisiniz!")
            else:
                progress_text="Yapay zekalar göreve çağrılıyor..."
                my_bar=st.progress(0, text=progress_text)

                clean_symbol, df, info= get_stock_data(sembol_input)
                if  df is None or df.empty:
                        st.error("Veri çekilemediği için analize devam edilemiyor. Lütfen biraz bekleyip tekrar deneyin.")
                        st.stop()

                try:
                    my_bar.progress(20, text="Veriler çekildi, teknik analiz yapılıyor...")
                    df=teknik_analiz(df)
                    
                    # --- ANALİZ ÖNCESİ VERİ TEMİZLİK ZIRHI ---
                    # Tüm NaN değerleri temizleyelim ki o meşhur hatayı bir daha görme
                    df = df.ffill().bfill().fillna(0) # NaN'ları doldur

                    dl_bot=deeplearning()
                    gemini_bot=Gemini(api_key=kullanici_api_key)
                    my_bar.progress(50, text="Pythorc sayısal tahmin yapıyor...")
                    
                    df_muhasebeci=df[['Open','High','Low','Close','Volume']].dropna()
                    if len(df_muhasebeci)<50:
                        sonuc_dl = {'yön': 'Veri Yetersiz', 'tahmin': 0, 'güven': 0}
                    else:
                        sonuc_dl=dl_bot.analiz_et(df_muhasebeci)
                    
                    ai_rapor=f"Yön: {sonuc_dl.get('yön', 'Nötr')}, hedef: {sonuc_dl.get('tahmin', 0)} TL, güven: %{sonuc_dl.get('güven', 0)}"
                    
                    my_bar.progress(70, text="Gemini yorumunu hazırlıyor...")
                    haberler_listesi=haber_cek_web(clean_symbol)
                    df_kısa=df.tail(30)
                    temel={
                        "FK": info.get('trailingPE', 'Yok'),
                        "PD/DD": info.get('priceToBook', 'Yok'),
                        "Sektor": info.get('sector', 'Bilinmiyor')
                    }    
                    analiz_sonucu=gemini_bot(clean_symbol, temel, df_kısa, haberler_listesi, ai_rapor)

                    if groq_api_key:
                        my_bar.progress(90, text="Groq analizi denetliyor...")
                        groq_bot = GroqDenetci(api_key=groq_api_key, model="llama-3.1-8b-instant")
                        agresif_yorum = groq_bot(df_kısa, analiz_sonucu,ai_rapor)
                    else:
                        # Groq yoksa direkt Gemini sonucunu bas
                        agresif_yorum="ℹ️ Groq API anahtarı girilmediği için agresif yorum pasif."

                except Exception as e:
                    st.error( f"Hisse bulunamadı ya da veri çekilemedi! {e}")
                    analiz_sonucu=""

                    my_bar.progress(100, text="Yorum Tamamlandı!")
                    time.sleep(0.5)
                    my_bar.empty()

                    # --- GRAFİK KISMI ---
                    # LargeUtf8 hatasından kurtulmak için veriyi saf listeye çeviriyoruz
                    # Bu sayede Arrow paketleme sistemini tamamen devre dışı bırakırız
                    st.subheader(f"📊 {clean_symbol} Analiz Paneli")
                    grafik_listesi = df['Close'].tolist() 
                    st.line_chart(grafik_listesi)

                    # --- METRİKLER (NaN Korumalı) ---
                    son_fiyat = float(df['Close'].iloc[-1])
                    rsi_deger = float(df['RSI'].iloc[-1])
                                    
                    c1,c2,c3,c4=st.columns(4)
                    son_fiyat=df['Close'].iloc[-1]
                    c1.metric("Son Fiyat",f"{son_fiyat:.2f}₺")
                    c2.metric("PyThorc hedefi", f"Yön: {sonuc_dl['yön']}, hedef: {sonuc_dl['tahmin']}₺, güven: %{sonuc_dl['güven']}")
                    c3.metric("RSI",f"{df['RSI'].iloc[-1]:.1f}")
                    c4.metric("MACD Sinyali", f"{df['MACD'].iloc[-1]:.2f}")
                
                    tab1,tab2,tab3=st.tabs(["📄 Uzun Vadeli Rapor", "🚀 Agresif Yorum", "🧮 Veri Tablosu"])
                    with tab1:
                        st.markdown(analiz_sonucu)
                    with tab2:
                        if "HATA" in agresif_yorum or "⚠️" in agresif_yorum:
                            st.error(agresif_yorum)
                        else:
                            st.success(agresif_yorum)
                    with tab3:
                        st.dataframe(df.tail(10))

elif secim == "Mega Tarama":
    st.subheader("📊 BIST100 Hızlı Yapay Zeka Taraması")
    st.markdown("Bu modül, BIST100 hisselerinin teknik göstergelerini ve PyTorch tahminlerini hesaplayarak fırsatları listeler.")

    # Güncel BIST100 Hisseleri (Gerektiğinde güncelleyebilirsiniz)
    bist100_hisseler = [
        "AEFES.IS", "AGHOL.IS", "AKBNK.IS", "AKCNS.IS", "AKSA.IS", "AKSEN.IS", "ALARK.IS", "ALBRK.IS", "ALGYO.IS", "ALKIM.IS",
            "ARCLK.IS", "ASELS.IS", "ASTOR.IS", "BERA.IS", "BIMAS.IS", "BRSAN.IS", "BRYAT.IS", "BUCIM.IS", "CANTE.IS", "CCOLA.IS",
            "CEMTS.IS", "CIMSA.IS", "DOAS.IS", "DOHOL.IS", "ECILC.IS", "EGEEN.IS", "EKGYO.IS", "ENJSA.IS", "ENKAI.IS", "EREGL.IS",
            "EUREN.IS", "FROTO.IS", "GARAN.IS", "GENIL.IS", "GESAN.IS", "GLYHO.IS", "GUBRF.IS", "HALKB.IS", "HEKTS.IS", "IPEKE.IS",
            "ISCTR.IS", "ISDMR.IS", "ISGYO.IS", "ISMEN.IS", "IZMDC.IS", "KARSN.IS", "KCAER.IS", "KCHOL.IS", "KONTR.IS", "KORDS.IS",
            "KRDMD.IS", "MGROS.IS", "ODAS.IS", "OTKAR.IS", "OYAKC.IS", "PETKM.IS", "PGSUS.IS", "SAHOL.IS",
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
            
            clean_symbol, df, info = get_stock_data(sembol)
            
            try:
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
            except Exception as e:
                st.error(f"Bir hata ile karşılaşıldı {e}")        
            # Yahoo Finance ban yememek için ufak bir bekleme süresi
            time.sleep(0.5) 
                
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
            erken_uyari = (
                (son.get('MACD_signal', 0) == 1) and
                (son.get('BOLL_signal', 0) == 1)
            )

            ralli = (
                (son.get('MACD_signal', 0) == 1) and
                (son.get('BOLL_signal', 0) == 1) and
                (son.get('VOLUME_signal', 0) == 1)
            )

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
        "KCHOL", "KONTR", "KRDMD", "OYAKC", "PETKM", "PGSUS", 
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
            
            clean_symbol, df, info = get_stock_data(sembol)
            
            try:
                df = teknik_analiz(df)
                sinyal_var_mi, mesaj = sinyal_kontrol(df)
                
                # Eğer sinyal varsa, derin analiz için listeye ekle
                if sinyal_var_mi:
                    bulunan_hisseler.append((info, clean_symbol, df, mesaj))
            except Exception as e:
                st.error(f"Hata! {e}")
                
            time.sleep(0.1) # Yahoo Finance ban koruması
            
        progress_bar.empty()
        
        # 2. AŞAMA: Bulunan Hisseler İçin Derin Analiz
        if not bulunan_hisseler:
            st.warning("Bugün hiçbir BIST30 hissesinde belirlediğiniz sinyaller (Ralli, Wonderkid, Erken Uyarı) bulunamadı. Nakitte kalmak da bir pozisyondur!")
        else:
            st.success(f"Tebrikler! Radara {len(bulunan_hisseler)} adet hisse takıldı. Derin yapay zeka analizi başlıyor...")
            analiz_bar = st.progress(0, text="Yapay zeka ajanları çalışıyor, lütfen API yanıtlarını bekleyin...")
            # Bulunan her hisse için expander (açılır kapanır kutu) oluştur
           
            for idx,(info, clean_symbol, df, mesaj) in enumerate(bulunan_hisseler):
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
                        info = get_fast_info(clean_symbol)
                        temel = {
                            "FK": info.get('trailingPE', 'Yok'),
                            "PD/DD": info.get('priceToBook', 'Yok'),
                            "Sektor": info.get('sector', 'Bilinmiyor')
                        }
                        
                        # Haberler ve LLM Raporları
                        haberler_listesi = haber_cek_web(clean_symbol)
                        analiz_sonucu = gemini_bot(clean_symbol, temel, df, haberler_listesi, ai_rapor)
                        agresif_yorum = groq_bot(df, analiz_sonucu,ai_rapor)
                        
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
                            if "HATA" in agresif_yorum or "⚠️" in agresif_yorum:
                                st.error(agresif_yorum)
                            else:
                                st.success(agresif_yorum)
                    except Exception as e:
                        st.error(f"⚠️ {clean_symbol} analizi sırasında bir hata oluştu (Muhtemelen API limiti aşıldı). Detay: {e}")
                time.sleep(7)
            analiz_bar.empty()
            st.balloons()
            st.success("✅ Tüm hisselerin derin yapay zeka analizi başarıyla tamamlandı! Yukarıdaki sekmeleri açarak raporları okuyabilirsiniz.")