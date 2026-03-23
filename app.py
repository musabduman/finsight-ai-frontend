import time
import requests

import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
import auth_ui

from watchlist import watchlist_sayfasi
from indicators.technical import teknik_analiz
from ai.pythorc import deeplearning
from ai.llm import Gemini, GroqDenetci, GroqChat

st.set_page_config(page_title="AI Borsa Asistanı", page_icon="📈", layout="wide")
is_ready = auth_ui.login_sidebar()
if not is_ready:
    # Giriş yoksa siteyi burada durdurur, aşağıdaki arayüz kodlarını HİÇ OKUMAZ.
    st.info("🚀 Borsa İstanbul Analisti'ne hoş geldiniz! Lütfen sol menüden giriş yapın.")
    st.stop()

# --- YENİ: ANAHTARLARI VERİTABANINDAN OTOMATİK GETİR ---
if "gemini_key" not in st.session_state or "groq_key" not in st.session_state:
    try:
        response = requests.get(f"https://finsight-ai-backend-u1cw.onrender.com/get_keys/{st.session_state.user_email}")
        if response.status_code == 200:
            keys = response.json()
            st.session_state.gemini_key = keys.get("gemini_key", "")
            st.session_state.groq_key = keys.get("groq_key", "")
        else:
            st.sidebar.error("⚠️ API anahtarları veritabanından alınamadı.")
    except Exception as e:
        st.sidebar.error(f"⚠️ API Bağlantı Hatası: {e}")


st.sidebar.title("🤖 Kontrol Paneli")
secim = st.sidebar.radio("Mod Seçiniz", ["İzleme Listesi","Tek Hisse Analizi", "BIST30 Tarama", "Mega Tarama"])

st.sidebar.info("""
**Aktif Ajanlar:**
* 🧠 Gemini 1.5 Flash (Analist)
* 🛡️ Groq (Agresif Analist)
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

#pytorc modelini çağırma
@st.cache_resource
def load_pytorch_model():
    return deeplearning()

def get_temel_hesapla(symbol):
    ticker = yf.Ticker(symbol)
    
    # Bilanço verisi
    income = ticker.financials      # Gelir tablosu
    balance = ticker.balance_sheet  # Bilanço
    fast = ticker.fast_info
    
    try:
        net_kar = income.loc['Net Income'].iloc[0]
        özkaynak = balance.loc['Stockholders Equity'].iloc[0]
        piyasa_degeri = fast['market_cap']
        
        fk = piyasa_degeri / net_kar if net_kar > 0 else "Zararda"
        pd_dd = piyasa_degeri / özkaynak if özkaynak > 0 else "Yok"
        kar_marji = net_kar / income.loc['Total Revenue'].iloc[0]
        
        return {
            "FK": round(fk, 2),
            "PD/DD": round(pd_dd, 2),
            "Kar Marji": f"%{round(kar_marji*100, 1)}"
        }
    except:
        return {"FK": "Yok", "PD/DD": "Yok", "Kar Marji": "Yok"}

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
main_col, chat_col = st.columns([3,1])

st.sidebar.markdown("---")
st.sidebar.subheader("🔑 API Ayarları")
# Kullanıcıdan API anahtarını şifreli (yıldızlı) şekilde alıyoruz
kullanici_api_key = st.session_state.get("gemini_key","")
groq_api_key = st.session_state.get("groq_key","")
agresif_yorum=""

dl_bot = load_pytorch_model()

api_veri = st.session_state.api_status or {"gemini_valid": False, "groq_valid": False}
gr_durum = api_veri.get("groq_valid", False)
g_durum = api_veri.get("gemini_valid", False)
if not g_durum:
    st.sidebar.warning("⚠️ Gemini API Key Hatalı!")
else:
    st.sidebar.success("✅ Gemini Hazır")

if not gr_durum:
    st.sidebar.warning("ℹ️ Groq anahtarı hatalı: Agresif yorumcu modu pasif.")
else:
    # İŞTE EKSİK OLAN SATIR BU! Kod anahtarı bulduğunda bu yeşil yazıyı basacak.
    st.sidebar.success("✅ Groq Hazır")
with main_col:
    if secim== "Tek Hisse Analizi":
        sembol_input=st.text_input("Hisse ismini giriniz (Örn: THYAO, GARAN)")
        analiz_button=st.button("Analizi Başlat", type="primary")

        col1,col2=st.columns([3,1])

        with col1:    
            if analiz_button:
                if not sembol_input:
                    st.warning("Lütfen bir Hisse ismi girin!")
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
                        df, fib_20, fib_200=teknik_analiz(df)
                        
                        son_fiyat=df['Close'].iloc[-1]
                        son_sbs = df['SBS'].iloc[-1]
                        # --- ANALİZ ÖNCESİ VERİ TEMİZLİK ZIRHI ---
                        # Tüm NaN değerleri temizleyelim ki o meşhur hatayı bir daha görme
                        df = df.ffill().bfill().fillna(0) # NaN'ları doldur

                        gemini_bot=Gemini(api_key=kullanici_api_key)
                        
                        my_bar.progress(50, text="Pythorc sayısal tahmin yapıyor...")
                        
                        sonuc_dl=dl_bot.analiz_et(df)
                        
                        ai_rapor=f"Yön: {sonuc_dl.get('yön', 'Nötr')}, hedef: {sonuc_dl.get('tahmin', 0)} TL, güven: %{sonuc_dl.get('güven', 0)}"
                        
                        # 3. Güvenlik kilidi (Eğer tahmin 0 döndüyse kullanıcıyı uyar)
                        if sonuc_dl.get('yön') == "YETERSİZ VERİ":
                            st.warning("⚠️ Hisse verisi çok yeni veya eksik olduğu için Kahin (PyTorch) analiz yapamadı.")
                        
                        my_bar.progress(70, text="Gemini yorumunu hazırlıyor...")
                        haberler_listesi=haber_cek_web(clean_symbol)
                        df_kısa=df.tail(30)
                        temel = get_temel_hesapla(clean_symbol)

                        analiz_sonucu=gemini_bot(clean_symbol, temel, df_kısa, haberler_listesi, ai_rapor, fib_200, son_sbs)

                        if groq_api_key:
                            my_bar.progress(90, text="Groq analizi denetliyor...")
                            groq_bot = GroqDenetci(api_key=groq_api_key, model="llama-3.1-8b-instant")
                            agresif_yorum = groq_bot(df_kısa, analiz_sonucu,ai_rapor, fib_20, son_sbs)
                        else:
                            # Groq yoksa direkt Gemini sonucunu bas
                            agresif_yorum="ℹ️ Groq API anahtarı girilmediği için agresif yorum pasif."
                        
                        st.session_state.aktif_analiz_baglami = f"""
                            İncelenen Hisse: {clean_symbol}
                            Son Fiyat: {son_fiyat}₺
                            Kahin (PyTorch) Tahmini: {sonuc_dl.get('tahmin')}₺ (Yön: {sonuc_dl.get('yön')})
                            Gemini Analisti Yorumu: {analiz_sonucu}
                            Groq Denetçi Yorumu: {agresif_yorum}
                            """
                        
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
                                    
                    # PyThorc sonuçlarını metriklerde gösterirken renkli kutucuklar kullanalım
                    st.subheader(f"📊 {clean_symbol} Analiz Paneli")

                    c1,c2,c3,c4,c5=main_col.columns(5)
                    
                    c1.metric("Son Fiyat",f"{son_fiyat:.2f}₺")
                    
                    yon_emoji = "🔼" if sonuc_dl['yön'] == "YÜKSELİŞ" else "🔽"
                    c2.metric("PyThorc Hedefi", f"{sonuc_dl['tahmin']:.2f}₺", f"{yon_emoji} {sonuc_dl['yön']}")

                    c3.metric("RSI",f"{df['RSI'].iloc[-1]:.1f}")
                    c4.metric("MACD Sinyali", f"{df['MACD'].iloc[-1]:.2f}")
                    
                    # YENİ EKLENEN SBS METRİĞİ
                    sbs_delta = "Baskı Pozitif" if son_sbs > 50 else "-Baskı Negatif"
                    c5.metric("Alım Baskısı (SBS)", f"%{son_sbs:.1f}", sbs_delta)
                    
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

            for i, sembol in enumerate(bist100_hisseler):
                # İlerleme çubuğunu güncelle
                progress_bar.progress((i + 1) / len(bist100_hisseler), text=f"({i+1}/{len(bist100_hisseler)}) {sembol} analiz ediliyor...")
                
                clean_symbol, df, info = get_stock_data(sembol)
                
                try:
                    # Teknik analiz verilerini hesapla
                    df, fib_20, fib_200 = teknik_analiz(df)
                    son_sbs = df['SBS'].iloc[-1]
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
                # Float karşılaştırma güvenli yardımcı fonksiyon
                def sig(key, default=0):
                    val = son.get(key, default)
                    try:
                        return int(float(val))
                    except:
                        return default
                def val(key, default=0):
                    
                    v = son.get(key, default)
                    try:
                        return float(v)
                    except:
                        return default
                
                sbs = df['SBS'].iloc[-1]
                macd_sig=sig('MACD_signal')
                boll_sig=sig('BOLL_signal')
                vol_sig=sig('VOLUME_signal')
                rsi=val('RSI', 50)
                width=val('Width', 1)
                macd_val= val('MACD', 0)
                fiyat= val('Close', 0)
                sma50= val('SMA_50', 0)
                sma200= val('SMA_200', 0)

                # --- SİNYAL TANIMLARI ---
                # 🚀 Ralli: Momentum + Hacim + Bollinger onayı
                ralli = (
                    macd_sig == 1 and
                    boll_sig == 1 and
                    vol_sig == 1 and
                    sbs >= 55
                )
                # 💎 Wonderkid: Sıkışma + Patlama beklentisi (eşik gevşetildi)
                wonderkid = (
                    width < 0.25 and
                    rsi < 65 and
                    macd_val > 0 and
                    sbs > 45  # Ana trend pozitif olsun
                )
                # ⚠️ Erken Uyarı: MACD dönüyor + ya Bollinger ya Hacim onayı
                erken_uyari = (
                    macd_sig == 1 and
                    (boll_sig == 1 or vol_sig == 1) and
                    sbs > 40
                )
                # 📈 Trend Takipçi (YENİ): Fiyat her iki ortalamanın üstünde + RSI güçlü
                trend_takipci = (
                    fiyat > sma50 and
                    fiyat > sma200 and
                    rsi > 50 and rsi < 75 and
                    macd_val > 0 and
                    sbs >= 50
                )
                # 🔄 RSI Dip Dönüşü (YENİ): Aşırı satımdan çıkış
                rsi_donus = (
                    rsi < 45 and
                    macd_sig == 1 and
                    sbs > 35    # Momentum dönüyor
                )
                # 🔥 Agresif Para Girişi: Sadece devasa alım baskısı ve hacim (Bonus)
                sbs_patlama = (
                    sbs >= 70 and
                    vol_sig == 1
                )

                # Öncelik sırasına göre sinyal döndür
                if ralli:
                    return True, "🚀 Ralli Modu"
                elif wonderkid:
                    return True, "💎 Wonderkid (Sıkışma)"
                elif trend_takipci:
                    return True, "📈 Trend Takipçi"
                elif erken_uyari:
                    return True, "⚠️ Erken Uyarı"
                elif rsi_donus:
                    return True, "🔄 RSI Dip Dönüşü"
                elif sbs_patlama:
                    return True, "🚀 Alım/Satım oranı yükseldi"
                return False, "Temiz"

            except Exception as e:
                return False, f"Hata: {e}"

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
                    df, fib_20, fib_200 = teknik_analiz(df)                
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
                            
                            son_sbs = df['SBS'].iloc[-1]

                            sonuc_dl=dl_bot.analiz_et(df)
                            
                            dl_tahmin = 0 if pd.isna(sonuc_dl.get('tahmin', 0)) else sonuc_dl.get('tahmin', 0)
                            dl_guven = 0 if pd.isna(sonuc_dl.get('güven', 0)) else sonuc_dl.get('güven', 0)
                            
                            ai_rapor = f"Yön: {sonuc_dl.get('yön', 'Nötr')}, hedef: {sonuc_dl.get('tahmin', 0)} TL, güven: %{sonuc_dl.get('güven', 0)}"
                        
                            # Temel Analiz Verileri
                            temel = get_temel_hesapla(clean_symbol)
                            
                            # Haberler ve LLM Raporları
                            haberler_listesi = haber_cek_web(clean_symbol)
                            analiz_sonucu = gemini_bot(clean_symbol, temel, df, haberler_listesi, ai_rapor, fib_200, son_sbs)
                            agresif_yorum = groq_bot(df, analiz_sonucu,ai_rapor,fib_20, son_sbs)
                            
                            # Metrikleri Göster
                            c1, c2, c3, c4, c5 = st.columns(5)
                            c1.metric("Son Fiyat", f"{df['Close'].iloc[-1]:.2f}₺")
                            c2.metric("Durum", mesaj)
                            c3.metric("Kahin (DL) Hedef", f"{sonuc_dl.get('tahmin', 0)}₺")
                            c4.metric("Kahin Güveni", f"%{sonuc_dl.get('güven', 0)}")
                            
                            # YENİ EKLENEN SBS METRİĞİ
                            sbs_delta = "Baskı Pozitif" if son_sbs > 50 else "-Baskı Negatif"
                            c5.metric("Alım Baskısı (SBS)", f"%{son_sbs:.1f}", sbs_delta)
                            
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

    elif secim == "İzleme Listesi":
            watchlist_sayfasi(get_stock_data, teknik_analiz)

with chat_col:
    st.markdown("### 💬 Asistan")
    st.markdown("---")

    if "chat_gecmisi" not in st.session_state:
        st.session_state.chat_gecmisi = []

    for msg in st.session_state.chat_gecmisi:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # chat_input yerine text_input + button
    soru = st.text_input("Bir şey sor...", key="chat_input")
    if st.button("Gönder") and soru:
        st.session_state.chat_gecmisi.append({"role": "user", "content": soru})
        chat_bot = GroqChat(api_key=groq_api_key)
        cevap = chat_bot.generate(st.session_state.chat_gecmisi)
        st.session_state.chat_gecmisi.append({"role": "assistant", "content": cevap})
        st.rerun()