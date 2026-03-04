# watchlist_page.py
# Bu fonksiyonu app.py'ye import et ve secim=="İzleme Listesi" durumunda cagir

import streamlit as st
import pandas as pd
import time

# app.py'deki mevcut fonksiyonlari kullanacagiz
# from app import get_stock_data, get_fast_info, teknik_analiz seklinde import edilmeli

def normalize_symbol(symbol: str):
    tr_to_en = str.maketrans("ıiğüşöçIİĞÜŞÖÇ", "IIGUSOCIIGUSOC")
    clean_symbol = str(symbol).translate(tr_to_en).upper().strip()
    if not clean_symbol.endswith(".IS"):
        clean_symbol += ".IS"
    return clean_symbol

def watchlist_sayfasi(get_stock_data, teknik_analiz):
    """
    Kullanim:
        app.py icinde:
        from watchlist_page import watchlist_sayfasi
        ...
        elif secim == "Izleme Listesi":
            watchlist_sayfasi(get_stock_data, get_temel_info, teknik_analiz)
    """

    st.subheader("İzleme Listesi")
    st.markdown("Takip etmek istediğin hisseleri buraya ekle, anlık durumlarını tek bakışta gör.")

    # ── SESSION STATE BASLAT ──────────────────────────────────────────
    if "watchlist" not in st.session_state:
        st.session_state.watchlist = []   # ["THYAO.IS", "GARAN.IS", ...]

    # ── HISSE EKLEME FORMU ───────────────────────────────────────────
    col_ekle, col_btn = st.columns([4, 1])

    with col_ekle:
        yeni_hisse = st.text_input(
            "Hisse Ekle",
            placeholder="Örn: THYAO, GARAN",
            label_visibility="collapsed",
            key="watchlist_input"
        )
    with col_btn:
        if st.button("➕ Ekle", use_container_width=True):
            if yeni_hisse:
                temiz = normalize_symbol(yeni_hisse)
                if temiz not in st.session_state.watchlist:
                    st.session_state.watchlist.append(temiz)
                    st.success(f"{temiz} listeye eklendi!")
                else:
                    st.warning(f"{temiz} zaten listede.")
            else:
                st.warning("Hisse adı gir.")

    # ── LİSTE BOŞSA UYARI ────────────────────────────────────────────
    if not st.session_state.watchlist:
        st.info("📭 İzleme listen boş. Yukarıdan hisse ekle.")
        return

    # ── YENİLE BUTONU ────────────────────────────────────────────────
    col_yenile, col_bos = st.columns([1, 4])
    with col_yenile:
        yenile = st.button("🔄 Fiyatları Yenile", use_container_width=True)

    st.markdown("---")

    # ── HİSSELERİ TARA VE GÖSTER ─────────────────────────────────────
    silinecek = None  # döngü içinde silme yapmak yerine döngü sonrası sil

    for sembol in st.session_state.watchlist:

        col_sembol, col_fiyat, col_degisim, col_rsi, col_durum, col_sil = st.columns(
            [2, 1.5, 1.5, 1, 2, 0.8]
        )

        with col_sembol:
            st.subheader({sembol.replace('.IS','')})

        try:
            clean_symbol, df, info = get_stock_data(sembol)

            if df is None or df.empty:
                with col_fiyat:
                    st.error("Veri yok")
            else:
                df = teknik_analiz(df)
                df = df.ffill().bfill().fillna(0)

                son_fiyat   = float(df['Close'].iloc[-1])
                onceki_gun  = float(df['Close'].iloc[-2]) if len(df) > 1 else son_fiyat
                degisim_yuz = ((son_fiyat - onceki_gun) / onceki_gun) * 100
                rsi_degeri  = float(df['RSI'].iloc[-1]) if 'RSI' in df.columns else 0

                # Fiyat
                with col_fiyat:
                    st.metric("Fiyat", f"{son_fiyat:.2f}₺")

                # Degisim
                with col_degisim:
                    renk = "normal"
                    ok   = "▲" if degisim_yuz >= 0 else "▼"
                    st.metric("Değişim", f"{ok} %{abs(degisim_yuz):.2f}")

                # RSI
                with col_rsi:
                    if rsi_degeri < 30:
                        rsi_label = "🟢"
                    elif rsi_degeri > 70:
                        rsi_label = "🔴"
                    else:
                        rsi_label = "🟡"
                    st.metric("RSI", f"{rsi_label} {rsi_degeri:.1f}")

                # Kisa durum yorumu
                with col_durum:
                    macd_signal = int(df['MACD_signal'].iloc[-1]) if 'MACD_signal' in df.columns else 0
                    macd_val    = float(df['MACD'].iloc[-1])      if 'MACD'        in df.columns else 0

                    if rsi_degeri < 30 and macd_val > 0:
                        durum = "🟢 Güçlü Al Bölgesi"
                    elif rsi_degeri > 70 and macd_val < 0:
                        durum = "🔴 Aşırı Alım"
                    elif macd_signal == 1:
                        durum = "📈 Momentum Pozitif"
                    elif macd_signal == -1:
                        durum = "📉 Momentum Negatif"
                    else:
                        durum = "➖ Nötr"

                    st.markdown(f"<br><small>{durum}</small>", unsafe_allow_html=True)

        except Exception as e:
            with col_fiyat:
                st.error(f"Hata: {e}")

        # Sil butonu
        with col_sil:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🗑️", key=f"sil_{sembol}", help=f"{sembol} listeden çıkar"):
                silinecek = sembol

        st.divider()

    # Döngü bittikten sonra sil (döngü içinde liste değiştirme hatası önlenir)
    if silinecek:
        st.session_state.watchlist.remove(silinecek)
        st.success(f"{silinecek} listeden çıkarıldı.")
        st.rerun()

    # ── LİSTEYİ TEMIZLE ──────────────────────────────────────────────
    st.markdown("---")
    if st.button("🗑️ Listeyi Tamamen Temizle", type="secondary"):
        st.session_state.watchlist = []
        st.rerun()