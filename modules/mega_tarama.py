import time
import pandas as pd
import streamlit as st

from services.veri import get_bulk_stocks, normalize_symbol
from indicators.technical import teknik_analiz


def mega_tarama(bist100_hisseler, dl_bot):

    st.subheader("📊 BIST100 Hızlı Yapay Zeka Taraması")
    st.markdown("Bu bölüm, BIST100 hisselerini hızlıca tarar ve yapay zeka destekli analizler sunar. **Not:** Bu işlem biraz zaman alabilir, lütfen sabırlı olun.")

    col1, col2 = st.columns([1, 1])

    # -------------------------------
    # 🚀 BAŞLAT
    # -------------------------------
    with col1:
        if st.button("Mega Taramayı Başlat", type="primary"):

            if "mega_tarama_sonuc" not in st.session_state:

                progress = st.progress(0, text="Veriler çekiliyor...")

                # 🔥 EN KRİTİK KISIM (TEK REQUEST)
                bulk_data = get_bulk_stocks(bist100_hisseler)

                if bulk_data is None:
                    st.error("Veri çekilemedi (muhtemelen rate limit).")
                    return

                sonuclar = []

                for i, sembol in enumerate(bist100_hisseler):

                    progress.progress(
                        (i + 1) / len(bist100_hisseler),
                        text=f"{sembol} analiz ediliyor..."
                    )

                    try:
                        clean = normalize_symbol(sembol)

                        if clean not in bulk_data:
                            continue

                        df = bulk_data[clean].copy()

                        if df.empty:
                            continue

                        df, _, _ = teknik_analiz(df)
                        df = df.ffill().bfill()

                        # --- DL MODEL ---
                        df_model = df[['Open','High','Low','Close','Volume']].dropna()

                        try:
                            sonuc = dl_bot.analiz_et(df_model)
                            yon = sonuc.get('yön', 'Nötr')
                            hedef = sonuc.get('tahmin', 0)
                            guven = sonuc.get('güven', 0)
                        except:
                            yon, hedef, guven = "Hata", 0, 0

                        # --- METRİKLER ---
                        son_fiyat = float(df['Close'].iloc[-1])
                        rsi = float(df['RSI'].iloc[-1]) if 'RSI' in df else 0
                        macd = float(df['MACD'].iloc[-1]) if 'MACD' in df else 0

                        sonuclar.append({
                            "Hisse": clean.replace(".IS", ""),
                            "Fiyat": round(son_fiyat, 2),
                            "Yön": yon,
                            "Hedef": round(hedef, 2),
                            "Güven": round(guven, 2),
                            "RSI": round(rsi, 2),
                            "MACD": round(macd, 2)
                        })

                    except Exception as e:
                        continue

                progress.empty()

                st.session_state.mega_tarama_sonuc = sonuclar

    # -------------------------------
    # 🗑️ RESET
    # -------------------------------
    with col2:
        if "mega_tarama_sonuc" in st.session_state:
            if st.button("🗑️ Sıfırla"):
                del st.session_state.mega_tarama_sonuc
                st.rerun()

    # -------------------------------
    # 📊 TABLO
    # -------------------------------
    if "mega_tarama_sonuc" in st.session_state:

        data = st.session_state.mega_tarama_sonuc

        if data:
            st.success(f"{len(data)} hisse analiz edildi")

            df = pd.DataFrame(data)

            def yon_icon(y):
                y = str(y).upper()
                if "YÜKSELİŞ" in y:
                    return f"🟢 {y}"
                elif "DÜŞÜŞ" in y:
                    return f"🔴 {y}"
                return f"🟡 {y}"

            def guven_icon(g):
                try:
                    g = float(g)
                    if g >= 80:
                        return f"🔥 %{g}"
                    elif g >= 60:
                        return f"👍 %{g}"
                    return f"⚠️ %{g}"
                except:
                    return g

            df["Yön"] = df["Yön"].apply(yon_icon)
            df["Güven"] = df["Güven"].apply(guven_icon)

            st.dataframe(df, width="stretch", hide_index=True)