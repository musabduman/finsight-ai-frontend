import streamlit as st
from services.hafıza import save_to_memory, get_memory_for_llm
from services.haber import StockNewsFetcher

def haber_akisi():

    st.title("📰 Piyasa Gündemi")

    gundem = st.text_input(
        "Gündem:",
        value="Borsa İstanbul şirket gelişmeleri"
    )

    if st.button("Haberleri Getir", type="primary", use_container_width=True):

        with st.spinner("Haberler çekiliyor..."):
                fetcher = StockNewsFetcher()
                haberler = fetcher.get_news()

                if haberler:
                    save_to_memory(haberler)

        with st.spinner("Hafıza taranıyor..."):
            sonuc = get_memory_for_llm(gundem, limit=10)

        if not sonuc or "bulunamadı" in sonuc.lower():
            st.warning("Güncel veri bulunamadı.")
        else:
            st.success("Hazır.")
            st.markdown(sonuc)