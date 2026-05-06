# main.py
from indicators.technical import teknik_analiz
from services.veri import get_stock, get_temel_hesapla
from services.hafıza import get_memory_for_llm
from services.haber import anlik_hisse_haberi_cek


def tek_hisse_run(sembol, dl_bot, gemini_bot, ollama_bot):
    clean_symbol, df, info = get_stock(sembol)

    if df is None or df.empty:
        return {"error": "Veri alınamadı"}

    df, fib_20, fib_200 = teknik_analiz(df)

    son_fiyat = df["Close"].iloc[-1]
    son_sbs = df["SBS"].iloc[-1]

    df = df.ffill().bfill()

    dl_result = dl_bot.analiz_et(df)

    ai_rapor = f"Yön: {dl_result.get('yön','Nötr')}, hedef: {dl_result.get('tahmin',0)}, güven: %{dl_result.get('güven',0)}"

    haber = anlik_hisse_haberi_cek(clean_symbol)

    rag = get_memory_for_llm(
        query=f"{clean_symbol} haber",
        limit=5,
        hisse_filtresi=clean_symbol
    )

    haberler = f"""
    🔴 CANLI HABERLER:
    {haber}

    📚 RAG:
    {rag}
    """

    temel = get_temel_hesapla(clean_symbol)

    gemini_rapor = gemini_bot(clean_symbol, temel, df.tail(30), haberler, ai_rapor, fib_200, son_sbs)

    ollama_rapor = ollama_bot.generate(df.tail(30), ai_rapor, fib_20, son_sbs)

    return {
        "symbol": clean_symbol,
        "df": df,
        "son_fiyat": son_fiyat,
        "son_sbs": son_sbs,
        "dl": dl_result,
        "gemini": gemini_rapor,
        "ollama": ollama_rapor
    }