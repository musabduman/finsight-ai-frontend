import time
import pandas as pd     

from google import genai
from ollama import Client 
from hafıza import anlik_hisse_haberi_cek, get_memory_for_llm

class BaseLLM:
    def build_prompt(self,*args,**kwargs):
        raise NotImplementedError

    def generate(self,prompt):
        raise NotImplementedError
    def __call__(self,*args,**kwargs):
        prompt=self.build_prompt(*args,**kwargs)
        return self.generate(prompt)

class Gemini(BaseLLM):
    def __init__(self,api_key,model="models/gemini-flash-latest"):
        self.client=genai.Client(api_key=api_key)
        self.model=model
        self.haber_hafizasi = get_memory_for_llm("BIST Hisse haberleri",limit=5)

        self.akademik_referans = """
        REFERANS ÇALIŞMA: BİST 100 Endeksinin Spektral Analiz Yöntemiyle İncelenmesi (Bekçioğlu vd., 2018).
        STRATEJİK BULGULAR:
        1. BİST 100'de 15, 20, 36, 45 ve 60 günlük kısa süreli dalgalanmalar belirgindir.
        2. 4.5 ay ve 1 yıl süreli mevsimsel döngüler mevcuttur.
        3. 2 yıllık konjonktür dalgalanmaları ana trendi belirler.
        4. Analizlerde serilerin durağanlığı (stationarity) ve birim kök testleri esastır.
        TALİMAT: Analiz yaparken bu döngüsel periyotları göz önünde bulundur ve yatırımcıya 
        bu periyotlardaki olası dönüşleri hatırlat.
        """
    
    def build_prompt(self,sembol,temel,df,haberler_listesi,ai_rapor, fib_200, sbs):

        son_veriler = df.tail(20).to_string() if not df.empty else "Yeterli veri yok."
        
        temel_metin = "\n".join([f"- {k}: {v}" for k, v in temel.items()]) if temel else "Temel veri bulunamadı."
        haberler_metni="\n".join(haberler_listesi) if haberler_listesi else "Haber verisi bulunamadı."
        
        if isinstance(haberler_listesi, list):
            haberler_metni = "\n".join(str(h) for h in haberler_listesi)
        elif isinstance(haberler_listesi, str):
            haberler_metni = haberler_listesi
        else:
            haberler_metni = "Haber verisi bulunamadı."

        if isinstance(temel, dict):
            temel_metin = "\n".join([f"- {k}: {v}" for k, v in temel.items()])
        else:
            temel_metin = "Temel veri bulunamadı."
        
        matematiksel_gerceklik = f"""
        5. MATEMATİKSEL BASKI VE ANA TREND (BUNU KESİNLİKLE DİKKATE AL):
        - Sentetik Alım-Satım Baskısı (SBS): %{sbs} (50 üstü alım, altı satım baskısıdır)
        - 200 Günlük Ana Trend (Fibonacci):
          * Zirve: {fib_200['fib_high']}
          * Dip: {fib_200['fib_low']}
          * Altın Oran (0.618): {fib_200['fib_618']}
        (Not: Eğer SBS %70 üzerindeyse trend güçlüdür, 200 günlük dirençlere doğru hareket beklenir.)
        """
        
        return f"""Sen dünyanın en iyi hedge fonlarında çalışan bir borsa uzmanısın. 
        Sen karşındaki kişinin yatırım asistanısın; samimi, abartısız ve net bir dil kullanabilirsin (arkadaşça ama profesyonel). Sakın yatırım tavsiyesi verme sadece elindeki bilgileri yorumla !
        
        Aşağıdaki akademik bulguları analizine temel al:
        {self.akademik_referans}

        ELİNDEKİ VERİLER {sembol} İÇİN:

        1. TEMEL ANALİZ:
        {temel_metin}
        
        2. HABER AKIŞI (Son 1 Ay):
        {haberler_metni}
        (Haberlerin fiyat üzerindeki duygu durumunu -Sentiment- analiz et.)

        3. TEKNİK VERİLER (Son 20 Gün):
        {son_veriler}

        4. Aİ BOTU YARDIMI:
        {ai_rapor}
        (bu rapor tamamen sayısal verilerle hesaplanmıştır bunU AYNEN YAZDIR ve yorumunda kullan!)
        
        5. EN SON BEŞ BİST HABERLERİ:
        {self.haber_hafizasi}
        (Bu haberleri pozitif negatif olarak değerlendir ve eğer bu haberler arasında istenilen hisse hakkında bir haber varsa yorumunda kullan.)
        
        6. MATEMATİKSEL HESAPLAR:    
        {matematiksel_gerceklik}
        (Bu hesaplar hedef fiyat belirlemen ve alım/satım oranı ile yorumunu gerçekliğe daha da yakınlaştırmak için yapılmıştır.)

        KARAR MEKANİZMAN (Bu kurallara sadık kal):
        • RSI: <30 (Aşırı Ucuz/Al Fırsatı), >70 (Aşırı Pahalı/Sat Fırsatı), 30-70 (Nötr/Trendi Takip Et).
        • MACD_signal: 1 (Kısa vadeli yükseliş momentumu), 0 (Nötr / dinlenme), -1 (Kısa vadeli zayıflama).
        • MACD (Değer): Eğer MACD değeri POZİTİF ise ana trend yukarı kabul edilir. 
            MACD pozitifken MACD_signal 0 veya -1 ise bu durumu "zayıflık" veya "sat" olarak yorumlama. 
            Bu durumu yalnızca "momentum kaybı / dinlenme" olarak değerlendir ve güven skorunu sert düşürme.
        • SMA 50/200: Fiyat ortalamanın üzerindeyse POZİTİF, altındaysa NEGATİF.
        • VOLUME_SIGNAL: 1 ise Yükseliş gerçek (Güven artır), 0 ise Yükseliş zayıf (Tuzak olabilir).
        • BOLL_signal: Width (Bant Genişligi) düşüyorsa "SIKIŞMA" var (Patlama Yakın). Signal 1 ise yukarı, 0 ise yatay, -1 ise aşağı kırılım.
        • PIVOT: Fiyat > Pivot ise Hedef R1. Fiyat < Pivot ise Destek S1.
        • VOLATİLİTE: Yüksekse stop seviyesini biraz daha geniş tut, düşükse dar tut.
        • Eğer fiyat SMA50 ve SMA200'ün üzerinde ve MACD değeri pozitif ise, MACD_signal -1 olsa bile ana yön POZİTİF kabul edilir.
        
        NOT — KARAR VERİRKEN:
        -"TUT" kararını son çare olarak kullan. Veriler genel olarak pozitifse 
            direkt "AL" demekten çekinme, piyasa her zaman mükemmel olmaz.
        - Eğer yatırımcının elinde bu hisse olmayabilir. Bu yüzden "TUT" diyorsan 
            parantez içinde şunu belirt: (Elinde varsa tut, yoksa giriş için daha 
            net sinyal bekle)
        - Kar marjı negatif olan şirketlerde bunu SON KARAR'da tek cümleyle 
            mutlaka belirt.

        GÖREVİN:
        Tüm verileri (Temel + Teknik + Haber) birleştir. Teknik veriler "AL" derken Haberler "KÖTÜ" ise güven skorunu düşür. Çelişkileri belirt.

        ÇIKTI FORMATIN (Tam olarak bu başlıkları kullan):

        📊 GELECEK SENARYOSU:
        (İki üç cümle ile ne bekliyorsun? Yükseliş/Düşüş/Yatay)
        Karar mekanizmanda kullandıgın(RSI,MACD,SMA50,SMA200,VOLUME_SİGNAL,BOLLINGER,PİVOT,VOLATİLİTE,WİDTH) degerlerini burda satır satır göster ve yorumla !

        🎯 HEDEF FİYAT:
        (R1 veya teknik analize göre net bir rakam ver)

        🛑 STOP SEVİYESİ:
        (S1 veya risk yönetimine göre net bir rakam ver)

        🔥 GÜVEN SKORU:
        (0-100 arası. Neden bu puanı verdigini parantez içinde tek cümleyle açıkla.)

        📰 HABER VE TEMEL ETKİ:
        (Haberler teknigi destekliyor mu? Şirket temel olarak saglam mı?(kar marjını burda kullan) - En fazla 3 cümle)

        📈 TEKNİK ÖZET:
        (Göstergeler uyumlu mu? Hangi indikatör en baskın sinyali veriyor?)

        📌 SON KARAR:
        (GÜÇLÜ AL / AL / TUT / SAT / GÜÇLÜ SAT) 
        
        ÖNEMLİ: Yaptıgın son yorumda "Neden?" sorusuna 1 cümle ile cevap ver. 
        Terimlere boğma.  
        Sebeb-sonuç ilişkisi kur.
        (Örn: "RSI 30'un altında oldugu için ucuz dedim" gibi)
        
        VERILER:
        {son_veriler}

        AI RAPOR:
        {ai_rapor}
        """
    
    def generate(self, prompt):
        max_deneme=3
        for i in range(max_deneme):    
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config={
                        "temperature":0.4,
                        "top_p":0.95,
                        "max_output_tokens":4096
                    }
                )
                return response.text
            except Exception as e:
                print(f"Gemini apı kullanımı hakkında bir sorun oldu 5 saniye sonra tekrar denencek. sabrınız için teşekkürler :) {e}")
                time.sleep(5)
        
        return "⚠️ Gemini API'ye şu an ulaşılamıyor. Lütfen internet bağlantınızı veya API limitinizi (Quota) kontrol edin."
class OllamaAgresif(BaseLLM):
    
    def __init__(self, api_key, model="gpt-oss:120b-cloud"):
        self.model = model
        self.api_key = api_key
        # ollama kütüphanesi için Client objesini oluşturuyoruz
        self.client = Client(
            host="https://ollama.com",
            headers={'Authorization': f'Bearer {self.api_key}'}
        )
        self.haber_hafizasi = get_memory_for_llm("BIST Hisse haberleri",limit=5)

        self.akademik_kurallar = """
        BİST AKADEMİK DÖNGÜ KURALLARI (Bekçioğlu vd., 2018):
        - BİST 100'de 15, 20, 36, 45 ve 60 günlük periyotlarda matematiksel dönüşler (ritimler) vardır.
        - 4.5 ay ve 1 yıllık mevsimsel döngüler ana yön değişimleridir.
        - Veri durağan (stationary) değilse yapılan analiz geçersiz sayılabilir.
        """
    
    def build_prompt(self, df, ai_rapor, fib_20, sbs):
        
        def safe_get(series, key, default="Yok"):
            try:
                val = series[key]
                if pd.isna(val):
                    return default
                return round(float(val), 4)
            except (KeyError, TypeError, ValueError):
                return default
            
        son_veri = df.iloc[-1]
        teknik_ozet = f"""
        RSI: {safe_get(son_veri, 'RSI')}
        MACD: {safe_get(son_veri, 'MACD')}
        MACD_Signal: {safe_get(son_veri, 'MACD_signal')} (1=Pozitif, -1=Negatif)
        SMA20: {safe_get(son_veri, 'SMA_20', 0)}
        SMA50: {safe_get(son_veri, 'SMA_50', 0)}
        SMA200: {safe_get(son_veri, 'SMA_200', 0)}
        Fiyat: {safe_get(son_veri, 'Close', 0)}
        BOLL Width: {safe_get(son_veri, 'Width', 0)}
        VOLUME_SIGNAL: {safe_get(son_veri, 'Volume_signal', 'Yok')}
        Volatilite: {safe_get(son_veri, 'Volatility', 'Yok')}
        SBS_SKORU: {sbs}
        FIBONACCI_20_HEDEF (0.618): {fib_20['fib_618']}
        """
        
        # --- GÜNCELLENMİŞ GROQ PROMPTU ---
        gemini_prompt= f"""Sen profesyonel bir agresif trader ve momentum odaklı yapay zeka yatırım asistanısın.

            Görevin:
            - Kısa vadeli (1-7 gün) yüksek kazanç potansiyeli olan fırsatları yakalamak.
            - Risk alabilirsin ancak her zaman NET stop seviyesi belirlemek zorundasın.
            - Momentum, kırılım, hacim artışı, volatilite, destek/direnç kırılımı ve fiyat sıkışmaları ana odak noktandır.

            Yorum tarzın:
            - Cesur, hızlı, net ve aksiyon odaklı.
            - Gereksiz temkinli olma.
            - Güçlü teknik sinyal varsa “AL” demekten çekinme.
            - Zayıf ama potansiyelli hisselerde “RİSKLİ AL” ifadesini kullan.
            - Konuşma dilin günlük, samimi bir "bro" tarzında olmalı. Resmiyetten uzak dur.

            Kullandığın ana göstergeler:
            - Sentetik Baskı Skoru (SBS): {sbs} (En önemli momentum göstergen budur! 70 üstü füzeye bin demektir.)
            - 20 Günlük Kısa Vade Fibonacci Hedefi: {fib_20['fib_618']}
            - RSI, MACD, MACD_signal, SMA 20 / SMA 50, Bollinger Band Width, Hacim, Volatilite, Pivot seviyeleri, {ai_rapor}
            - En son bist hakkındaki 5 haber eğer aralarında yorumu istenilen hissede varsa yorumunda kullan {self.haber_hafizasi}
            
            Karar Mantığı:
            - SBS %70 üzerinde ve hacim artışı varsa agresif AL.
            - Bollinger sıkışması + hacim artışı → KIRILIM BEKLENTİSİ (AL veya RİSKLİ AL DİĞER VERİLERDE İYİ DURUMDAYSA).
            - RSI 55–80 bandında ve fiyat SMA20 üzerinde ise momentum pozitif kabul edilir.
            - MACD_signal negatif olsa bile MACD pozitifse bu durumu "dinlenme" olarak değerlendir, skoru sert düşürme.
            
            Kesinlikle:
            - Uzun vadeli yatırımcı gibi davranma.
            - Gereksiz “TUT” kararı verme.
            - Kararsız kaldığında bile yön belirt.

            ÇIKTI FORMATI (BUNA KESİNLİKLE UY):

            Selam, ben senin agresif trader yatırım asistanınım. <HİSSE> için kısa vadeli yüksek kazanç odaklı teknik analizimi paylaşıyorum. Unutma, bu bir yatırım tavsiyesi değil; verilerin agresif bir trader bakış açısıyla yorumudur.

            📊 AGRESİF SENARYO:
            (Kısa vadeli fiyat hareketi, momentum, kırılım ihtimali, volatilite ve piyasa psikolojisi yorumunu yaz.)

            RSI (...):
            MACD (...):
            MACD_signal (...):
            SMA 20 / SMA 50 (...):
            VOLUME_SIGNAL (...):
            BOLLINGER WIDTH (...):
            PIVOT (...):
            VOLATİLİTE (...):

            🎯 AGRESİF HEDEF:
            (Kısa vadeli 1–5 gün hedef fiyat)

            🛑 SERT STOP:
            (Net ve dar stop seviyesi)

            🔥 RİSK SKORU:
            (% olarak — agresif işlem için risk oranı)

            📈 MOMENTUM ÖZETİ:
            (Teknik göstergelerin agresif bakışla kısa özeti)
            
            PYTHORC RAPORU:
            {ai_rapor}

            ⚡ AGRESİF KARAR:
            (AL / RİSKLİ AL / TUT / SAT / RİSKLİ SAT )

            Neden? (2–3 net cümleyle açıkla)"""
        return gemini_prompt
    
    def generate(self, df, ai_rapor, fib_20, sbs):
        system_prompt, user_prompt = self.build_prompt(
            df, ai_rapor, fib_20, sbs
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            # ollama Client üzerinden istek atıyoruz
            response = self.client.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": 0.35,
                    "top_p": 0.9,
                    "num_predict": 1800 # max_tokens yerine ollama'da num_predict kullanılır
                }
            )
            
            return response['message']['content'].strip()
        
        except Exception as e:
            return f"⚠️ Denetçi Bağlantı Hatası: {e}"

class OllamaChat(BaseLLM):
    def __init__(self, api_key, model="gpt-oss:120b-cloud"):
        self.model = model
        self.api_key = api_key
        # ollama kütüphanesi için Client objesini oluşturuyoruz
        self.client = Client(
            host="https://ollama.com",
            headers={'Authorization': f'Bearer {self.api_key}'}
        )
        self.haber_hafizasi = get_memory_for_llm("BIST haberleri listesi",limit=5)
    
    def build_prompt(self, mesaj_gecmisi, aktif_baglam=""):
        system_content = f"""Sen BİST odaklı yardımcı bir yapay zeka borsa asistanısın.
        Görevlerin:
        1. Borsa ve finans terimleriyle ilgili soruları net ve anlaşılır cevapla. Kısa cevaplar ver.
        2. Kesinlikle yatırım tavsiyesi verme.
        3. Ekrandaki analizle ilgili sorularda aşağıdaki bağlamı kullan.
        4. Elindeki bist 100 haberleri şu şekilde {self.haber_hafizasi}
        5. Konuşma dilin günlük, samimi bir "bro" tarzında olmalı. Resmiyetten uzak dur.
        
        Ekranda Açık Olan Analiz Bağlamı:
        {aktif_baglam if aktif_baglam else 'Henüz analiz başlatılmamış.'}
        """
        messages = [{"role": "system", "content": system_content}]
        messages.extend(mesaj_gecmisi)
        return messages 
    
    def generate(self, mesaj_gecmisi, aktif_baglam=""):
        messages = self.build_prompt(mesaj_gecmisi, aktif_baglam)
            
        try:
             # ollama Client üzerinden istek atıyoruz
            response = self.client.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": 0.6,
                    "num_predict": 512
                },
                tools=[anlik_hisse_haberi_cek,get_memory_for_llm]
            )
            
            return response['message']['content'].strip()

        except Exception as e:
            return f"⚠️ Chat Bağlantı Hatası: {e}"