from google import genai
from groq import Groq
import time

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
    
    def build_prompt(self,sembol,temel,df,haberler_listesi,ai_rapor):

        son_veriler = df.tail(20).to_string() if not df.empty else "Yeterli veri yok."
        
        temel_metin = "\n".join([f"- {k}: {v}" for k, v in temel.items()]) if temel else "Temel veri bulunamadı."
        haberler_metni="\n".join(haberler_listesi) if haberler_listesi else "Haber verisi bulunamadı."
        
        #ÖNEMLİ: Yaptıgın son yorumda "Neden?" sorusuna cevap ver. Terimlere bogmadan, çokta uzatmadan, sonucun hangi veriden kaynaklandıgını açıkla. (Örn: "RSI 30'un altında oldugu için ucuz dedim" gibi).
        return f"""Sen dünyanın en iyi hedge fonlarında çalışan bir borsa uzmanısın. 
        Sen karşındaki kişinin yatırım asistanısın; samimi, abartısız ve net bir dil kullanabilirsin (arkadaşça ama profesyonel). Sakın yatırım tavsiyesi verme sadece elindeki bilgileri yorumla !
    
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

        KARAR MEKANİZMAN (Bu kurallara sadık kal):
        • RSI: <30 (Aşırı Ucuz/Al Fırsatı), >70 (Aşırı Pahalı/Sat Fırsatı), 30-70 (Nötr/Trendi Takip Et).
        • MACD: 1 (Al/Yükseliş), -1 (Sat/Düşüş).
        • SMA 50/200: Fiyat ortalamanın üzerindeyse POZİTİF, altındaysa NEGATİF.
        • VOLUME_SIGNAL: 1 ise Yükseliş gerçek (Güven artır), 0 ise Yükseliş zayıf (Tuzak olabilir).
        • BOLLINGER: Width (Bant Genişligi) düşüyorsa "SIKIŞMA" var (Patlama Yakın). Signal 1 ise yukarı, 0 ise yatay.
        • PIVOT: Fiyat > Pivot ise Hedef R1. Fiyat < Pivot ise Destek S1.
        • VOLATİLİTE: Yüksekse stop seviyesini biraz daha geniş tut, düşükse dar tut.

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
                print(f"Gemini apı kullanımı hakkında bir sorun oldu 5 saniye sonra tekrar denencek. sabrınız için teşekkürler :)")
                time.sleep(5)

class GroqDenetci(BaseLLM):
    
    def __init__(self,api_key,model="llama-3.1-8b-instant"):
        self.model=model
        self.client=Groq(api_key=api_key)
    @staticmethod
    def groq_safe(text):
        if not isinstance(text, str):
            return text
        return text
    
    def build_prompt(self,df,analiz_sonucu):

        son_veri = df.iloc[-1]
        teknik_ozet = f"""
        RSI: {son_veri.get('RSI', 'Yok')}
        MACD_Signal: {son_veri.get('MACD_signal', 'Yok')} (1=Pozitif, -1=Negatif)
        Fiyat: {son_veri.get('Close', 0)}
        SMA200: {son_veri.get('SMA_200', 0)}
        """
        
        analiz_sonucu_safe = self.groq_safe(analiz_sonucu)

        # --- GÜNCELLENMİŞ GROQ PROMPTU ---
        return f"""SEN BİR DENETÇİSİN (AUDITOR).
        GÖREVİN: Aşağıdaki 'Gemini Raporu'nu, 'Gerçek Teknik Veriler' ile karşılaştırıp YALAN SÖYLÜYOR MU kontrol etmek.

        GERÇEK TEKNİK VERİLER:
        {teknik_ozet}

        GEMINI RAPORU:
        {analiz_sonucu_safe}

        KURALLAR:
        1. Eğer RSI 70'in üzerindeyse ve Gemini "Ucuz" diyorsa -> BU BİR HATADIR.
        2. Eğer MACD Sinyali -1 (Negatif) ise ve Gemini "Yükseliş trendi" diyorsa -> BU BİR HATADIR.
        3. Eğer Fiyat, SMA200'ün altındaysa ve Gemini "Boğa piyasası" diyorsa -> BU BİR HATADIR.
        4. SAYILARIN KÜSÜRATLARINA KARIŞMA ONLARI KISATABİLİR GEMİNİ. SAYININ KÜSÜRLERİNİ HATA SAYMA.
        
        CEVAP FORMATI (BU KURALLARA KESİNLİKLE UY):
        - Eğer hata yoksa KESİNLİKLE açıklama yapma, madde madde sayma, özet geçme veya süreçten bahsetme. SADECE tek bir satır halinde şunu yaz: "✅ Analiz Onaylandı."
        - Eğer hata varsa SADECE hatanın ne olduğunu tek bir cümleyle yaz: "⚠️ HATA TESPİT EDİLDİ: [Neden]"
        
        UNUTMA: Gevezelik etme, boş konuşma, sadece net sonucu ver!
        """
    
    def generate(self, prompt):
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            return f"⚠️ Denetçi Bağlantı Hatası: {e}"