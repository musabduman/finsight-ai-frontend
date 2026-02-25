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
    
    def build_prompt(self,sembol,temel,df,haberler_listesi,ai_rapor):

        son_veriler = df.tail(20).to_string() if not df.empty else "Yeterli veri yok."
        
        temel_metin = "\n".join([f"- {k}: {v}" for k, v in temel.items()]) if temel else "Temel veri bulunamadı."
        haberler_metni="\n".join(haberler_listesi) if haberler_listesi else "Haber verisi bulunamadı."
        

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
class GroqDenetci(BaseLLM):
    
    def __init__(self,api_key,model="llama-3.1-8b-instant"):
        self.model=model
        self.client=Groq(api_key=api_key)
        
        self.akademik_kurallar = """
        BİST AKADEMİK DÖNGÜ KURALLARI (Bekçioğlu vd., 2018):
        - BİST 100'de 15, 20, 36, 45 ve 60 günlük periyotlarda matematiksel dönüşler (ritimler) vardır.
        - 4.5 ay ve 1 yıllık mevsimsel döngüler ana yön değişimleridir[cite: 607].
        - Veri durağan (stationary) değilse yapılan analiz geçersiz sayılabilir[cite: 110, 118].
        """
    
    def build_prompt(self,df,analiz_sonucu,ai_rapor):

        son_veri = df.iloc[-1]
        teknik_ozet = f"""
        RSI: {son_veri.get('RSI', 'Yok')}
        MACD_Signal: {son_veri.get('MACD_signal', 'Yok')} (1=Pozitif, -1=Negatif)
        Fiyat: {son_veri.get('Close', 0)}
        SMA200: {son_veri.get('SMA_200', 0)}
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

            Kullandığın ana göstergeler:
            - RSI
            - MACD
            - MACD_signal
            - SMA 20 / SMA 50
            - Bollinger Band Width
            - Hacim (Volume & Volume Signal)
            - Volatilite
            - Pivot seviyeleri
            - Ai raporu {ai_rapor}
            Karar Mantığı:
            - MACD pozitif ve hacim artışı varsa agresif AL.
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
        user_content = f"TEKNİK VERİ: {teknik_ozet}\n\nGEMINI RAPORU: {analiz_sonucu}"
        return f"{gemini_prompt}\n\n{user_content}"
    
    def generate(self, prompt):
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=0.4, # Mallığı bitiren altın ayar burası!
                max_tokens=4096 # Cevabı kısa tutmaya zorluyoruz
            )
            res = chat_completion.choices[0].message.content.strip()
            return res.split('\n')[0]
        
        except Exception as e:
            return f"⚠️ Denetçi Bağlantı Hatası: {e}"