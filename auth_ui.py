import streamlit as st
import requests
import re
import time

# Backend URL'niz (api.py hangi adreste çalışıyorsa)
BASE_URL = "https://finsight-ai-backend-u1cw.onrender.com"


def login_sidebar():
    st.sidebar.title("🔐 Üye Sistemi")

    # --- 1. OTURUM HAFIZASINI İLKLENDİRME (F5 Sorununu Çözen Kısım) ---
    # Eğer bu anahtarlar hafızada yoksa oluşturuyoruz.
    # 'not in' kullanımı sayesinde sayfa yenilense bile True olan değer False'a dönmez.
    if 'username' not in st.session_state:
        st.session_state.username = ""
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'verified' not in st.session_state:
        st.session_state.verified = False
    if 'user_email' not in st.session_state:
        st.session_state.user_email = ""
    if 'awaiting_verification' not in st.session_state:
        st.session_state.awaiting_verification = False
    if 'verify_email' not in st.session_state:
        st.session_state.verify_email = ""

    # --- 2. GİRİŞ YAPILMAMIŞSA: GİRİŞ/KAYIT FORMU ---
    if not st.session_state.logged_in:
        menu = st.sidebar.tabs(["Giriş Yap", "Kayıt Ol"])

        # --- GİRİŞ YAP SEKMESİ ---
        with menu[0]:
            k_bilgi = st.text_input("E-posta ya da Kullanıcı Adı", key="login_email")
            pw = st.text_input("Şifre", type="password", key="login_pass")

            if st.button("Oturum Aç"):
                if k_bilgi and pw:
                    try:
                        # 'email' yerine 'kullanici_bilgisi' yolluyoruz
                        res = requests.post(f"{BASE_URL}/login", json={"kullanici_bilgisi": k_bilgi, "password": pw})

                        if res.status_code == 200:
                            data = res.json()
                            st.session_state.logged_in = True
                            st.session_state.verified = True
                            # Veritabanından gelen bilgileri hafızaya al
                            st.session_state.user_email = data.get("email")
                            st.session_state.username = data.get("username")  # İŞTE BURASI!

                            st.sidebar.success("✅ Giriş Başarılı!")
                            st.rerun()  # Sayfayı hemen tazele ki main.py değişikliği görsün
                        else:
                            st.sidebar.error("❌ Hatalı bilgi veya şifre!")
                    except requests.exceptions.ConnectionError:
                        st.sidebar.error("❌ API Sunucusuna ulaşılamıyor! (api.py açık mı?)")
                else:
                    st.sidebar.warning("⚠️ Lütfen alanları doldurun.")

        # --- KAYIT OL SEKMESİ ---
        with menu[1]:
            n_user = st.text_input("Kullanıcı Adı", key="reg_user")
            n_email = st.text_input("E-posta", key="reg_email")
            n_pw = st.text_input("Şifre", type="password", key="reg_pass")

            # --- DİNAMİK ŞİFRE KONTROLÜ BAŞLIYOR ---
            uzunluk_ok = len(n_pw) >= 8
            buyuk_ok = bool(re.search(r'[A-Z]', n_pw))
            kucuk_ok = bool(re.search(r'[a-z]', n_pw))
            sayi_ok = bool(re.search(r'\d', n_pw))
            ozel_ok = bool(re.search(r'[!@#$%^&*(),.?":{}|<>]', n_pw))

            # Tüm şartlar sağlandı mı?
            sifre_gecerli = uzunluk_ok and buyuk_ok and kucuk_ok and sayi_ok and ozel_ok

            # Şartları şifre kutusunun hemen altında alt alta liste şeklinde gösteriyoruz
            st.markdown(f"""
                <div style="font-size: 14px; color: #888;">
                    {'✅' if uzunluk_ok else '❌'} En az 8 karakter <br>
                    {'✅' if buyuk_ok else '❌'} En az 1 büyük harf <br>
                    {'✅' if kucuk_ok else '❌'} En az 1 küçük harf <br>
                    {'✅' if sayi_ok else '❌'} En az 1 rakam <br>
                    {'✅' if ozel_ok else '❌'} En az 1 özel karakter (!@#$ vb.)
                </div>
                """, unsafe_allow_html=True)

            n_pw2 = st.text_input("Şifre (Tekrar)", type="password", key="reg_pass2")
            gemini = st.text_input( label="Gemini API Key",
                    type="password",
                    help="Google AI Studio'dan alabilirsiniz."  # <-- İşte o soru işaretini çıkaran sihirli kod
            )
            groq = st.text_input(
                label="Groq API Key (Agresif Yorumcu)",
                type="password",
                help="Groq'un kendi sitesinden alabilirsiniz." # <-- Groq için olan ipucu
            )

            if st.button("Hesabı Oluştur"):
                if not sifre_gecerli:
                    st.error("⚠️ Lütfen şifrenizin yukarıdaki tüm güvenlik şartlarını sağladığından emin olun.")
                elif n_pw != n_pw2:
                    st.warning("⚠️ Şifreler uyuşmuyor!")
                elif not n_user or not n_email:
                    st.warning("⚠️ Lütfen kullanıcı adı ve e-posta alanlarını doldurun.")
                else:
                    # 1. Payload SADECE her şey doğruysa burada yaratılıyor
                    payload = {
                        "username": n_user,
                        "email": n_email,
                        "password": n_pw,
                        "api_key": gemini,  # Kendi kodundaki değişkenlerin
                        "groq_api_key": groq
                    }

                    # 2. İstek de TAM OLARAK burada, payload'ın hemen altında atılıyor
                    try:
                        res = requests.post(f"{BASE_URL}/register", json=payload)

                        if res.status_code == 200:
                            st.session_state.awaiting_verification = True
                            st.session_state.verify_email = n_email
                            st.success("✅ Kayıt Başarılı! Lütfen terminale düşen kodu giriniz.")
                            time.sleep(2)
                            st.rerun()
                        else:
                            hata_mesaji = res.json().get("detail", "Bilinmeyen bir hata.")
                            st.error(hata_mesaji)
                    except Exception as e:
                        st.error(f"❌ Bağlantı hatası: {e}")
    # --- 3. GİRİŞ YAPILMIŞSA: HOŞ GELDİN VE ÇIKIŞ ---
    else:
        # --- KULLANICI GİRİŞ YAPMIŞ DURUMDA ---
        st.sidebar.success(f"✅ Hoş geldin:\n{st.session_state.username}")

        # API Kontrol Paneli
        st.sidebar.markdown("---")
        # API DURUM PANELİ
        st.sidebar.subheader("🔌 API Bağlantı Durumu")

        if 'api_status' not in st.session_state or st.session_state.api_status is None:
            with st.sidebar.status("API'ler test ediliyor...", expanded=True) as status:
                try:
                    res = requests.post(f"{BASE_URL}/check_api_keys", json={"email": st.session_state.user_email})
                    if res.status_code == 200:
                        st.session_state.api_status = res.json()
                    else:
                        st.session_state.api_status = {"gemini_valid": False, "groq_valid": False}
                except:
                    st.session_state.api_status = {"gemini_valid": False, "groq_valid": False}

                # Güvenli veri çekimi
                api_veri = st.session_state.api_status or {"gemini_valid": False, "groq_valid": False}
                g_durum = api_veri.get("gemini_valid", False)
                gr_durum = api_veri.get("groq_valid", False)

                # BUNLAR KUTUNUN İÇİNDE (with bloğunda) OLDUĞU İÇİN YANA GİDER
                st.write(f"**{'🟢' if g_durum else '🔴'} Gemini API**")
                st.write(f"**{'🟢' if gr_durum else '🔴'} Groq API**")

                status.update(label="Test tamamlandı!", state="complete", expanded=False)
        else:
            # Hafızadaki sonuçları göster
            api_veri = st.session_state.api_status or {"gemini_valid": False, "groq_valid": False}
            g_durum = api_veri.get("gemini_valid", False)
            gr_durum = api_veri.get("groq_valid", False)

            with st.sidebar.status("Test tamamlandı!", state="complete", expanded=False):
                # BUNLAR DA KUTUNUN İÇİNDE!
                st.write(f"**{'🟢' if g_durum else '🔴'} Gemini API**")
                st.write(f"**{'🟢' if gr_durum else '🔴'} Groq API**")

        # Uyarı mesajı eğer anahtarlar boşsa (Kutunun altında kalır ama yan menüde olur)
        if not st.session_state.api_status or not st.session_state.api_status.get("gemini_valid"):
            st.sidebar.warning("⚠️ Bazı API anahtarlarınız geçersiz veya boş. Analiz işlemleri hata verebilir.")


        st.sidebar.markdown("---")

        # Çıkış Butonu
        if st.sidebar.button("Çıkış Yap", key="essiz_cikis_butonu"):
            st.session_state.logged_in = False
            st.session_state.verified = False
            st.session_state.user_email = ""
            st.session_state.username = ""
            # Çıkış yaparken API test hafızasını da siliyoruz ki başka hesaba girince onunki baştan test edilsin
            if 'api_status' in st.session_state:
                del st.session_state['api_status']
            st.rerun()

        return True  # En sona bunu eklemeyi unutma

    # --- 4. SONUÇ DÖNDÜR ---
    # main.py bu sonucu kullanarak ana paneli açar veya açmaz.
    return st.session_state.logged_in and st.session_state.verified