import streamlit as st
import requests
import re
import time

BASE_URL = "https://finsight-ai-backend-u1cw.onrender.com"


def login_sidebar():
    st.sidebar.title("🔐 Üye Sistemi")

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
    if 'gemini_key' not in st.session_state:
        st.session_state.gemini_key = ""
    if 'ollama_key' not in st.session_state:
        st.session_state.ollama_key = ""

    if not st.session_state.logged_in:
        menu = st.sidebar.tabs(["Giriş Yap", "Kayıt Ol"])

        with menu[0]:
            k_bilgi = st.text_input("E-posta ya da Kullanıcı Adı", key="login_email")
            pw = st.text_input("Şifre", type="password", key="login_pass")

            if st.button("Oturum Aç"):
                if k_bilgi and pw:
                    try:
                        res = requests.post(f"{BASE_URL}/login", json={"kullanici_bilgisi": k_bilgi, "password": pw})

                        if res.status_code == 200:
                            data = res.json()
                            st.session_state.logged_in = True
                            st.session_state.verified = True
                            st.session_state.user_email = data.get("email")
                            st.session_state.username = data.get("username")

                            # --- KEY'LERİ BACKEND'DEN ÇEK VE SESSION'A YAZ ---
                            try:
                                key_res = requests.get(f"{BASE_URL}/get_keys/{data.get('email')}")
                                if key_res.status_code == 200:
                                    keys = key_res.json()
                                    st.session_state.gemini_key = keys.get("gemini_key", "")
                                    st.session_state.ollama_key = keys.get("ollama_key", "")
                            except Exception:
                                pass  # key çekilemezse uygulama yine açılır, uyarı app.py'de gösterilir

                            st.sidebar.success("✅ Giriş Başarılı!")
                            st.rerun()
                        else:
                            st.sidebar.error("❌ Hatalı bilgi veya şifre!")
                    except requests.exceptions.ConnectionError:
                        st.sidebar.error("❌ API Sunucusuna ulaşılamıyor!")
                else:
                    st.sidebar.warning("⚠️ Lütfen alanları doldurun.")

        with menu[1]:
            n_user = st.text_input("Kullanıcı Adı", key="reg_user")
            n_email = st.text_input("E-posta", key="reg_email")
            n_pw = st.text_input("Şifre", type="password", key="reg_pass")

            uzunluk_ok = len(n_pw) >= 8
            buyuk_ok = bool(re.search(r'[A-Z]', n_pw))
            kucuk_ok = bool(re.search(r'[a-z]', n_pw))
            sayi_ok = bool(re.search(r'\d', n_pw))
            ozel_ok = bool(re.search(r'[!@#$%^&*(),.?":{}|<>]', n_pw))

            sifre_gecerli = uzunluk_ok and buyuk_ok and kucuk_ok and sayi_ok and ozel_ok

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
            gemini = st.text_input(label="Gemini API Key", type="password", help="Google AI Studio'dan alabilirsiniz.")
            ollama = st.text_input(label="Ollama API Key (Agresif Yorumcu)", type="password", help="Ollama.com'dan alabilirsiniz.")

            if st.button("Hesabı Oluştur"):
                if not sifre_gecerli:
                    st.error("⚠️ Şifreniz güvenlik şartlarını sağlamıyor.")
                elif n_pw != n_pw2:
                    st.warning("⚠️ Şifreler uyuşmuyor!")
                elif not n_user or not n_email:
                    st.warning("⚠️ Lütfen tüm alanları doldurun.")
                else:
                    payload = {
                        "username": n_user,
                        "email": n_email,
                        "password": n_pw,
                        "api_key": gemini,
                        "ollama_api_key": ollama
                    }
                    try:
                        res = requests.post(f"{BASE_URL}/register", json=payload)
                        if res.status_code == 200:
                            st.session_state.awaiting_verification = True
                            st.session_state.verify_email = n_email
                            st.success("✅ Kayıt Başarılı!")
                            time.sleep(2)
                            st.rerun()
                        else:
                            hata_mesaji = res.json().get("detail", "Bilinmeyen bir hata.")
                            st.error(hata_mesaji)
                    except Exception as e:
                        st.error(f"❌ Bağlantı hatası: {e}")
    else:
        st.sidebar.success(f"✅ Hoş geldin:\n{st.session_state.username}")

        st.sidebar.markdown("---")
        st.sidebar.subheader("🔌 API Bağlantı Durumu")

        if 'api_status' not in st.session_state or st.session_state.api_status is None:
            with st.sidebar.status("API'ler test ediliyor...", expanded=True) as status:
                try:
                    res = requests.post(f"{BASE_URL}/check_api_keys", json={"email": st.session_state.user_email})
                    if res.status_code == 200:
                        st.session_state.api_status = res.json()
                    else:
                        st.session_state.api_status = {"gemini_valid": False, "ollama_valid": True}
                except:
                    st.session_state.api_status = {"gemini_valid": False, "ollama_valid": True}

                api_veri = st.session_state.api_status or {"gemini_valid": False, "ollama_valid": True}
                g_durum = api_veri.get("gemini_valid", False)
                gr_durum = api_veri.get("ollama_valid", True)

                st.write(f"**{'🟢' if g_durum else '🔴'} Gemini API**")
                st.write(f"**{'🟢' if gr_durum else '🔴'} Ollama API**")

                status.update(label="Test tamamlandı!", state="complete", expanded=False)
        else:
            api_veri = st.session_state.api_status or {"gemini_valid": False, "ollama_valid": True}
            g_durum = api_veri.get("gemini_valid", False)
            gr_durum = api_veri.get("ollama_valid", True)

            with st.sidebar.status("Test tamamlandı!", state="complete", expanded=False):
                st.write(f"**{'🟢' if g_durum else '🔴'} Gemini API**")
                st.write(f"**{'🟢' if gr_durum else '🔴'} Ollama API**")

        if not st.session_state.api_status or not st.session_state.api_status.get("gemini_valid"):
            st.sidebar.warning("⚠️ Bazı API anahtarlarınız geçersiz veya boş.")

        st.sidebar.markdown("---")

        if st.sidebar.button("Çıkış Yap", key="essiz_cikis_butonu"):
            for key in ['logged_in', 'verified', 'user_email', 'username', 'gemini_key', 'ollama_key', 'api_status']:
                st.session_state[key] = "" if key not in ['logged_in', 'verified'] else False
            if 'api_status' in st.session_state:
                del st.session_state['api_status']
            st.rerun()

        return True

    return st.session_state.logged_in and st.session_state.verified