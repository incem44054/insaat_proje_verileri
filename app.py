# -*- coding: utf-8 -*-

"""
İnşaat Proje Maliyeti Tahmin Aracı
"""

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# ---------------------------------------------------------
# Dosya yolları
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

BASIT_MODEL_PATH = MODELS_DIR / "maliyet_modeli_basit.pkl"
GELISMIS_MODEL_PATH = MODELS_DIR / "maliyet_modeli_gelismis.pkl"
META_PATH = MODELS_DIR / "meta.json"


# ---------------------------------------------------------
# Sayfa ayarları
# ---------------------------------------------------------

st.set_page_config(
    page_title="İnşaat Maliyet Tahmin Aracı",
    page_icon="🏗️",
    layout="centered",
)


# ---------------------------------------------------------
# Renkler
# ---------------------------------------------------------

PRIMARY = "#1F5C99"
NAVY = "#0A2342"
GREEN = "#2E7D32"
RED = "#C0392B"


# ---------------------------------------------------------
# Stil
# ---------------------------------------------------------

st.markdown(
    f"""
    <style>
        .main {{
            background-color: #F4F7FA;
        }}

        .stApp header {{
            background-color: transparent;
        }}

        h1 {{
            color: {NAVY};
        }}

        .app-header {{
            background-color: {NAVY};
            padding: 1.3rem 1.6rem;
            border-radius: 10px;
            margin-bottom: 1.2rem;
        }}

        .app-header h1 {{
            color: white;
            margin: 0;
            font-size: 1.6rem;
        }}

        .app-header p {{
            color: #D5E8F0;
            margin: 0.3rem 0 0 0;
            font-size: 0.95rem;
        }}

        .result-box {{
            background-color: {PRIMARY};
            color: white;
            padding: 1.4rem;
            border-radius: 10px;
            text-align: center;
            margin: 1rem 0;
        }}

        .result-box .value {{
            font-size: 2.3rem;
            font-weight: 700;
        }}

        .result-box .label {{
            font-size: 0.95rem;
            opacity: 0.85;
        }}

        .warn-box {{
            background-color: #FDEDEC;
            border: 1.5px solid {RED};
            color: {RED};
            padding: 0.9rem 1.1rem;
            border-radius: 8px;
            font-size: 0.92rem;
            margin-top: 0.6rem;
        }}

        .ok-box {{
            background-color: #EAF6EC;
            border: 1.5px solid {GREEN};
            color: {GREEN};
            padding: 0.9rem 1.1rem;
            border-radius: 8px;
            font-size: 0.92rem;
            margin-top: 0.6rem;
        }}

        footer {{
            visibility: hidden;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Başlık
# ---------------------------------------------------------

st.markdown(
    """
    <div class="app-header">
        <h1>🏗️ İnşaat Proje Maliyeti Tahmin Aracı</h1>
        <p>İnşaat Mühendisliğinde Yapay Zekâ Uygulamaları — Hafta 2 Lab Projesi</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Model ve meta verisini yükleme
# ---------------------------------------------------------

@st.cache_resource
def modelleri_yukle():
    """
    Model dosyalarını ve meta.json dosyasını yükler.
    """

    eksik_dosyalar = []

    for dosya in [
        BASIT_MODEL_PATH,
        GELISMIS_MODEL_PATH,
        META_PATH,
    ]:
        if not dosya.exists():
            eksik_dosyalar.append(str(dosya))

    if eksik_dosyalar:
        raise FileNotFoundError(
            "Aşağıdaki dosyalar bulunamadı:\n"
            + "\n".join(eksik_dosyalar)
        )

    model_basit = joblib.load(BASIT_MODEL_PATH)
    model_gelismis = joblib.load(GELISMIS_MODEL_PATH)

    with open(META_PATH, "r", encoding="utf-8") as dosya:
        meta = json.load(dosya)

    return model_basit, model_gelismis, meta


try:
    model_basit, model_gelismis, meta = modelleri_yukle()

except Exception as hata:
    st.error("Model dosyaları yüklenirken hata oluştu.")

    st.code(str(hata))

    st.info(
        "Dosya yapısının şu şekilde olduğundan emin olun:\n\n"
        "app.py\n"
        "models/maliyet_modeli_basit.pkl\n"
        "models/maliyet_modeli_gelismis.pkl\n"
        "models/meta.json"
    )

    st.stop()


# ---------------------------------------------------------
# Meta verisinden güvenli aralık alma
# ---------------------------------------------------------

def meta_araligi(anahtar, varsayilan_min, varsayilan_max):
    """
    meta.json içinde ilgili aralık varsa onu kullanır.
    Yoksa varsayılan değerleri kullanır.
    """

    bilgi = meta.get(anahtar, {})

    minimum = bilgi.get("min", varsayilan_min)
    maksimum = bilgi.get("max", varsayilan_max)

    return float(minimum), float(maksimum)


alan_min, alan_max = meta_araligi(
    "alan_m2",
    varsayilan_min=50,
    varsayilan_max=10000,
)

kat_min, kat_max = meta_araligi(
    "kat_sayisi",
    varsayilan_min=1,
    varsayilan_max=40,
)

yil_min, yil_max = meta_araligi(
    "insaat_yili",
    varsayilan_min=2015,
    varsayilan_max=2030,
)


# ---------------------------------------------------------
# Kenar çubuğu
# ---------------------------------------------------------

st.sidebar.markdown("### ⚙️ Model Seçimi")

model_secimi = st.sidebar.radio(
    "Hangi modeli kullanmak istersiniz?",
    [
        "Basit Model",
        "Gelişmiş Model",
    ],
    help=(
        "Basit model alan ve kat sayısını kullanır. "
        "Gelişmiş model alan, kat sayısı, inşaat yılı "
        "ve zemin sınıfını kullanır."
    ),
)

gelismis_mi = model_secimi == "Gelişmiş Model"


st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Model Bilgisi")


if gelismis_mi:
    model_bilgisi = meta.get("gelismis_model", {})

    st.sidebar.metric(
        "Test R²",
        f"{model_bilgisi.get('r2', 0):.3f}",
    )

    st.sidebar.metric(
        "Train R²",
        f"{model_bilgisi.get('train_r2', 0):.3f}",
    )

    st.sidebar.metric(
        "Test MAE",
        f"{model_bilgisi.get('mae', 0):,.0f} TL",
    )

    st.sidebar.caption(
        "Train R² ile Test R² arasındaki büyük fark, "
        "modelde aşırı öğrenme riski olabileceğini gösterir."
    )

else:
    model_bilgisi = meta.get("basit_model", {})

    st.sidebar.metric(
        "Test R²",
        f"{model_bilgisi.get('r2', 0):.3f}",
    )

    st.sidebar.metric(
        "Test MAE",
        f"{model_bilgisi.get('mae', 0):,.0f} TL",
    )

    st.sidebar.caption(
        "Basit model yalnızca alan ve kat sayısını kullanır."
    )


st.sidebar.markdown("---")

st.sidebar.caption(
    f"Eğitim verisi: {meta.get('n_proje', 'Bilinmiyor')} proje kaydı"
)


# ---------------------------------------------------------
# Proje bilgileri
# ---------------------------------------------------------

st.markdown("### Proje Bilgilerini Girin")

col1, col2 = st.columns(2)


with col1:
    alan_m2 = st.number_input(
        "Taban Alanı (m²)",
        min_value=float(alan_min),
        max_value=float(alan_max),
        value=float(max(alan_min, min(1200, alan_max))),
        step=50.0,
    )


with col2:
    kat_sayisi = st.number_input(
        "Kat Sayısı",
        min_value=float(kat_min),
        max_value=float(kat_max),
        value=float(max(kat_min, min(8, kat_max))),
        step=1.0,
    )


zemin_sinifi = None
insaat_yili = None


if gelismis_mi:

    zemin_siniflari = meta.get(
        "zemin_siniflari",
        ["A", "B", "C", "D"],
    )

    if not zemin_siniflari:
        zemin_siniflari = ["A", "B", "C", "D"]

    col3, col4 = st.columns(2)

    with col3:
        zemin_sinifi = st.selectbox(
            "Zemin Sınıfı",
            zemin_siniflari,
            index=0,
        )

    with col4:
        insaat_yili = st.number_input(
            "İnşaat Bitiş Yılı",
            min_value=float(yil_min),
            max_value=float(yil_max),
            value=float(yil_max),
            step=1.0,
        )


# ---------------------------------------------------------
# Ekstrapolasyon kontrolü
# ---------------------------------------------------------

def araligin_disinda_mi(deger, minimum, maksimum):
    """
    Değerin eğitim verisi aralığında olup olmadığını kontrol eder.
    """

    disinda = deger < minimum or deger > maksimum

    return disinda


uyarilar = []


if araligin_disinda_mi(alan_m2, alan_min, alan_max):
    uyarilar.append(
        f"Alan değeri ({alan_m2:,.0f} m²), "
        f"eğitim aralığının ({alan_min:,.0f}–{alan_max:,.0f} m²) dışında."
    )


if araligin_disinda_mi(kat_sayisi, kat_min, kat_max):
    uyarilar.append(
        f"Kat sayısı ({kat_sayisi:,.0f}), "
        f"eğitim aralığının ({kat_min:,.0f}–{kat_max:,.0f}) dışında."
    )


if gelismis_mi and insaat_yili is not None:
    if araligin_disinda_mi(insaat_yili, yil_min, yil_max):
        uyarilar.append(
            f"İnşaat yılı ({insaat_yili:,.0f}), "
            f"eğitim aralığının ({yil_min:,.0f}–{yil_max:,.0f}) dışında."
        )


# ---------------------------------------------------------
# Tahmin fonksiyonları
# ---------------------------------------------------------

def basit_model_tahmini():
    """
    Basit model için tahmin oluşturur.
    """

    veri = pd.DataFrame(
        [
            {
                "alan_m2": alan_m2,
                "kat_sayisi": kat_sayisi,
            }
        ]
    )

    return model_basit.predict(veri)[0]


def gelismis_model_tahmini():
    """
    Gelişmiş model için tahmin oluşturur.

    Özellik isimlerini meta.json içindeki features listesinden alır.
    Böylece modelin beklediği sütun sırası korunur.
    """

    gelismis_bilgi = meta.get("gelismis_model", {})

    ozellikler = gelismis_bilgi.get(
        "features",
        [
            "alan_m2",
            "kat_sayisi",
            "insaat_yili",
            "zemin_sinifi_B",
            "zemin_sinifi_C",
            "zemin_sinifi_D",
        ],
    )

    satir = {}

    # Modelin istediği bütün özellikleri başlangıçta sıfırla
    for ozellik in ozellikler:
        satir[ozellik] = 0

    # Sayısal değerleri ekle
    if "alan_m2" in satir:
        satir["alan_m2"] = alan_m2

    if "kat_sayisi" in satir:
        satir["kat_sayisi"] = kat_sayisi

    if "insaat_yili" in satir:
        satir["insaat_yili"] = insaat_yili

    # Zemin sınıfını one-hot formatına dönüştür
    zemin_sutunu = f"zemin_sinifi_{zemin_sinifi}"

    if zemin_sutunu in satir:
        satir[zemin_sutunu] = 1

    veri = pd.DataFrame([satir])

    # Modelin eğitimde kullandığı sıraya göre sütunları düzenle
    veri = veri[ozellikler]

    return model_gelismis.predict(veri)[0]


# ---------------------------------------------------------
# Tahmin butonu
# ---------------------------------------------------------

if st.button(
    "💰 Maliyeti Tahmin Et",
    type="primary",
    use_container_width=True,
):

    try:
        if gelismis_mi:
            tahmin = gelismis_model_tahmini()
        else:
            tahmin = basit_model_tahmini()

        tahmin = float(tahmin)

    except Exception as hata:
        st.error("Tahmin yapılırken hata oluştu.")
        st.code(str(hata))

        st.info(
            "Özellikle meta.json içindeki features listesi ile "
            "modelin eğitimde kullandığı sütun adlarını kontrol edin."
        )

        st.stop()


    # -----------------------------------------------------
    # Sonucu göster
    # -----------------------------------------------------

    if uyarilar:

        st.markdown(
            f"""
            <div class="result-box" style="background-color:{RED};">
                <div class="value">{tahmin:,.0f} TL</div>
                <div class="label">
                    Tahmini Toplam Maliyet — GÜVENİLİR DEĞİL
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="warn-box">
                <b>⚠️ Ekstrapolasyon uyarısı:</b>
            """,
            unsafe_allow_html=True,
        )

        for uyari in uyarilar:
            st.write(f"- {uyari}")

        st.markdown(
            """
            Model bu değer aralığında yeterli veri görmemiş olabilir.
            Bu nedenle tahmin dikkatli değerlendirilmelidir.
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        st.markdown(
            f"""
            <div class="result-box">
                <div class="value">{tahmin:,.0f} TL</div>
                <div class="label">
                    Tahmini Toplam Maliyet
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="ok-box">
                ✅ Girdi değerleri eğitim verisinin aralığı içinde.
                Tahmin makul bir güvenilirlik taşıyor.
            </div>
            """,
            unsafe_allow_html=True,
        )


    # -----------------------------------------------------
    # Modelin hesaplama açıklaması
    # -----------------------------------------------------

    with st.expander("📐 Model bu tahmini nasıl hesapladı?"):

        if gelismis_mi:

            katsayilar = meta.get(
                "gelismis_model",
                {},
            ).get(
                "coefs",
                {},
            )

            st.markdown(
                f"""
                Gelişmiş model, özelliklerin katsayılarını kullanarak
                tahmin üretir.

                - Alan katsayısı: **{katsayilar.get("alan_m2", 0):,.0f} TL/m²**
                - Kat katsayısı: **{katsayilar.get("kat_sayisi", 0):,.0f} TL/kat**
                - Yıl katsayısı: **{katsayilar.get("insaat_yili", 0):,.0f} TL/yıl**
                - Zemin sınıfı etkisi:

                  - B: **{katsayilar.get("zemin_sinifi_B", 0):,.0f} TL**
                  - C: **{katsayilar.get("zemin_sinifi_C", 0):,.0f} TL**
                  - D: **{katsayilar.get("zemin_sinifi_D", 0):,.0f} TL**
                """
            )

        else:

            basit_bilgi = meta.get("basit_model", {})

            alan_katsayisi = basit_bilgi.get(
                "alan_katsayisi",
                0,
            )

            kat_katsayisi = basit_bilgi.get(
                "kat_katsayisi",
                0,
            )

            sabit = basit_bilgi.get(
                "intercept",
                basit_bilgi.get("sabit", 0),
            )

            st.markdown(
                f"""
                Basit modelin kullandığı formül:

                **Maliyet = ({alan_katsayisi:,.0f} × Alan) +
                ({kat_katsayisi:,.0f} × Kat) +
                {sabit:,.0f}**
                """
            )

        st.caption(
            "Bu uygulama bir karar destek aracıdır. "
            "Nihai karar her zaman uzman mühendis tarafından verilmelidir."
        )


# ---------------------------------------------------------
# Alt bilgi
# ---------------------------------------------------------

st.markdown("---")

st.caption(
    "Bu araç, makine öğrenmesi modeli kullanılarak "
    "inşaat proje maliyeti tahmini yapmak amacıyla hazırlanmıştır."
)
