# -*- coding: utf-8 -*-
"""
Hafta 2 dersinde kurduğumuz iki modeli (Hücre 5 ve Hücre 5-Devam) birebir
aynı kodla eğitir ve Streamlit uygulamasının kullanacağı dosyalara kaydeder.
"""
import os
import json
import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

DATA_PATH = "insaat_proje_verileri.csv"
OUT_DIR = "models"
os.makedirs(OUT_DIR, exist_ok=True)
df = pd.read_csv(DATA_PATH)

# ---------------------------------------------------------------
# MODEL 1 — Hücre 5: Basit model (alan_m2, kat_sayisi)
# ---------------------------------------------------------------
X = df[['alan_m2', 'kat_sayisi']]
y = df['toplam_maliyet_TL']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
basit_metrikleri = {
    "mae": float(mean_absolute_error(y_test, y_pred)),
    "r2": float(r2_score(y_test, y_pred)),
    "alan_katsayisi": float(model.coef_[0]),
    "kat_katsayisi": float(model.coef_[1]),
    "features": ["alan_m2", "kat_sayisi"],
}
joblib.dump(model, f"{OUT_DIR}/maliyet_modeli_basit.pkl")
print("Basit model kaydedildi:", basit_metrikleri)

# ---------------------------------------------------------------
# MODEL 2 — Hücre 5 (Devam): Gelişmiş model (+zemin_sinifi, insaat_yili)
# ---------------------------------------------------------------
df_encoded = pd.get_dummies(df, columns=['zemin_sinifi'], drop_first=True)
ozellikler = ['alan_m2', 'kat_sayisi', 'insaat_yili',
              'zemin_sinifi_B', 'zemin_sinifi_C', 'zemin_sinifi_D']
X2 = df_encoded[ozellikler]

X2_train, X2_test, y_train2, y_test2 = train_test_split(
    X2, y, test_size=0.2, random_state=42)

model2 = LinearRegression()
model2.fit(X2_train, y_train2)

y_pred2 = model2.predict(X2_test)
gelismis_metrikleri = {
    "mae": float(mean_absolute_error(y_test2, y_pred2)),
    "r2": float(r2_score(y_test2, y_pred2)),
    "train_r2": float(r2_score(y_train2, model2.predict(X2_train))),
    "features": ozellikler,
    "coefs": {f: float(c) for f, c in zip(ozellikler, model2.coef_)},
}
joblib.dump(model2, f"{OUT_DIR}/maliyet_modeli_gelismis.pkl")
print("Gelişmiş model kaydedildi:", gelismis_metrikleri)

# ---------------------------------------------------------------
# Metaveri: min/max aralıkları (ekstrapolasyon uyarısı için) + metrikler
# ---------------------------------------------------------------
meta = {
    "alan_m2": {"min": int(df["alan_m2"].min()), "max": int(df["alan_m2"].max())},
    "kat_sayisi": {"min": int(df["kat_sayisi"].min()), "max": int(df["kat_sayisi"].max())},
    "insaat_yili": {"min": int(df["insaat_yili"].min()), "max": int(df["insaat_yili"].max())},
    "zemin_siniflari": sorted(df["zemin_sinifi"].unique().tolist()),
    "n_proje": int(len(df)),
    "basit_model": basit_metrikleri,
    "gelismis_model": gelismis_metrikleri,
}
with open(f"{OUT_DIR}/meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print("\nMetaveri kaydedildi:")
print(json.dumps(meta, ensure_ascii=False, indent=2))
