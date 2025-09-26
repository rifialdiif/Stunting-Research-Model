# 1. Import library yang dibutuhkan
import pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

# 2. Inisialisasi aplikasi FastAPI
app = FastAPI(
    title="API Deteksi Stunting",
    description="API ini memanfaatkan model Machine Learning untuk mendeteksi status gizi balita beserta confidence score-nya.",
    version="1.1.0" # Versi update
)

# 3. Definisikan struktur input data (tidak ada perubahan)
class StuntingFeatures(BaseModel):
    umur_bulan: int
    tinggi_badan_cm: float
    jenis_kelamin_encoded: int

# 4. Muat model dan encoder (tidak ada perubahan)
try:
    with open('model/model_klasifikasi_stunting.pkl', 'rb') as file:
        model = pickle.load(file)
    
    with open('model/label_encoder_stunting.pkl', 'rb') as file:
        le_status_gizi = pickle.load(file)
    print("Model dan Encoder berhasil dimuat.")
except FileNotFoundError:
    print("Error: Pastikan file 'model_klasifikasi_stunting.pkl' dan 'encoder_status_gizi.pkl' ada di direktori yang sama.")
    model = None
    le_status_gizi = None

# 5. Buat endpoint untuk prediksi (DENGAN PERUBAHAN)
@app.post("/predict")
async def predict_stunting(features: StuntingFeatures):
    """
    Endpoint untuk memprediksi status gizi balita.
    
    - **umur_bulan**: Usia balita dalam bulan (integer).
    - **tinggi_badan_cm**: Tinggi badan balita dalam cm (float).
    - **jenis_kelamin_encoded**: Jenis kelamin yang sudah di-encode (0: laki-laki, 1: perempuan).
    """
    if model is None or le_status_gizi is None:
        return {"error": "Model atau Encoder tidak berhasil dimuat. Periksa log server."}

    # Ubah data input menjadi format numpy array
    input_data = np.array([[
        features.umur_bulan,
        features.tinggi_badan_cm,
        features.jenis_kelamin_encoded
    ]])

    # --- PERUBAHAN UTAMA DIMULAI DI SINI ---

    # 1. Lakukan prediksi probabilitas untuk setiap kelas
    probabilities = model.predict_proba(input_data)
    
    # 2. Dapatkan confidence score (probabilitas tertinggi)
    confidence_score = np.max(probabilities)
    
    # 3. Dapatkan indeks kelas dengan probabilitas tertinggi
    prediction_encoded = np.argmax(probabilities)
    
    # 4. Ubah hasil prediksi dari angka kembali ke label teks
    prediction_label = le_status_gizi.inverse_transform([prediction_encoded])

    # --- PERUBAHAN SELESAI ---

    # Kembalikan hasil prediksi baru dalam format JSON
    return {
        "prediction_label": prediction_label[0],
        "confidence_score": float(confidence_score)
    }

# Endpoint root untuk verifikasi sederhana
@app.get("/")
def read_root():
    return {"status": "API Deteksi Stunting berjalan dengan baik!"}