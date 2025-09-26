# API Deteksi Stunting

API ini menyediakan endpoint untuk memprediksi status gizi balita berdasarkan data usia, tinggi badan, dan jenis kelamin. Model machine learning yang digunakan telah dilatih dan disimpan dalam file pickle.

---

## Base URL

```
http://<host>:<port>/
```

Contoh default saat dijalankan lokal:

```
http://127.0.0.1:8000/
```

---

## Endpoint

### 1. Prediksi Status Gizi Balita

**URL:** `/predict`

**Method:** `POST`

**Content-Type:** `application/json`

#### Request Body

| Field                 | Tipe  | Deskripsi                                  |
| --------------------- | ----- | ------------------------------------------ |
| umur_bulan            | int   | Usia balita dalam bulan                    |
| tinggi_badan_cm       | float | Tinggi badan balita dalam cm               |
| jenis_kelamin_encoded | int   | Jenis kelamin (0: laki-laki, 1: perempuan) |

**Contoh JSON:**

```json
{
  "umur_bulan": 30,
  "tinggi_badan_cm": 85,
  "jenis_kelamin_encoded": 0
}
```

#### Response

| Field            | Tipe   | Deskripsi                                       |
| ---------------- | ------ | ----------------------------------------------- |
| prediction_label | string | Hasil prediksi status gizi (label)              |
| confidence_score | float  | Nilai confidence score (probabilitas tertinggi) |

**Contoh Response:**

```json
{
  "prediction_label": "Normal",
  "confidence_score": 0.92
}
```

---

## Cara Menggunakan

1. **Jalankan server API:**
   ```
   uvicorn main:app --reload
   ```
2. **Kirim request POST ke endpoint `/predict`**
   - Gunakan tools seperti Postman, Insomnia, atau curl.
   - Contoh dengan curl (PowerShell):
     ```powershell
     curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d "{`"umur_bulan`":30,`"tinggi_badan_cm`":85,`"jenis_kelamin_encoded`":0}"
     ```

---

## Dokumentasi Otomatis

Setelah server berjalan, dokumentasi interaktif dapat diakses di:

- Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Redoc: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## Catatan

- Pastikan file model (`model_klasifikasi_stunting.pkl` dan `label_encoder_stunting.pkl`) tersedia di folder `model/`.
- Jika ada error terkait model/encoder, cek log server dan pastikan file sudah benar.
- Untuk integrasi mobile, gunakan endpoint `/predict` dengan format JSON seperti di atas.

---
