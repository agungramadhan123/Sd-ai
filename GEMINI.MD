Bertindak sebagai Senior MLOps dan Backend Engineer. Saya memiliki proyek FastAPI yang sudah ada untuk analisis sentimen Twitter menggunakan model scikit-learn yang telah dilatih sebelumnya. Saya ingin meningkatkan aplikasi ini dengan beberapa fitur baru dan mempersiapkannya untuk penerapan produksi.

Berikut adalah struktur basis kode saya saat ini:
- `main.py` (Aplikasi FastAPI yang menangani prediksi dan pencatatan SQLite)
- `config.py` (Pengaturan konfigurasi seperti MAX_INPUT_LENGTH=10000)
- `processing.py` (Modul pra-pemrosesan teks)
- `models/` (Direktori yang berisi file .pkl)
- `static/` (Direktori yang berisi frontend index.html)

Silakan buat kode yang diperbarui dan file konfigurasi yang diperlukan untuk mengimplementasikan peningkatan berikut:

**1. Pemilihan Multi-Model (Backend & Pydantic)**
- Perbarui model Pydantic `TweetRequest` untuk menyertakan bidang `model_name` opsional (defaultnya adalah "logistic_regression").

- Memperbaiki pengelola konteks `lifespan` di `main.py` untuk memuat kamus beberapa model (misalnya, "logistic_regression" dan "random_forest") dari direktori `models/` alih-alih hanya satu `_model_pipeline` global.

- Memperbarui endpoint `/predict` untuk mengarahkan permintaan inferensi ke model yang dipilih dan menangani kesalahan jika model yang diminta tidak ada.

**2. Dashboard Real-Time (WebSocket)**
- Menambahkan endpoint WebSocket baru di `main.py` (misalnya, `/ws/dashboard`) yang mengirimkan statistik prediksi real-time dan log riwayat terbaru ke klien segera setelah prediksi baru dibuat dan dicatat ke dalam basis data.

**3. Pemisahan Frontend melalui Docker Compose**
- Menghentikan penyajian direktori `static/` secara langsung melalui `StaticFiles` FastAPI.

- Buat file `docker-compose.yml` yang mendefinisikan dua layanan:
a. `backend`: Aplikasi FastAPI yang dikontainerisasi menggunakan `Dockerfile.backend`.

b. `frontend`: Kontainer Nginx yang menyajikan file HTML/JS/CSS dari direktori `static/` pada port 80.
- Berikan kode `Dockerfile.backend`.

**4. Persiapan Penyebaran untuk Render**
- Berikan file Blueprint `render.yaml` (Infrastruktur sebagai Kode) untuk dengan mudah menyebarkan layanan backend FastAPI sebagai Layanan Web pada tingkatan gratis Render.

**5. Harap baca terlebih dahulu seluruh folder saya untuk menyesuaikan pekerjaan ini dengan project saya.
Harap berikan kode lengkap dan terbaru untuk `main.py`, `Dockerfile.backend`, `docker-compose.yml`, dan `render.yaml`. Gunakan praktik penulisan kode yang bersih layaknya manusia enggeneering yang mengetik kode tanpa menggunakan emot,  dan simbol apapun,  pengetikan yang tepat, dan pastikan penanganan kesalahan yang benar (HTTP 422 untuk input > 10000 karakter, HTTP 400 untuk model tidak ditemukan).