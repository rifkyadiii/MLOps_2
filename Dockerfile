# Gunakan base image Python yang ringan
FROM python:3.10-slim

# Set direktori kerja di dalam container
WORKDIR /app

# Salin semua file proyek ke dalam direktori kerja
COPY . .

# Install semua library yang dibutuhkan
RUN pip install --no-cache-dir -r requirements.txt

# Port yang akan diekspos oleh container
EXPOSE 7860

# Perintah untuk menjalankan aplikasi saat container dimulai
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]