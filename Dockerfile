# Gunakan base image Python
FROM python:3.10-slim

# Set direktori kerja di dalam container
WORKDIR /app

# Salin semua file proyek ke dalam container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt