# Proyek Klasifikasi Harga Handphone dengan TFX Pipeline

- Nama: Moch Rifky Aulia Adikusumah
- Username Dicoding: rifkyadi

---

## Detail Proyek

| Deskripsi | Keterangan |
| :--- | :--- |
| Dataset | [Mobile Price Classification](https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification) - Dataset ini berisi data spesifikasi handphone dan targetnya adalah `price_range` (0=murah, 1=sedang, 2=mahal, 3=sangat mahal). |
| Masalah | Banyaknya model handphone baru dengan berbagai spesifikasi membuat konsumen dan produsen sulit untuk menentukan posisi harga yang tepat. Kesalahan dalam penentuan harga dapat menyebabkan produk tidak laku atau keuntungan yang tidak maksimal. |
| Solusi Machine Learning | Membangun sebuah model klasifikasi yang dapat memprediksi rentang harga (`price_range`) sebuah handphone berdasarkan fitur-fitur teknisnya (seperti RAM, kapasitas baterai, resolusi kamera, dll.). Solusi ini diimplementasikan dalam sebuah pipeline TFX yang otomatis, mulai dari data ingestion, validasi, preprocessing, training, evaluasi, hingga siap untuk deployment. Target utama adalah mencapai akurasi klasifikasi di atas 90% pada data validasi. |
| Metode Pengolahan Data | Menggunakan komponen `Transform` dari TFX. Fitur-fitur numerik (seperti `battery_power`, `ram`, `px_width`, dll.) akan dinormalisasi menggunakan standard scaling (Z-score) untuk menyamakan skala dan membantu model Konvergen lebih cepat. Fitur kategorikal (seperti `four_g`, `dual_sim`, dll.) yang sudah dalam format biner (0 atau 1) akan langsung digunakan. |
| Arsitektur Model | Menggunakan model Deep Neural Network (DNN) yang dibangun dengan TensorFlow/Keras. Arsitektur model terdiri dari: Beberapa layer `Dense` dengan fungsi aktivasi ReLU, sebuah layer `Dropout` untuk mencegah overfitting, layer output `Dense` dengan 4 unit (sesuai jumlah kelas `price_range`) dan fungsi aktivasi Softmax untuk klasifikasi multi-kelas. |
| Metrik Evaluasi | Metrik utama yang digunakan adalah Sparse Categorical Accuracy. Akurasi dipilih karena distribusi kelas pada dataset ini seimbang, sehingga akurasi dapat menjadi representasi performa model yang baik. |
| Performa Model | Model yang telah dilatih dan dievaluasi menggunakan TFX Evaluator berhasil mencapai akurasi sebesar 89% pada set data evaluasi. |
| Opsi deployment | Model ini di-deploy sebagai sebuah layanan API menggunakan FastAPI dan dikemas dalam sebuah container Docker. Aplikasi di-hosting secara publik di Hugging Face Spaces untuk aksesibilitas global. Untuk monitoring, server Prometheus dan dashboard Grafana dijalankan secara lokal di lingkungan Docker, yang kemudian memantau (scrape) metrik dari endpoint aplikasi yang sedang live di cloud. Pendekatan hybrid ini dipilih untuk memanfaatkan platform hosting gratis sekaligus memiliki kendali penuh atas stack monitoring. |
| Web app | Endpoint utama untuk prediksi dapat diakses melalui request POST ke URL berikut: [Mobile Price Prediction](https://huggingface.co/spaces/rifkyadiii/mobile-price-prediction)|
| Monitoring | Monitoring secara real-time menggunakan stack Prometheus dan Grafana mengonfirmasi bahwa API berjalan dengan sangat responsif dan stabil, yang dibuktikan dengan latensi P95 di bawah 100ms dan tingkat error server yang konsisten di angka nol. Selain itu, metrik penggunaan resource CPU dan memori juga terpantau dalam batas normal tanpa adanya indikasi anomali, yang menandakan efisiensi dan keandalan sistem secara keseluruhan. |