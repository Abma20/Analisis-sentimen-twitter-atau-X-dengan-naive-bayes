Analisis Sentimen Produk di Media Sosial X dengan Naive Bayes
📝 Deskripsi Proyek
Proyek ini bertujuan untuk melakukan analisis sentimen terhadap opini publik mengenai [Sebutkan Nama Produk, misal: iPhone 15 / Layanan Grab / dll] di platform media sosial X (sebelumnya Twitter).

Dengan menggunakan algoritma Naive Bayes Classifier, sistem ini mengklasifikasikan tweet ke dalam tiga kategori sentimen:

Positif

Negatif

Netral

Proyek ini bermanfaat untuk memahami persepsi pengguna terhadap produk dan dapat digunakan sebagai bahan evaluasi bisnis.

Shutterstock

🚀 Fitur Utama
Data Crawling: Mengambil data ulasan/tweet dari API X (atau dataset statis).

Text Preprocessing (Bahasa Indonesia):

Cleaning (menghapus URL, simbol, angka).

Case Folding (mengubah huruf menjadi kecil).

Tokenization.

Stopword Removal (menghapus kata umum yang tidak bermakna).

Stemming (menggunakan library Sastrawi untuk kata dasar).

Feature Extraction: Menggunakan TF-IDF (Term Frequency-Inverse Document Frequency).

Modeling: Implementasi algoritma Multinomial Naive Bayes.

Evaluasi: Laporan akurasi, Presisi, Recall, dan F1-Score.

Visualisasi: Word Cloud dan Confusion Matrix.
🛠️ Teknologi yang Digunakan
Bahasa Pemrograman: Python

Data Manipulation: Pandas, NumPy

NLP Library: NLTK, Sastrawi (untuk Bahasa Indonesia)

Machine Learning: Scikit-learn

Visualisasi: Matplotlib, Seaborn, WordCloud
