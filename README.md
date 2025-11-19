Analisis Sentimen Produk di Media Sosial X dengan Naive Bayes
📝 Deskripsi Proyek
Proyek ini bertujuan untuk melakukan analisis sentimen terhadap opini publik mengenai parfum di platform media sosial X (sebelumnya Twitter).

Dengan menggunakan algoritma Naive Bayes Classifier, sistem ini mengklasifikasikan tweet ke dalam tiga kategori sentimen:

Positif

Negatif

Netral

Proyek ini bermanfaat untuk memahami persepsi pengguna terhadap produk dan dapat digunakan sebagai bahan evaluasi bisnis.

<img width="3999" height="2761" alt="image" src="https://github.com/user-attachments/assets/e12aa96a-60ea-426b-8e39-1854adbb5ed7" />

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
<img width="3999" height="3999" alt="image" src="https://github.com/user-attachments/assets/c78fd326-62fe-4412-ad74-1fcc9ce06074" />


🛠️ Teknologi yang Digunakan
Bahasa Pemrograman: Python

Data Manipulation: Pandas, NumPy

NLP Library: NLTK, Sastrawi (untuk Bahasa Indonesia)

Machine Learning: Scikit-learn

Visualisasi: Matplotlib, Seaborn, WordCloud

📜 Lisensi
Didistribusikan di bawah Lisensi MIT. Lihat LICENSE untuk informasi lebih lanjut.

Dibuat oleh Majid

LinkedIn: https://www.linkedin.com/in/majid-703905382/

Email: mazidabdul987@gmail.com
