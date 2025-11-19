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

<img width="1365" height="717" alt="Screenshot 2025-07-28 162942" src="https://github.com/user-attachments/assets/bab3fd90-3d2a-4504-ab6b-2fecb115fcb5" />

<img width="1365" height="719" alt="Screenshot 2025-07-28 163440" src="https://github.com/user-attachments/assets/2d9ba90d-622c-45d4-93ae-746bfc892f56" />

<img width="1364" height="713" alt="Screenshot 2025-07-28 163535" src="https://github.com/user-attachments/assets/23eb13d6-0945-4b97-b13a-666c31ab2da7" />

<img width="1364" height="714" alt="Screenshot 2025-07-28 163237" src="https://github.com/user-attachments/assets/0ee5bc8c-8e85-4557-b691-e177150b6725" />


📜 Lisensi
Didistribusikan di bawah Lisensi MIT. Lihat LICENSE untuk informasi lebih lanjut.

Dibuat oleh Majid

LinkedIn: https://www.linkedin.com/in/majid-703905382/

Email: mazidabdul987@gmail.com
