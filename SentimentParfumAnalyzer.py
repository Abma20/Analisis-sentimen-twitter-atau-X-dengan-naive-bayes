import tkinter
from tkinter import filedialog, messagebox
import customtkinter as ctk
import pandas as pd
import re
import threading
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from wordcloud import WordCloud

# --- Kamus untuk mengkategorikan jenis wangi (aroma umum) ---
KAMUS_WANGI = {
    'Manis (Gourmand)': ['vanilla', 'manis', 'coklat', 'kopi', 'caramel', 'gourmand', 'almond', 'madu', 'honey'],
    'Segar (Fresh)': ['segar', 'fresh', 'citrus', 'lemon', 'jeruk', 'mint', 'aquatic', 'laut', 'marine', 'clean'],
    'Bunga (Floral)': ['bunga', 'floral', 'mawar', 'rose', 'melati', 'jasmine', 'lily', 'tuberose', 'gardenia'],
    'Kayu (Woody)': ['kayu', 'woody', 'cedarwood', 'sandalwood', 'oud', 'gaharu', 'vetiver', 'patchouli', 'nilam'],
    'Rempah (Spicy)': ['rempah', 'spicy', 'cinnamon', 'kayu manis', 'lada', 'pepper', 'cardamom', 'kapulaga', 'clove', 'cengkeh'],
    'Buah (Fruity)': ['buah', 'fruity', 'apel', 'apple', 'peach', 'pir', 'pear', 'berry', 'leci', 'lychee']
}

# --- Kamus baru khusus untuk nama parfum ---
KAMUS_PARFUM = {
    'Aroma kopi yang dipadukan dengan aroma vanial (YSL Black Opium)': ['black opium'],
    'Victoria\'s Secret Scandalous': ['scandalous'],
    'Aroma perpaduan rempah (seperti cengkeh), aroma kulit (leather), dan kesegaran sitrus (Aigner Blue)': ['aigner blue'],
    'Aroma perpaduan litchi, bergamot, dan aroma laut yang ringan (Dunhill Desire Blue)': ['dunhill blue'],
    'aroma perpaduan dari buah-buahan dan bunga (Victoria\'s Secret)': ['victoria secret'],
    'Aroma segar dan maskulin dengan sentuhan pedas dan amber (Dior Sauvage)': ['dior sauvage', 'sauvage'],
    'Klasik, feminin, dan elegan dengan dominasi aldehida, bunga melati, mawar, dan cendana (Chanel No. 5)': ['chanel no 5', 'chanel number 5', 'chanel no. 5'],
    'Aroma mewah dan berani dengan perpaduan nanas, bergamot, apel, dan musk, sering diasosiasikan dengan kesuksesan (Creed Aventus)': ['creed aventus', 'aventus'],
    'Aroma manis, hangat, dan kompleks dengan sentuhan saffron, amberwood, dan cedar. Sering digambarkan sebagai aroma "sugary" dan sedikit "medicinal" namun sangat adiktif (Baccarat Rouge 540)': ['baccarat rouge 540', 'baccarat 540', 'br540'],
    'aroma yang seringkali didominasi oleh satu atau dua bahan utama, seperti lime basil & mandarin, atau peony & blush suede. Aromanya cenderung segar, bersih, dan natura (Jo Malone London)': ['jo malone'],
    'Aroma maskulin yang kuat dan berani dengan sentuhan jeruk bali, mint, mawar, dan kulit. Sering digambarkan sebagai aroma "pesta" atau "malam" (Paco Rabanne 1 Million)': ['1 million', 'one million', 'paco rabanne 1 million'],
    'Aroma manis dan gourmand dengan perpaduan iris, blackcurrant, pir, dan praline. Memberikan kesan bahagia dan feminin (Lancôme La Vie Est Belle)': ['la vie est belle', 'lancome la vie'],
    'Aroma akuatik dan segar yang ikonik dengan sentuhan jeruk, melati, dan rempah-rempah. Memberikan kesan bersih, maskulin, dan energik (Giorgio Armani Acqua di Gio)': ['acqua di gio', 'adg'],
    'Aroma segar, oriental, dan sedikit vanila dengan sentuhan mint, apel hijau, dan tonka bean. Memberikan kesan kuat, menggoda, dan maskulin (Versace Eros)': ['versace eros', 'eros'],
    'Aroma hangat, mewah, dan kaya dengan dominasi tembakau, vanila, kakao, dan buah kering. Memberikan kesan dewasa, misterius, dan nyaman (Tom Ford Tobacco Vanille)': ['tom ford', 'tobacco vanille']
}

# =============================================================================
# KELAS UTAMA APLIKASI
# =============================================================================

class FullSentimentSystem(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Sistem Analisis Sentimen & Emosi Parfum")
        self.geometry("1366x768")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.main_tabview = ctk.CTkTabview(self, corner_radius=10)
        self.main_tabview.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        self.tab1 = self.main_tabview.add("Langkah 1: Persiapan & Koreksi Data")
        self.tab2 = self.main_tabview.add("Langkah 2: Latih Model & Analisis")
        self.setup_correction_tab()
        self.setup_analysis_tab()

    def setup_correction_tab(self):
        self.correction_raw_file_path = ""
        self.correction_df = None
        self.correction_index = 0
        self.tab1.grid_columnconfigure(0, weight=1)
        self.tab1.grid_rowconfigure(2, weight=1)
        file_selection_frame = ctk.CTkFrame(self.tab1)
        file_selection_frame.grid(row=0, column=0, padx=20, pady=20, sticky="ew")
        ctk.CTkLabel(file_selection_frame, text="Pilih File CSV Mentah (Tanpa Label):").grid(row=0, column=0, padx=10, pady=10)
        ctk.CTkButton(file_selection_frame, text="Pilih File...", command=self.select_correction_file).grid(row=0, column=1, padx=10, pady=10)
        self.correction_file_label = ctk.CTkLabel(file_selection_frame, text="Belum ada file", text_color="gray")
        self.correction_file_label.grid(row=0, column=2, padx=10, pady=10)
        self.run_bootstrap_button = ctk.CTkButton(self.tab1, text="Buat Label Awal & Mulai Koreksi", height=40, command=self.run_bootstrap_and_start_correction, state="disabled")
        self.run_bootstrap_button.grid(row=1, column=0, padx=20, pady=10, sticky="ew")
        self.correction_ui_frame = ctk.CTkScrollableFrame(self.tab1, label_text="Area Koreksi Data")
        self.correction_ui_frame.grid(row=2, column=0, padx=20, pady=10, sticky="nsew")

    def select_correction_file(self):
        self.correction_raw_file_path = filedialog.askopenfilename(title="Pilih File CSV Mentah", filetypes=[("CSV files", "*.csv")])
        if self.correction_raw_file_path:
            self.correction_file_label.configure(text=self.correction_raw_file_path.split('/')[-1], text_color="white")
            self.run_bootstrap_button.configure(state="normal")
            
    def run_bootstrap_and_start_correction(self):
        try:
            df = pd.read_csv(self.correction_raw_file_path)
            if 'full_text' not in df.columns:
                messagebox.showerror("Error", "File CSV harus memiliki kolom bernama 'full_text'.")
                return
            df['cleaned_text'] = df['full_text'].apply(self.clean_text)
            df['initial_label'] = df['cleaned_text'].apply(self.label_otomatis_parfum)
            df['final_label'] = df['initial_label']
            self.correction_df = df
            self.correction_index = 0
            self.run_bootstrap_button.configure(state="disabled")
            self.build_correction_ui()
            self.display_correction_row()
        except Exception as e:
            messagebox.showerror("Error", f"Gagal memproses file:\n{e}")

    def build_correction_ui(self):
        for widget in self.correction_ui_frame.winfo_children():
            widget.destroy()
        self.correction_ui_frame.grid_columnconfigure(1, weight=1)
        self.correction_status_label = ctk.CTkLabel(self.correction_ui_frame, text="", font=ctk.CTkFont(size=14, weight="bold"))
        self.correction_status_label.grid(row=0, column=0, columnspan=3, pady=10)
        ctk.CTkLabel(self.correction_ui_frame, text="Teks Ulasan Asli:").grid(row=1, column=0, columnspan=3, padx=10, sticky="w")
        self.correction_text_display = ctk.CTkTextbox(self.correction_ui_frame, height=150, state="disabled")
        self.correction_text_display.grid(row=2, column=0, columnspan=3, padx=10, pady=5, sticky="ew")
        ctk.CTkLabel(self.correction_ui_frame, text="Label Tebakan Mesin:").grid(row=3, column=0, padx=10, pady=10, sticky="e")
        self.initial_label_display = ctk.CTkLabel(self.correction_ui_frame, text="", font=ctk.CTkFont(weight="bold"))
        self.initial_label_display.grid(row=3, column=1, padx=10, pady=10, sticky="w")
        ctk.CTkLabel(self.correction_ui_frame, text="Koreksi Anda (Final Label):").grid(row=4, column=0, padx=10, pady=10, sticky="e")
        self.final_label_var = ctk.StringVar()
        self.final_label_menu = ctk.CTkOptionMenu(self.correction_ui_frame, variable=self.final_label_var, values=["positif", "negatif", "netral"], command=self.save_current_correction)
        self.final_label_menu.grid(row=4, column=1, padx=10, pady=10, sticky="w")
        nav_frame = ctk.CTkFrame(self.correction_ui_frame, fg_color="transparent")
        nav_frame.grid(row=5, column=0, columnspan=3, pady=20)
        self.prev_button = ctk.CTkButton(nav_frame, text="<< Sebelumnya", command=lambda: self.navigate_correction(-1))
        self.prev_button.pack(side="left", padx=10)
        self.next_button = ctk.CTkButton(nav_frame, text="Selanjutnya >>", command=lambda: self.navigate_correction(1))
        self.next_button.pack(side="left", padx=10)
        self.save_corrected_button = ctk.CTkButton(self.correction_ui_frame, text="Simpan Semua Koreksi ke File CSV", height=40, command=self.save_corrected_file)
        self.save_corrected_button.grid(row=6, column=0, columnspan=3, padx=10, pady=20, sticky="ew")

    def display_correction_row(self):
        if self.correction_df is None or self.correction_index >= len(self.correction_df):
            self.correction_status_label.configure(text="Selesai! Semua data telah ditinjau.")
            return
        total_rows = len(self.correction_df)
        self.correction_status_label.configure(text=f"Menampilkan Ulasan {self.correction_index + 1} dari {total_rows}")
        current_row = self.correction_df.iloc[self.correction_index]
        self.correction_text_display.configure(state="normal")
        self.correction_text_display.delete("1.0", "end")
        self.correction_text_display.insert("1.0", current_row['full_text'])
        self.correction_text_display.configure(state="disabled")
        self.initial_label_display.configure(text=str(current_row['initial_label']).upper())
        self.final_label_var.set(current_row['final_label'])
        self.prev_button.configure(state="normal" if self.correction_index > 0 else "disabled")
        self.next_button.configure(state="normal" if self.correction_index < total_rows - 1 else "disabled")

    def save_current_correction(self, choice):
        if self.correction_df is not None:
            self.correction_df.loc[self.correction_df.index[self.correction_index], 'final_label'] = choice
            
    def navigate_correction(self, direction):
        self.save_current_correction(self.final_label_var.get())
        new_index = self.correction_index + direction
        if 0 <= new_index < len(self.correction_df):
            self.correction_index = new_index
            self.display_correction_row()

    def save_corrected_file(self):
        self.save_current_correction(self.final_label_var.get())
        if self.correction_df is not None:
            save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")], title="Simpan File yang Telah Dikoreksi", initialfile="data_latih_terkoreksi.csv")
            if save_path:
                self.correction_df.to_csv(save_path, index=False)
                messagebox.showinfo("Sukses", f"File berhasil disimpan.\n\nFile ini sekarang siap digunakan sebagai 'File Latih' di Langkah 2.")
        else:
            messagebox.showwarning("Peringatan", "Tidak ada data untuk disimpan.")

    def setup_analysis_tab(self):
        self.analysis_original_file_path = ""
        self.analysis_manual_file_path = ""
        self.final_df = None
        self.tab2.grid_columnconfigure(1, weight=1); self.tab2.grid_rowconfigure(0, weight=1)
        analysis_control_frame = ctk.CTkFrame(self.tab2, width=280, corner_radius=10)
        analysis_control_frame.grid(row=0, column=0, padx=20, pady=20, sticky="ns")
        analysis_control_frame.grid_propagate(False)
        ctk.CTkLabel(analysis_control_frame, text="Panel Analisis", font=ctk.CTkFont(size=20, weight="bold")).pack(pady=20)
        ctk.CTkLabel(analysis_control_frame, text="1. Pilih File CSV Asli (Mentah)", anchor="w").pack(padx=20, pady=(10, 5))
        ctk.CTkButton(analysis_control_frame, text="Pilih File...", command=lambda: self.select_analysis_file('original')).pack(padx=20, fill="x")
        self.analysis_original_label = ctk.CTkLabel(analysis_control_frame, text="Belum ada file", text_color="gray", wraplength=240)
        self.analysis_original_label.pack(padx=20, pady=5)
        ctk.CTkLabel(analysis_control_frame, text="2. Pilih File Latih (Hasil Koreksi)", anchor="w").pack(padx=20, pady=(10, 5))
        ctk.CTkButton(analysis_control_frame, text="Pilih File...", command=lambda: self.select_analysis_file('manual')).pack(padx=20, fill="x")
        self.analysis_manual_label = ctk.CTkLabel(analysis_control_frame, text="Belum ada file", text_color="gray", wraplength=240)
        self.analysis_manual_label.pack(padx=20, pady=5)
        self.analysis_run_button = ctk.CTkButton(analysis_control_frame, text="Latih Model & Terapkan", command=self.start_full_analysis_thread, state="disabled", height=40)
        self.analysis_run_button.pack(padx=20, pady=20, fill="x")
        self.analysis_status_label = ctk.CTkLabel(analysis_control_frame, text="", wraplength=240)
        self.analysis_status_label.pack(padx=20, pady=10)
        self.analysis_progressbar = ctk.CTkProgressBar(analysis_control_frame, mode='indeterminate')
        self.analysis_save_button = ctk.CTkButton(analysis_control_frame, text="Simpan Hasil Akhir", command=self.save_analysis_results, state="disabled")
        self.analysis_save_button.pack(padx=20, pady=20, fill="x", side="bottom")

        self.analysis_results_tabview = ctk.CTkTabview(self.tab2, corner_radius=10)
        self.analysis_results_tabview.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.analysis_results_tabview.add("Rekomendasi Parfum") 
        self.analysis_results_tabview.add("Kategori Wangi Populer")
        self.analysis_results_tabview.add("Evaluasi Model")
        self.analysis_results_tabview.add("Distribusi Sentimen")
        self.analysis_results_tabview.add("Distribusi Emosi")
        self.analysis_results_tabview.add("WordCloud")
        self.analysis_results_tabview.add("Data Latih Digunakan")
        self.analysis_results_tabview.add("Contoh Hasil")
        self.analysis_results_tabview.set("Rekomendasi Parfum") 

    def select_analysis_file(self, file_type):
        path = filedialog.askopenfilename(title=f"Pilih File {file_type.title()}", filetypes=[("CSV files", "*.csv")])
        if path:
            if file_type == 'original':
                self.analysis_original_file_path = path
                self.analysis_original_label.configure(text=path.split('/')[-1], text_color="white")
            elif file_type == 'manual':
                self.analysis_manual_file_path = path
                self.analysis_manual_label.configure(text=path.split('/')[-1], text_color="white")
        if self.analysis_original_file_path and self.analysis_manual_file_path:
            self.analysis_run_button.configure(state="normal")
            
    def start_full_analysis_thread(self):
        self.analysis_run_button.configure(state="disabled")
        self.analysis_save_button.configure(state="disabled")
        self.analysis_status_label.configure(text="Memulai analisis...")
        self.analysis_progressbar.pack(pady=5, padx=20, fill="x")
        self.analysis_progressbar.start()
        analysis_thread = threading.Thread(target=self.run_full_analysis)
        analysis_thread.start()

    def run_full_analysis(self):
        try:
            self.analysis_status_label.configure(text="Membaca & membersihkan data...")
            df_original = pd.read_csv(self.analysis_original_file_path)
            df_manual = pd.read_csv(self.analysis_manual_file_path)

            if 'final_label' not in df_manual.columns:
                messagebox.showerror("Error", "File data latih harus memiliki kolom bernama 'final_label' hasil dari koreksi di Langkah 1.")
                self.analysis_status_label.configure(text="Error: Kolom 'final_label' tidak ditemukan.", text_color="red")
                self.analysis_progressbar.stop(); self.analysis_progressbar.pack_forget()
                self.analysis_run_button.configure(state="normal")
                return

            self.display_training_data(df_manual)

            df_original['cleaned_text'] = df_original['full_text'].apply(self.clean_text)
            df_original.dropna(subset=['cleaned_text'], inplace=True)
            df_original = df_original[df_original['cleaned_text'] != '']

            self.analysis_status_label.configure(text="Mempersiapkan data latih...")
            df_manual.dropna(subset=['cleaned_text', 'final_label'], inplace=True)
            df_manual = df_manual[df_manual['cleaned_text'].str.strip() != '']
            X_train = df_manual['cleaned_text']
            y_train = df_manual['final_label']

            self.analysis_status_label.configure(text="Melatih model Machine Learning...")
            vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1,2))
            X_train_tfidf = vectorizer.fit_transform(X_train)
            model = MultinomialNB()
            model.fit(X_train_tfidf, y_train)

            self.analysis_status_label.configure(text="Mengevaluasi performa model...")
            y_pred_eval = model.predict(X_train_tfidf)
            report = classification_report(y_train, y_pred_eval, zero_division=0)
            cm = confusion_matrix(y_train, y_pred_eval, labels=model.classes_)
            self.display_evaluation(report, cm, model.classes_)

            self.analysis_status_label.configure(text="Memprediksi sentimen & emosi...")
            full_data_tfidf = vectorizer.transform(df_original['cleaned_text'])
            full_data_predictions = model.predict(full_data_tfidf)
            df_original['predicted_label'] = full_data_predictions

            df_original['predicted_emotion'] = df_original['cleaned_text'].apply(self.deteksi_emosi_parfum)
            
            df_original['kategori_wangi'] = df_original['cleaned_text'].apply(self.kategorikan_wangi)
            df_original['parfum_ditemukan'] = df_original['cleaned_text'].apply(self.deteksi_parfum)
            
            df_manual_subset = df_manual[['cleaned_text', 'final_label']]
            df_with_training_info = pd.merge(df_original, df_manual_subset, on='cleaned_text', how='left')
            self.final_df = df_with_training_info
            
            self.display_distribution(self.final_df['predicted_label'])
            self.display_emotion_distribution(self.final_df['predicted_emotion'])
            self.display_all_wordclouds(self.final_df)
            self.display_sample_data(self.final_df)
            
            self.display_perfume_recommendations(self.final_df)
            self.display_fragrance_category_popularity(self.final_df)


            self.analysis_status_label.configure(text="Analisis Selesai!", text_color="light green")
            self.analysis_save_button.configure(state="normal")

        except Exception as e:
            self.analysis_status_label.configure(text=f"Error: {e}", text_color="red")
            messagebox.showerror("Error", f"Terjadi kesalahan:\n{e}")
        finally:
            self.analysis_progressbar.stop()
            self.analysis_progressbar.pack_forget()
            self.analysis_run_button.configure(state="normal")

    def display_evaluation(self, report, cm, labels):
        tab = self.analysis_results_tabview.tab("Evaluasi Model")
        for widget in tab.winfo_children(): widget.destroy()
        ctk.CTkLabel(tab, text="Laporan Klasifikasi (pada data latih)", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)
        report_textbox = ctk.CTkTextbox(tab, height=200, font=("Courier New", 12))
        report_textbox.pack(fill="x", padx=10); report_textbox.insert("0.0", report); report_textbox.configure(state="disabled")
        fig, ax = plt.subplots(figsize=(6, 4)); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=labels, yticklabels=labels)
        ax.set_title("Confusion Matrix"); ax.set_xlabel("Predicted Label"); ax.set_ylabel("True Label")
        canvas = FigureCanvasTkAgg(fig, master=tab); canvas.draw(); canvas.get_tk_widget().pack(pady=10, fill="both", expand=True)

    def display_distribution(self, predicted_labels):
        tab = self.analysis_results_tabview.tab("Distribusi Sentimen")
        for widget in tab.winfo_children(): widget.destroy()
        fig, ax = plt.subplots(figsize=(6, 4)); counts = predicted_labels.value_counts(); sns.barplot(x=counts.index, y=counts.values, ax=ax, palette="viridis")
        ax.set_title("Distribusi Hasil Sentimen pada Seluruh Data"); ax.set_ylabel("Jumlah Teks")
        canvas = FigureCanvasTkAgg(fig, master=tab); canvas.draw(); canvas.get_tk_widget().pack(pady=10, fill="both", expand=True)
    
    def display_emotion_distribution(self, predicted_emotions):
        tab = self.analysis_results_tabview.tab("Distribusi Emosi")
        for widget in tab.winfo_children(): widget.destroy()
        fig, ax = plt.subplots(figsize=(6, 4)); counts = predicted_emotions.value_counts(); sns.barplot(x=counts.index, y=counts.values, ax=ax, palette="plasma")
        ax.set_title("Distribusi Hasil Emosi pada Seluruh Data"); ax.set_ylabel("Jumlah Teks")
        canvas = FigureCanvasTkAgg(fig, master=tab); canvas.draw(); canvas.get_tk_widget().pack(pady=10, fill="both", expand=True)

    def display_all_wordclouds(self, df):
        tab = self.analysis_results_tabview.tab("WordCloud")
        for widget in tab.winfo_children(): widget.destroy()
        scrollable_frame = ctk.CTkScrollableFrame(tab)
        scrollable_frame.pack(fill="both", expand=True)

        # Daftar Stopword untuk Bahasa Indonesia & Konteks Parfum
        from wordcloud import STOPWORDS as WC_STOPWORDS
        
        stopwords_indonesia = set(WC_STOPWORDS)
        custom_stopwords = {
            # Kata umum Bahasa Indonesia
            'dan', 'di', 'ke', 'dari', 'yang', 'ini', 'itu', 'dengan', 'juga', 'tapi', 'ya',
            'saya', 'aku', 'dia', 'mereka', 'kita', 'kami', 'anda', 'sih', 'kok', 'deh',
            'untuk', 'pada', 'karena', 'adalah', 'yaitu', 'namun', 'saat', 'setelah', 'sebelum',
            'seperti', 'telah', 'akan', 'atau', 'saja', 'lalu',
            
            # Kata umum dalam konteks ulasan
            'sangat', 'banget', 'sekali', 'aja', 'lumayan', 'cukup',
            'parfum', 'parfume', 'wangi', 'wanginya', 'aroma', 'aromanya', 'bau', 'baunya',
            'yg', 'ga', 'gak', 'ngga', 'gk', 'enggak', 'udah', 'sdh', 'kalo', 'kalau', 'biar',
            'review', 'beli', 'produk', 'botol', 'nya', 'recommended', 'rekomendasi'
        }
        stopwords_indonesia.update(custom_stopwords)

        def create_wordcloud_on_canvas(parent_frame, text_series, title):
            wc_frame = ctk.CTkFrame(parent_frame)
            wc_frame.pack(pady=10, padx=10, fill="x")
            ctk.CTkLabel(wc_frame, text=title, font=ctk.CTkFont(size=16, weight="bold")).pack(pady=(10, 5))
            full_text = ' '.join(text_series.astype(str))
            if full_text.strip():
                try:
                    # Menambahkan parameter "stopwords" ke WordCloud
                    wordcloud = WordCloud(width=800, height=400, background_color='#2b2b2b', 
                                          colormap='viridis', max_words=100, contour_color='steelblue', 
                                          contour_width=1, color_func=lambda *args, **kwargs: "white",
                                          stopwords=stopwords_indonesia).generate(full_text)
                                          
                    fig, ax = plt.subplots(figsize=(10, 5), facecolor='#2b2b2b')
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    canvas = FigureCanvasTkAgg(fig, master=wc_frame)
                    canvas.draw()
                    canvas.get_tk_widget().pack(pady=(0, 10), fill="x", expand=True)
                except Exception as e:
                    ctk.CTkLabel(wc_frame, text=f"Gagal membuat WordCloud: {e}").pack(pady=20)
            else:
                ctk.CTkLabel(wc_frame, text="Tidak ada data teks untuk menghasilkan WordCloud ini.").pack(pady=20)

        create_wordcloud_on_canvas(scrollable_frame, df['cleaned_text'], "WordCloud Keseluruhan")
        create_wordcloud_on_canvas(scrollable_frame, df[df['predicted_label'] == 'positif']['cleaned_text'], "WordCloud Sentimen Positif")
        create_wordcloud_on_canvas(scrollable_frame, df[df['predicted_label'] == 'negatif']['cleaned_text'], "WordCloud Sentimen Negatif")
        create_wordcloud_on_canvas(scrollable_frame, df[df['predicted_label'] == 'netral']['cleaned_text'], "WordCloud Sentimen Netral")

    def display_training_data(self, df):
        tab = self.analysis_results_tabview.tab("Data Latih Digunakan")
        for widget in tab.winfo_children(): widget.destroy()
        ctk.CTkLabel(tab, text=f"Data Latih yang Digunakan untuk Melatih Model ({len(df)} baris)", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)
        textbox = ctk.CTkTextbox(tab, font=("Courier New", 11)); textbox.pack(fill="both", expand=True, padx=10, pady=10)
        data_string = df[['full_text', 'final_label']].to_string(); textbox.insert("0.0", data_string); textbox.configure(state="disabled")

    def display_sample_data(self, df):
        tab = self.analysis_results_tabview.tab("Contoh Hasil")
        for widget in tab.winfo_children(): widget.destroy()
        sample_textbox = ctk.CTkTextbox(tab, font=("Courier New", 11)); sample_textbox.pack(fill="both", expand=True, padx=10, pady=10)
        kolom_tampilan = ['created_at', 'username', 'full_text', 'predicted_label', 'predicted_emotion', 'kategori_wangi', 'parfum_ditemukan']
        kolom_yang_ada = [kolom for kolom in kolom_tampilan if kolom in df.columns]
        sample_data = df[kolom_yang_ada].head(100).to_string()
        sample_textbox.insert("0.0", sample_data); sample_textbox.configure(state="disabled")

    def save_analysis_results(self):
        if self.final_df is not None:
            save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")], title="Simpan Hasil Analisis Akhir")
            if save_path:
                try:
                    self.final_df.to_csv(save_path, index=False)
                    messagebox.showinfo("Sukses", f"File berhasil disimpan di:\n{save_path}")
                except Exception as e:
                    messagebox.showerror("Error", f"Gagal menyimpan file:\n{e}")
        else:
            messagebox.showwarning("Peringatan", "Tidak ada data untuk disimpan.")

    def display_perfume_recommendations(self, df):
        tab = self.analysis_results_tabview.tab("Rekomendasi Parfum")
        for widget in tab.winfo_children(): widget.destroy()

        if 'parfum_ditemukan' not in df.columns or df['parfum_ditemukan'].dropna().empty:
            ctk.CTkLabel(tab, text="Tidak ada nama parfum spesifik yang terdeteksi di data.").pack(pady=20, padx=20)
            return

        df_exploded = df.explode('parfum_ditemukan').dropna(subset=['parfum_ditemukan'])
        
        if df_exploded.empty:
            ctk.CTkLabel(tab, text="Tidak ada nama parfum spesifik yang terdeteksi di data.").pack(pady=20, padx=20)
            return

        summary = df_exploded.groupby('parfum_ditemukan').size().reset_index(name='jumlah_disebut')
        summary = summary.sort_values(by='jumlah_disebut', ascending=False).head(10)

        textbox = ctk.CTkTextbox(tab, font=("Courier New", 12))
        textbox.pack(fill="both", expand=True, padx=10, pady=10)

        header_title = "--- Top 10 Parfum Paling Banyak Dibicarakan ---\n\n"
        
        if summary.empty:
            textbox.insert("0.0", header_title + "Tidak ada data parfum untuk ditampilkan.")
            textbox.configure(state="disabled")
            return

        # Menentukan lebar kolom pertama (Parfum) + padding
        col1_width = summary['parfum_ditemukan'].str.len().max() + 2
        col2_header = "Jumlah Disebut"
        
        # Membuat header tabel dengan pembatas "|"
        table_header = f"{'Parfum':<{col1_width}} | {col2_header}\n"
        # Membuat garis pemisah horizontal yang sesuai
        table_separator = f"{'-' * col1_width}-+-{'-' * len(col2_header)}\n"
        
        table_content = header_title
        table_content += table_header
        table_content += table_separator

        # Menambahkan setiap baris data ke tabel dengan format dan pembatas
        for _, row in summary.iterrows():
            parfum_name = row['parfum_ditemukan']
            count = row['jumlah_disebut']
            table_content += f"{parfum_name:<{col1_width}} | {count}\n"

        textbox.insert("0.0", table_content)
        textbox.configure(state="disabled")

    def display_fragrance_category_popularity(self, df):
        tab = self.analysis_results_tabview.tab("Kategori Wangi Populer")
        for widget in tab.winfo_children(): widget.destroy()

        if 'kategori_wangi' not in df.columns or df['kategori_wangi'].dropna().empty:
            ctk.CTkLabel(tab, text="Tidak ada kategori wangi yang terdeteksi di data.").pack(pady=20, padx=20)
            return

        df_exploded = df.explode('kategori_wangi').dropna(subset=['kategori_wangi'])
        
        if df_exploded.empty:
            ctk.CTkLabel(tab, text="Tidak ada kategori wangi yang terdeteksi di data.").pack(pady=20, padx=20)
            return

        summary = df_exploded.groupby('kategori_wangi').size().reset_index(name='jumlah_kemunculan')
        summary = summary.sort_values(by='jumlah_kemunculan', ascending=False)

        textbox = ctk.CTkTextbox(tab, font=("Courier New", 12))
        textbox.pack(fill="both", expand=True, padx=10, pady=10)

        header_title = "--- Popularitas Kategori Wangi (Berdasarkan Penyebutan) ---\n\n"

        if summary.empty:
            textbox.insert("0.0", header_title + "Tidak ada data kategori untuk ditampilkan.")
            textbox.configure(state="disabled")
            return

        # Menentukan lebar kolom pertama (Kategori) + padding
        col1_width = summary['kategori_wangi'].str.len().max() + 2
        col2_header = "Jumlah Kemunculan"

        # Membuat header tabel dengan pembatas "|"
        table_header = f"{'Kategori Wangi':<{col1_width}} | {col2_header}\n"
        # Membuat garis pemisah horizontal yang sesuai
        table_separator = f"{'-' * col1_width}-+-{'-' * len(col2_header)}\n"

        table_content = header_title
        table_content += table_header
        table_content += table_separator

        # Menambahkan setiap baris data
        for _, row in summary.iterrows():
            category_name = row['kategori_wangi']
            count = row['jumlah_kemunculan']
            table_content += f"{category_name:<{col1_width}} | {count}\n"

        textbox.insert("0.0", table_content)
        textbox.configure(state="disabled")

    @staticmethod
    def clean_text(text):
        if not isinstance(text, str): return ""
        text = text.lower(); text = re.sub(r'https?://\S+|www\.\S+', '', text); text = re.sub(r'@\w+', '', text); text = re.sub(r'#\w+', '', text); text = re.sub(r'\d+', '', text); text = re.sub(r'[^\w\s]', '', text); text = re.sub(r'\s+', ' ', text).strip()
        return text

    @staticmethod
    def label_otomatis_parfum(text):
        kamus_positif = {'wangi', 'tahan lama', 'awet', 'lembut', 'segar', 'fresh', 'unik', 'elegan', 'mewah', 'suka', 'enak', 'nagih', 'otentik', 'aweet', 'long lasting', 'compliment', 'pujian', 'mantap', 'best', 'original'}
        kamus_negatif = {'pusing', 'menyengat', 'pasaran', 'aneh', 'alkohol', 'apek', 'tidak tahan lama', 'ga awet', 'ngga awet', 'terlalu kuat', 'bikin enek', 'kecewa', 'palsu', 'kw', 'zonk'}
        if any(frasa in text for frasa in ['tidak wangi', 'kurang wangi', 'bukan selera']): return 'negatif'
        if any(frasa in text for frasa in ['sangat wangi', 'wangi banget']): return 'positif'
        score_pos = sum(1 for kata in kamus_positif if kata in text); score_neg = sum(1 for kata in kamus_negatif if kata in text)
        if score_pos > score_neg: return 'positif'
        elif score_neg > score_pos: return 'negatif'
        else: return 'netral'
    
    @staticmethod
    def deteksi_emosi_parfum(text):
        kamus_senang = {'suka', 'suka banget', 'enak', 'nagih', 'compliment', 'pujian', 'mantap', 'best', 'senang', 'happy', 'bahagia'}
        kamus_kagum = {'unik', 'elegan', 'mewah', 'long lasting', 'tahan lama', 'awet', 'original', 'berkelas'}
        kamus_kecewa = {'kecewa', 'zonk', 'tidak sesuai', 'ga awet', 'ngga awet', 'tidak tahan lama', 'palsu', 'kw', 'menyesal'}
        kamus_marah = {'pusing', 'menyengat', 'bikin enek', 'terlalu kuat', 'apek', 'aneh'}
        if any(kata in text for kata in kamus_kecewa): return 'Kecewa'
        if any(kata in text for kata in kamus_marah): return 'Marah'
        if any(kata in text for kata in kamus_senang): return 'Senang'
        if any(kata in text for kata in kamus_kagum): return 'Kagum'
        return 'Netral'
        
    @staticmethod
    def kategorikan_wangi(text):
        kategori_ditemukan = set()
        for kategori, keywords in KAMUS_WANGI.items():
            if any(keyword in text for keyword in keywords):
                kategori_ditemukan.add(kategori)
        return list(kategori_ditemukan) if kategori_ditemukan else None

    @staticmethod
    def deteksi_parfum(text):
        parfum_ditemukan = set()
        for nama_parfum, keywords in KAMUS_PARFUM.items():
            if any(keyword in text for keyword in keywords):
                parfum_ditemukan.add(nama_parfum)
        return list(parfum_ditemukan) if parfum_ditemukan else None


if __name__ == "__main__":
    app = FullSentimentSystem()
    app.mainloop()