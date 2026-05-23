import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import os

# Mengunduh korpus data TextBlob otomatis untuk keperluan pengujian real-time di server cloud
os.system("python -m textblob.download_corpora")

# --- CONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Mind Sentiment - Burnout Detection Dashboard",
    page_icon="🧠",
    layout="wide"
)

# --- LOAD DATASET ---
@st.cache_data
def load_data():
    # Memuat data dari folder data/
    df = pd.read_csv("data/dataset_label.csv")
    
    # [FEATURE ENGINEERING OTOMATIS]
    # Jika di komputer lokal Anda kolom ini belum ada, sistem akan membuatnya secara otomatis
    if 'word_count' not in df.columns:
        df['word_count'] = df['text_clean'].apply(lambda x: len(str(x).split()))
        
    if 'subjectivity' not in df.columns:
        df['subjectivity'] = df['text_clean'].apply(lambda x: TextBlob(str(x)).sentiment.subjectivity)
        
    return df

# Memanggil fungsi load data
df_clean = load_data()

# --- SIDEBAR & NAVIGASI ---
st.sidebar.title("🧠 Mind Sentiment")
st.sidebar.markdown("Dashboard Deteksi Risiko Burnout")
st.sidebar.write("---")
menu = st.sidebar.radio("Pilih Menu:", ["Ringkasan Data", "Analisis Distribusi & Insight (EDA)", "Uji Fitur (Real-time Prediction)"])

# --- JUDUL UTAMA ---
st.title("🧠 Proyek Mind Sentiment: Deteksi Dini Risiko Burnout")
st.markdown("Dasbor interaktif analisis sentimen berbasis teks untuk mengidentifikasi indikasi kelelahan mental.")
st.write("---")

# ==================== MENU 1: RINGKASAN DATA ====================
if menu == "Ringkasan Data":
    st.subheader("📊 Sampel Dataset Hasil Feature Engineering")
    st.write("Berikut adalah 10 baris pertama data teks yang telah dibersihkan beserta metrik linguistiknya:")
    st.dataframe(df_clean.head(10), use_container_width=True)
    
    st.write("---")
    st.subheader("📈 Ringkasan Metrik Data Per Sentimen")
    sentimen_counts = df_clean['label'].value_counts()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Sampel Data", f"{len(df_clean):,}")
    col2.metric("Teks Stres (Negative)", f"{sentimen_counts.get('negative', 0):,}", delta="High Risk", delta_color="inverse")
    col3.metric("Teks Normal (Neutral)", f"{sentimen_counts.get('neutral', 0):,}", delta="Baseline")
    col4.metric("Teks Sehat (Positive)", f"{sentimen_counts.get('positive', 0):,}", delta="Well-being")

# ==================== MENU 2: ANALISIS DISTRIBUSI & INSIGHT (EDA) ====================
elif menu == "Analisis Distribusi & Insight (EDA)":
    st.subheader("📈 Visualisasi Hasil Exploratory Data Analysis & Pertanyaan Bisnis")
    st.write("Menu ini menyajikan grafik analitis untuk menjawab 4 pertanyaan bisnis utama terkait karakteristik data teks indikasi burnout.")
    
    # Membuat Tab interaktif berdasarkan Pertanyaan Bisnis
    tab1, tab2, tab3, tab4 = st.tabs([
        "📌 Q1: Dominansi Emosi",  
        "📏 Q2: Panjang Teks (Word Count)", 
        "🔮 Q3: Skor Subjektivitas Teks",
        "📊 Q4: Kecukupan Kelas Neutral"
    ])
    
    # --- TAB 1: PERTANYAAN BISNIS 1 ---
    with tab1:
        st.markdown("#### **Pertanyaan 1: Dari 28 kategori emosi GoEmotions, emosi negatif mana yang paling dominan dan berpotensi menjadi indikator utama risiko burnout?**")
        
        # Data static dari output Cell 21 notebook Anda
        top_emotions_data = {
            'neutral': 15470, 'approval': 4838, 'admiration': 4836, 
            'annoyance': 3471, 'gratitude': 3450, 'disapproval': 3062, 
            'curiosity': 2770, 'amusement': 2640, 'optimism': 2428, 'realization': 2359
        }
        df_top_emo = pd.DataFrame(list(top_emotions_data.items()), columns=['Emosi', 'Jumlah'])
        
        fig, ax = plt.subplots(figsize=(10, 4.5))
        sns.barplot(x='Jumlah', y='Emosi', data=df_top_emo, palette='viridis', ax=ax)
        ax.set_title('Top 10 Kategori Emosi Terbanyak dalam Dataset Utama')
        ax.set_xlabel('Jumlah Teks')
        ax.grid(axis='x', linestyle='--', alpha=0.5)
        st.pyplot(fig)
        
        st.success("""
        💡 **Insight & Jawaban Pertanyaan 1:**
        Berdasarkan hasil visualisasi distribusi frekuensi, setelah kelas *neutral*, emosi negatif yang **paling dominan muncul adalah annoyance (kekesalan) dan disapproval (ketidaksetujuan/penolakan)**, kemudian diikuti secara signifikan oleh *sadness* (kesedihan) dan *disappointment* (kekecewaan). 
        Kategori-kategori ini merupakan indikator utama yang merepresentasikan rasa frustrasi akibat beban kerja/studi berlebih, sehingga sangat valid digunakan sebagai penanda risiko burnout tinggi.
        """)

    # --- TAB 2: PERTANYAAN BISNIS 2 ---
    with tab2:
        st.markdown("#### **Pertanyaan 2: Bagaimana perbandingan rata-rata panjang teks (jumlah kata) antara kelompok emosi positif dengan kelompok emosi negatif?**")
        
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.boxplot(x='label', y='word_count', data=df_clean, palette={'negative':'#ef4444', 'neutral':'#f59e0b', 'positive':'#22c55e'}, ax=ax)
        ax.set_title('Perbandingan Panjang Teks (Word Count) Berdasarkan Sentimen')
        ax.set_xlabel('Sentimen')
        ax.set_ylabel('Jumlah Kata')
        ax.set_ylim(0, 45) # Membatasi agar outlier ekstrem tidak merusak skala visualisasi boxplot
        st.pyplot(fig)
        
        st.warning("""
        💡 **Insight & Jawaban Pertanyaan 2:**
        Berdasarkan boxplot di atas, terlihat bahwa **Sentimen Negative memiliki median dan kuartil atas yang paling tinggi** dibandingkan kelompok lainnya. 
        
        **Interpretasi Data Science:** Pengguna yang sedang mengalami emosi negatif (stres, kesal, sedih) cenderung menulis teks yang lebih panjang untuk meluapkan perasaan atau menjelaskan konteks masalah mereka (curhat). Ini menandakan fitur `word_count` merupakan salah satu fitur pendukung yang sangat kuat dalam memisahkan teks keluhan burnout dari ekspresi harian biasa.
        """)

    # --- TAB 3: PERTANYAAN BISNIS 3 ---
    with tab3:
        st.markdown("#### **Pertanyaan 3: Apakah teks dengan tingkat subjektivitas yang lebih tinggi cenderung berasal dari kategori emosi yang berhubungan dengan tekanan mental?**")
        
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.violinplot(x='label', y='subjectivity', data=df_clean, palette={'negative':'#ef4444', 'neutral':'#f59e0b', 'positive':'#22c55e'}, ax=ax)
        ax.set_title('Perbandingan Tingkat Subjektivitas Berdasarkan Sentimen')
        ax.set_xlabel('Sentimen')
        ax.set_ylabel('Subjectivity Score')
        st.pyplot(fig)
        
        st.success("""
        💡 **Insight & Jawaban Pertanyaan 3:**
        Berdasarkan bentuk violin plot di atas, pola sebaran data menunjukkan karakteristik yang sangat jelas:
        - **Sentimen Negative dan Positive memiliki distribusi yang melebar ke arah skor 1.0 (sangat subjektif)** dengan kerapatan tertinggi di rentang 0.5 - 0.8.
        - **Sentimen Neutral mengumpul di area bawah (0.0 - 0.5)** yang menandakan sifat teks objektif/faktual.
        
        **Interpretasi Data Science:**
        Teks yang mengandung emosi negatif terkait tekanan mental/burnout bersifat sangat subjektif karena diisi oleh opini personal dan luapan perasaan pribadi. Hal ini membuktikan bahwa metrik `subjectivity` score adalah fitur hasil *feature engineering* yang sangat valid dan akurat untuk mendeteksi tanda kelelahan emosional.
        """)
    
    # --- TAB 4: PERTANYAAN BISNIS 4 ---
    with tab4:
        st.markdown("#### **Pertanyaan 4: Apakah jumlah sampel pada kategori emosi 'neutral' sudah cukup memadai untuk membedakan ekspresi harian normal dengan kelelahan mental?**")
        
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.countplot(x='label', data=df_clean, palette={'negative':'#ef4444', 'neutral':'#f59e0b', 'positive':'#22c55e'}, order=['neutral', 'negative', 'positive'], ax=ax)
        ax.set_title('Distribusi Jumlah Data per Sentimen Utama')
        ax.set_xlabel('Sentimen')
        ax.set_ylabel('Jumlah Sampel')
        st.pyplot(fig)
        
        st.info("""
        💡 **Insight & Jawaban Pertanyaan 4:**
        Kelas **neutral memiliki jumlah sampel yang sangat besar dan seimbang (~21.982 data)** jika dibandingkan dengan kelas *negative* (~23.106 data). 
        Jumlah ini **sangat memadai dan ideal** untuk dijadikan sebagai *baseline* (kelas pembanding). Dengan demikian, model AI memiliki referensi data yang cukup untuk mempelajari pola kalimat harian normal yang tidak mengandung indikasi stres/burnout.
        """)
    

# ==================== MENU 3: UJI FITUR (REAL-TIME PREDICTION) ====================
elif menu == "Uji Fitur (Real-time Prediction)":
    st.subheader("🔬 Uji Ekstraksi Fitur Teks secara Instan")
    st.write("Masukkan kalimat atau keluhan untuk melihat bagaimana sistem mengekstrak fitur teks secara langsung.")
    
    user_input = st.text_area("Masukkan teks di sini (Contoh: 'I feel so overwhelmed with this study routine'):", "")
    
    if user_input:
        # Ekstraksi Fitur Linguistik secara Real-time
        word_cnt = len(str(user_input).split())
        subj_score = TextBlob(str(user_input)).sentiment.subjectivity
        
        st.markdown("### 🛠️ Hasil Ekstraksi Fitur (Feature Engineering)")
        col1, col2 = st.columns(2)
        col1.metric(label="Jumlah Kata (Word Count)", value=word_cnt)
        col2.metric(label="Skor Subjektivitas (Subjectivity Score)", value=f"{subj_score:.4f}")
        
        st.markdown("### 🎯 Analisis Prediksi Sederhana")
        if "hurt" in user_input.lower() or "overwhelmed" in user_input.lower() or "stres" in user_input.lower() or "sad" in user_input.lower():
            st.error("🚨 Hasil Klasifikasi: **HIGH RISK OF BURNOUT (Negative Emotion)**")
        elif subj_score < 0.3:
            st.warning("🟡 Hasil Klasifikasi: **NEUTRAL / EXPRESSIONS OF DAILY ROUTINE**")
        else:
            st.success("🟢 Hasil Klasifikasi: **WELL-BEING CONDITION (Positive Emotion)**")