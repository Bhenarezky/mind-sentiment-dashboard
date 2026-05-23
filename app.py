import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob

# --- CONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Mind Sentiment - Burnout Detection Dashboard",
    page_icon="🧠",
    layout="wide"
)

# --- FUNGSI FEATURE ENGINEERING INSTAN ---
def hitung_fitur_linguistik(teks):
    """Melakukan feature engineering ekstraksi metrik teks secara real-time"""
    if not teks:
        return 0, 0.0
    jumlah_kata = len(str(teks).split())
    skor_subjektivitas = TextBlob(str(teks)).sentiment.subjectivity
    return jumlah_kata, skor_subjektivitas

# --- LOAD DATASET ---
@st.cache_data # Menghemat memori agar load data lebih cepat
def load_data():
    # Pastikan file dataset_label.csv berada di folder yang sama
    df = pd.read_csv("data/dataset_label.csv")
    return df

try:
    df_clean = load_data()
except FileNotFoundError:
    st.error("Gagal memuat data. Pastikan file 'dataset_label.csv' sudah diletakkan di direktori yang sama!")
    st.stop()

# --- SIDEBAR & NAVIGASI ---
st.sidebar.title("Navigasi Dasbor")
menu = st.sidebar.radio("Pilih Menu:", ["Ringkasan Data", "Analisis Distribusi & Insight", "Uji Fitur (Real-time Prediction)"])

# --- JUDUL UTAMA ---
st.title("🧠 Proyek Mind Sentiment: Deteksi Dini Risiko Burnout")
st.markdown("Dasbor interaktif analisis sentimen berbasis teks untuk mengidentifikasi indikasi kelelahan mental.")
st.write("---")

# ==================== MENU 1: RINGKASAN DATA ====================
if menu == "Ringkasan Data":
    st.subheader("📊 Sampel Dataset Hasil Preprocessing")
    st.dataframe(df_clean.head(10), use_container_width=True)
    
    # Statistik Sederhana
    st.subheader("📉 Ringkasan Total Data Per Sentimen")
    sentimen_counts = df_clean['label'].value_counts()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Sampel", f"{len(df_clean):,}")
    col2.metric("Teks Indikasi Stres (Negative)", f"{sentimen_counts.get('negative', 0):,}")
    col3.metric("Teks Normal/Harian (Neutral)", f"{sentimen_counts.get('neutral', 0):,}")

# ==================== MENU 2: ANALISIS DISTRIBUSI & INSIGHT ====================
elif menu == "Analisis Distribusi & Insight":
    st.subheader("📈 Visualisasi Hasil Feature Engineering")
    
    # Hitung fitur linguistik pada dataset untuk kebutuhan visualisasi dasbor
    if 'word_count' not in df_clean.columns:
        df_clean['word_count'] = df_clean['text_clean'].apply(lambda x: len(str(x).split()))
    if 'subjectivity' not in df_clean.columns:
        df_clean['subjectivity'] = df_clean['text_clean'].apply(lambda x: TextBlob(str(x)).sentiment.subjectivity)

    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Perbandingan Panjang Teks (Word Count) per Sentimen**")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(x='label', y='word_count', data=df_clean, palette={'negative':'#ef4444', 'positive':'#22c55e', 'neutral':'#f59e0b'}, ax=ax)
        ax.set_ylim(0, 40)
        st.pyplot(fig)
        st.info("💡 **Insight:** Sentimen *Negative* cenderung memiliki distribusi jumlah kata yang lebih panjang. Ini menandakan pengguna yang terindikasi stres cenderung menulis curhatan yang lebih detail.")

    with col2:
        st.write("**Tingkat Subjektivitas Teks Berdasarkan Sentimen**")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.violinplot(x='label', y='subjectivity', data=df_clean, palette={'negative':'#ef4444', 'positive':'#22c55e', 'neutral':'#f59e0b'}, ax=ax)
        st.pyplot(fig)
        st.info("💡 **Insight:** Teks beralur emosi negatif atau positif memiliki skor subjektivitas yang condong ke rentang tinggi (0.5 - 1.0), sedangkan teks netral bersifat lebih objektif/faktual.")

# ==================== MENU 3: UJI FITUR (REAL-TIME PREDICTION) ====================
elif menu == "Uji Fitur (Real-time Prediction)":
    st.subheader("🔬 Simulasi Ekstraksi Fitur Teks secara Instan")
    st.write("Masukkan kalimat atau keluhan untuk melihat bagaimana sistem *Data Scientist* mengekstrak fitur teks secara langsung.")
    
    user_input = st.text_area("Masukkan teks di sini (Contoh: 'I feel so overwhelmed with this study routine'):", "")
    
    if user_input:
        word_cnt, subj_score = hitung_fitur_linguistik(user_input)
        
        # Tampilkan hasil feature engineering
        st.markdown("### 🛠️ Hasil Ekstraksi Fitur (Feature Engineering)")
        col1, col2 = st.columns(2)
        col1.metric(label="Jumlah Kata (Word Count)", value=word_cnt)
        col2.metric(label="Skor Subjektivitas (Subjectivity Score)", value=f"{subj_score:.4f}")
        
        # Logika klasifikasi rule-based sederhana sebagai contoh interaktif dasbor
        st.markdown("### 🎯 Analisis Prediksi Sederhana")
        if "hurt" in user_input.lower() or "overwhelmed" in user_input.lower() or "stres" in user_input.lower():
            st.error("🚨 Hasil Klasifikasi: **HIGH RISK OF BURNOUT (Negative Emotion)**")
        elif subj_score < 0.25:
            st.warning("🟡 Hasil Klasifikasi: **NEUTRAL / EXPRESSIONS OF DAILY ROUTINE**")
        else:
            st.success("🟢 Hasil Klasifikasi: **WELL-BEING CONDITION (Positive Emotion)**")