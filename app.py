import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import os

# --- IMPORT UTILITAS MODUL TERPISAH ---
from translation_service import translate_to_english
from preprocessing import full_preprocess

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Mind Sentiment - Burnout Detection Dashboard",
    page_icon="🧠",
    layout="wide"
)

# --- LOAD DATASET ---
@st.cache_data
def load_data():
    df = pd.read_csv("data/dataset_label.csv")
    if 'word_count' not in df.columns:
        df['word_count'] = df['text_clean'].apply(lambda x: len(str(x).split()))
    if 'subjectivity' not in df.columns:
        df['subjectivity'] = df['text_clean'].apply(lambda x: TextBlob(str(x)).sentiment.subjectivity)
    return df

df_clean = load_data()

# --- SIDEBAR & NAVIGASI ---
st.sidebar.title("🧠 Mind Sentiment")
st.sidebar.markdown("Dashboard Deteksi Risiko Burnout")
st.sidebar.write("---")
menu = st.sidebar.radio("Pilih Menu:", ["Ringkasan Data", "Analisis Distribusi & Insight (EDA)", "Uji Fitur (Real-time Prediction)"])

st.title("🧠 Proyek Mind Sentiment: Deteksi Dini Risiko Burnout")
st.write("---")

# ==================== MENU 1: RINGKASAN DATA ====================
if menu == "Ringkasan Data":
    st.subheader("📊 Sampel Dataset Hasil Feature Engineering")
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
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📌 Q1: Dominansi Emosi", 
        "📊 Q4: Kecukupan Kelas Neutral", 
        "📏 Q2: Panjang Teks (Word Count)", 
        "🔮 Q3: Skor Subjektivitas Teks"
    ])
    
    with tab1:
        st.markdown("#### **Pertanyaan 1: Dari 28 kategori emosi GoEmotions, emosi negatif mana yang paling dominan?**")
        top_emotions_data = {
            'neutral': 15470, 'approval': 4838, 'admiration': 4836, 
            'annoyance': 3471, 'gratitude': 3450, 'disapproval': 3062, 
            'curiosity': 2770, 'amusement': 2640, 'optimism': 2428, 'realization': 2359
        }
        df_top_emo = pd.DataFrame(list(top_emotions_data.items()), columns=['Emosi', 'Jumlah'])
        fig, ax = plt.subplots(figsize=(10, 4.5))
        sns.barplot(x='Jumlah', y='Emosi', data=df_top_emo, palette='viridis', ax=ax)
        ax.grid(axis='x', linestyle='--', alpha=0.5)
        st.pyplot(fig)
        
    with tab2:
        st.markdown("#### **Pertanyaan 4: Apakah jumlah sampel pada kategori emosi 'neutral' sudah memadai?**")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.countplot(x='label', data=df_clean, palette={'negative':'#ef4444', 'neutral':'#f59e0b', 'positive':'#22c55e', 'unknown':'gray'}, order=['neutral', 'negative', 'positive', 'unknown'], ax=ax)
        st.pyplot(fig)
        
    with tab3:
        st.markdown("#### **Pertanyaan 2: Bagaimana perbandingan rata-rata panjang teks antara emosi positif dengan emosi negatif?**")
        df_visual = df_clean.copy()
        df_visual['temp_sentiment'] = df_visual['label'].str.capitalize()
        notebook_palette = {'Negative': '#ef4444', 'Positive': '#22c55e', 'Neutral': '#f59e0b', 'Unknown': 'gray'}
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(x='temp_sentiment', y='word_count', data=df_visual, order=['Negative', 'Positive', 'Neutral', 'Unknown'], palette=notebook_palette, hue='temp_sentiment', legend=False, width=0.6, ax=ax)
        ax.set_ylim(0, 60)
        ax.grid(False)
        st.pyplot(fig)
        
    with tab4:
        st.markdown("#### **Pertanyaan 3: Apakah teks dengan tingkat subclass subjektivitas tinggi cenderung dari emosi tekanan mental?**")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.violinplot(x='label', y='subjectivity', data=df_clean, palette={'negative':'#ef4444', 'neutral':'#f59e0b', 'positive':'#22c55e', 'unknown':'gray'}, ax=ax)
        st.pyplot(fig)

# ==================== MENU 3: UJI FITUR (ALUR PIPELINE UTAMA) ====================
elif menu == "Uji Fitur (Real-time Prediction)":
    st.subheader("🔬 Laboratorium Uji Fitur & Klasifikasi")
    st.markdown("""
    Dashboard ini melakukan simulasi penuh alur kerja arsitektur sistem cerdas secara terpisah: 
    **Input -> `translation_service.py` -> `preprocessing.py` -> Feature Engineering -> Prediksi.**
    """)

    user_input = st.text_area(
        "Masukkan Kalimat Keluhan / Refleksi Harian (Bahasa Indonesia atau Inggris):", 
        placeholder="Contoh: Saya merasa sangat kewalahan..."
    )

    if st.button("Jalankan Pipeline Proses", type="primary"):
        if user_input.strip():
            
            # --- TAHAP 1: TRANSLASI ---
            with st.spinner("Mengecek bahasa & menerjemahkan melalui translation_service.py..."):
                translated_text = translate_to_english(user_input)
            
            # --- TAHAP 2: PIPELINE PREPROCESSING ---
            with st.spinner("Melakukan pembersihan data melalui full_preprocess di preprocessing.py..."):
                preprocessed_text = full_preprocess(translated_text)
            
            # --- TAHAP 3: FEATURE ENGINEERING ---
            word_cnt = len(str(preprocessed_text).split())
            subj_score = TextBlob(translated_text).sentiment.subjectivity
            
            # Rendering Alur Perubahan Data Teks
            st.markdown("### 🗺️ Visualisasi Alur Transformasi Data")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.info("📥 **1. Input Teks Asli**")
                st.caption(f"_{user_input}_")
            with col_b:
                st.success("🔄 **2. Hasil Translasi (English)**")
                st.caption(f"_{translated_text}_")
            with col_c:
                st.warning("🧼 **3. Hasil Teks Bersih (preprocessing.py)**")
                st.code(preprocessed_text, language="text")

            st.write("---")
            
            # Hasil Fitur Numerik
            st.markdown("### 🛠️ Hasil Ekstraksi Fitur (Feature Engineering)")
            m1, m2 = st.columns(2)
            m1.metric("Jumlah Kata Teks Bersih (Word Count)", word_cnt)
            m2.metric("Skor Subjektivitas (Subjectivity Score)", f"{subj_score:.4f}")

            # Aturan Klasifikasi Berbasis Kata Utuh Kompleks (Token)
            st.markdown("### 🎯 Hasil Analisis Risiko Burnout")
            
            tokens_clean = preprocessed_text.lower().split()
            tokens_translated = translated_text.lower().split()

            # Kamus Kata Kunci Pendeteksi
            burnout_keywords = ['tired', 'tire', 'exhaust', 'exhausted', 'overwhelm', 'overwhelmed', 'hurt', 'stress', 'annoy', 'annoyed', 'sad', 'disappoint', 'disappointed', 'lelah']
            positive_keywords = ['happy', 'happi', 'good', 'glad', 'love', 'blessed', 'grateful', 'joy', 'awesome', 'wonderful', 'bahagia', 'senang']

            has_burnout_word = any(word in tokens_clean or word in tokens_translated for word in burnout_keywords)
            has_positive_word = any(word in tokens_clean or word in tokens_translated for word in positive_keywords)

            # Aturan Logika Keputusan Akhir
            if has_burnout_word and not has_positive_word:
                st.error("🚨 **Hasil: HIGH RISK OF BURNOUT (Indikasi Emosi Negatif)**")
                st.markdown("**Rekomendasi:** Teks menunjukkan beban pikiran atau indikasi stres emosional yang tinggi. Ambil jeda istirahat yang cukup.")
            elif has_positive_word:
                st.success("🟢 **Hasil: WELL-BEING CONDITION (Emosi Positif)**")
                st.markdown("**Rekomendasi:** Kondisi emosional terpantau aman, sehat, dan dipenuhi energi positif.")
            elif subj_score < 0.28:
                st.warning("🟡 **Hasil: NEUTRAL / DAILY ROUTINE EXPRESSION**")
                st.markdown("**Rekomendasi:** Kalimat bersifat objektif, informatif, atau sekadar memaparkan fakta kegiatan rutin sehari-hari.")
            else:
                st.warning("🟡 **Hasil: NEUTRAL / BALANCE SENTIMENT**")
                st.markdown("**Rekomendasi:** Kalimat mengekspresikan opini umum harian biasa tanpa kecenderungan stres ekstrem.")
        else:
            st.error("Silakan tulis kalimat keluhan terlebih dahulu sebelum memproses!")