import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================
# KONFIGURASI HALAMAN
# ===============================
st.set_page_config(
    page_title="Analisis Keterlambatan Pengiriman",
    layout="wide"
)

st.title("📦 Analisis Keterlambatan Pengiriman")
st.markdown("---")

# ===============================
# LOAD DATA (TANPA UPLOAD)
# ===============================
@st.cache_data
def load_data():
    return pd.read_csv("train.csv")   # ganti jika nama file berbeda

df = load_data()

# ===============================
# PREVIEW DATA
# ===============================
st.subheader("📊 Preview Dataset")
st.dataframe(df.head())

st.subheader("📌 Informasi Dataset")
col1, col2 = st.columns(2)

with col1:
    st.write("**Tipe Data**")
    st.dataframe(df.dtypes)

with col2:
    st.write("**Statistik Deskriptif**")
    st.dataframe(df.describe().round(2))

# ===============================
# MISSING VALUES
# ===============================
st.subheader("🚨 Missing Values")

missing_df = pd.DataFrame({
    "Jumlah Missing": df.isnull().sum(),
    "Persentase (%)": (df.isnull().mean() * 100).round(2)
})

st.dataframe(missing_df[missing_df["Jumlah Missing"] > 0])

# ===============================
# ANALISIS UNIVARIAT NUMERIK
# ===============================
st.subheader("📈 Analisis Univariat (Numerik)")

num_cols = df.select_dtypes(include=np.number).columns
selected_num = st.selectbox("Pilih Variabel Numerik", num_cols)

fig, ax = plt.subplots()
sns.histplot(df[selected_num], kde=True, ax=ax)
ax.set_title(f"Distribusi {selected_num}")
st.pyplot(fig)

# ===============================
# ANALISIS KATEGORIKAL
# ===============================
cat_cols = df.select_dtypes(include="object").columns

if len(cat_cols) > 0:
    st.subheader("📊 Analisis Variabel Kategorikal")
    selected_cat = st.selectbox("Pilih Variabel Kategorikal", cat_cols)

    fig, ax = plt.subplots()
    df[selected_cat].value_counts().plot(kind="bar", ax=ax)
    ax.set_title(f"Distribusi {selected_cat}")
    st.pyplot(fig)

# ===============================
# KORELASI
# ===============================
st.subheader("🔗 Matriks Korelasi")

corr = df[num_cols].corr()

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# ===============================
# KESIMPULAN
# ===============================
st.subheader("📝 Kesimpulan")
st.markdown("""
- Dataset berhasil dimuat secara lokal tanpa upload
- Distribusi data menunjukkan variasi pada beberapa variabel
- Korelasi antar variabel mayoritas lemah hingga sedang
- Analisis ini dapat dikembangkan ke tahap pemodelan
""")
