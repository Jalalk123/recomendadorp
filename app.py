import streamlit as st
import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Perfumería Real", layout="centered")

# --- DISEÑO DORADO MATE (Inspirado en tu imagen) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }
    
    .stApp { background-color: #ffffff; }

   /* COLOR NEGRO SOLO PARA LOS LABELS DE SELECTBOX */
[data-testid="stWidgetLabel"] label,
.stSelectbox label,
.stMultiSelect label {
    color: #000000 !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
}


    /* Botones estilo dorado mate */
    .stButton > button {
        background-color: #b0965d !important;
        color: white !important;
        border-radius: 6px !important;
        padding: 0.7rem 2rem !important;
        border: none !important;
        width: 100% !important;
        font-weight: 600 !important;
        text-transform: uppercase;
    }
    
    /* Tarjetas de perfumes */
    .perfume-card {
        border-left: 5px solid #b0965d;
        background-color: #f9f9f9;
        padding: 20px;
        border-radius: 0 10px 10px 0;
        margin-bottom: 15px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# --- CARGA DE DATOS SEGURA ---
@st.cache_data
def load_data():
    file_global = 'fra_cleaned_prueba_sincomas.csv'
    file_stock = 'stock_de_la_perfumeria.xlsx - Completo (1).csv'
    
    if not os.path.exists(file_global) or not os.path.exists(file_stock):
        return None, None, f"Faltan archivos"

    df_g = pd.read_csv(file_global)
    df_s = pd.read_csv(file_stock)
    
    def limpiar(text):
        if pd.isna(text): return ""
        text = str(text).lower()
        for item in ['[', ']', '"', '{', '}', 'middle: ', 'top: ', 'base: ', 'null']:
            text = text.replace(item, "")
        return text.strip()

    df_g['notes_clean'] = df_g['notes'].apply(limpiar)
    df_g['brand'] = df_g['brand'].str.strip()
    df_g['perfume'] = df_g['perfume'].str.strip()
    df_g['key'] = df_g['brand'].str.lower() + "|" + df_g['perfume'].str.lower()
    df_s['key'] = df_s['brand'].str.strip().str.lower() + "|" + df_s['perfume'].str.strip().str.lower()
    
    indices = df_g[df_g['key'].isin(df_s['key'])].index.tolist()
    return df_g, indices, None

df_global, indices_stock, error_msg = load_data()

if not error_msg:
    cv = CountVectorizer(tokenizer=lambda x: x.split(' '), token_pattern=None)
    matrix_bow = cv.fit_transform(df_global['notes_clean'])

    # --- INTERFAZ ---
    st.markdown("<h1 style='text-align: center; color: #000; font-weight: 800; margin-bottom:0;'>Perfumeria Real</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #b0965d; font-weight: 300; font-size: 1.2rem;'>Todos tus perfumes en un solo lugar</p>", unsafe_allow_html=True)
    st.write("---")

    # Los labels de estos selectbox ahora aparecerán en negro por el CSS arriba
    marca = st.selectbox("1. Selecciona la marca de referencia", sorted(df_global['brand'].unique()))
    perfumes_f = sorted(df_global[df_global['brand'] == marca]['perfume'].unique())
    perfume = st.selectbox("2. Elige el perfume que te gusta", perfumes_f)

    if st.button("VER PRODUCTOS SIMILARES"):
        idx_base = df_global[(df_global['brand'] == marca) & (df_global['perfume'] == perfume)].index[0]
        sims = cosine_similarity(matrix_bow[idx_base], matrix_bow).flatten()
        
        recoms = []
        for idx in indices_stock:
            if idx == idx_base: continue
            if sims[idx] > 0:
                recoms.append({
                    'Marca': df_global.iloc[idx]['brand'].upper(),
                    'Perfume': df_global.iloc[idx]['perfume'],
                    'Similitud': sims[idx],
                    'Notas': df_global.iloc[idx]['notes_clean'].replace(' ', ', ')
                })
        
        if recoms:
            df_res = pd.DataFrame(recoms).sort_values(by='Similitud', ascending=False).head(5)
            st.markdown("### ✨ Recomendaciones encontradas en Stock:")
            for _, row in df_res.iterrows():
                st.markdown(f"""
                <div class="perfume-card">
                    <h3 style="margin:0; color:#b0965d;">{row['Perfume']}</h3>
                    <p style="margin:0; font-weight:600; color:#333;">{row['Marca']}</p>
                    <p style="font-size: 0.9rem; color: #666; margin-top:10px;"><b>Notas olfativas:</b> {row['Notas']}</p>
                    <p style="text-align:right; font-size:0.8rem; color:#b0965d; margin:0;">Coincidencia: {row['Similitud']:.0%}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("No encontramos fragancias similares en stock actualmente.")

