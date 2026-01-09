import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import gradio as gr
import warnings

# --- CONFIGURACIÓN Y MOTOR (Se mantiene igual) ---
warnings.filterwarnings("ignore", category=UserWarning)

try:
    df_global = pd.read_csv('fra_cleaned_prueba_sincomas.csv')
    df_stock_ref = pd.read_csv('stock_de_la_perfumeria.xlsx - Completo (1).csv')
except FileNotFoundError:
    print("⚠️ Error: Archivos no encontrados.")

def limpiar_notas(text):
    if pd.isna(text): return ""
    text = str(text).lower()
    items_to_remove = ['[', ']', '"', '{', '}', 'middle: ', 'top: ', 'base: ', 'null']
    for item in items_to_remove: text = text.replace(item, "")
    return text.strip()

df_global['notes_clean'] = df_global['notes'].apply(limpiar_notas)
df_global['brand'] = df_global['brand'].str.strip()
df_global['perfume'] = df_global['perfume'].str.strip()
df_global['key'] = df_global['brand'].str.lower() + "|" + df_global['perfume'].str.lower()
df_stock_ref['key'] = df_stock_ref['brand'].str.strip().str.lower() + "|" + df_stock_ref['perfume'].str.strip().str.lower()
indices_en_stock = df_global[df_global['key'].isin(df_stock_ref['key'])].index.tolist()

cv = CountVectorizer(tokenizer=lambda x: x.split(' '), token_pattern=None)
matrix_bow = cv.fit_transform(df_global['notes_clean'])

def update_perfumes(marca):
    nombres = sorted(df_global[df_global['brand'] == marca]['perfume'].unique().tolist())
    return gr.Dropdown(choices=nombres, value=None, interactive=True)

def recomendar(marca_sel, perfume_sel):
    if not marca_sel or not perfume_sel:
        return pd.DataFrame({"Aviso": ["Selecciona marca y perfume"]})
    try:
        idx_base = df_global[(df_global['brand'] == marca_sel) & (df_global['perfume'] == perfume_sel)].index[0]
        vector_base = matrix_bow[idx_base]
        sims = cosine_similarity(vector_base, matrix_bow).flatten()
        
        recomendaciones = []
        for idx in indices_en_stock:
            if idx == idx_base: continue
            p_sim = sims[idx]
            if p_sim > 0:
                recomendaciones.append({
                    'Marca': df_global.iloc[idx]['brand'].upper(),
                    'Perfume': df_global.iloc[idx]['perfume'],
                    'Similitud_Val': p_sim,
                    'Notas': df_global.iloc[idx]['notes_clean']
                })
        
        if not recomendaciones:
            return pd.DataFrame({"Mensaje": ["No se encontraron similares en stock"]})

        df_res = pd.DataFrame(recomendaciones).sort_values(by='Similitud_Val', ascending=False).head(7)
        df_res['Similitud'] = df_res['Similitud_Val'].map("{:.0%}".format)
        df_res['Notas'] = df_res['Notas'].str.replace(' ', ', ')
        return df_res[['Marca', 'Perfume', 'Similitud', 'Notas']]
    except Exception as e:
        return pd.DataFrame({"Error": [str(e)]})

# --- ESTILO "PERFUMERÍA REAL" (Basado en la imagen) ---

css_custom = """
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;800&display=swap');

body { background-color: #ffffff; font-family: 'Poppins', sans-serif; }
.gradio-container { background-color: #ffffff !important; }

/* Títulos */
.title_container { text-align: center; margin-bottom: 30px; }
.title_container h1 { font-weight: 800; font-size: 2.8rem; color: #000000; margin-bottom: 0; }
.title_container p { font-weight: 300; color: #666; font-size: 1.1rem; }

/* Botón Estilo Dorado Mate */
.btn_gold { 
    background-color: #b0965d !important; 
    color: white !important; 
    border-radius: 8px !important; 
    border: none !important;
    font-weight: 600 !important;
    padding: 10px 20px !important;
}
.btn_gold:hover { background-color: #967f4a !important; }

/* Inputs Estilo Imagen */
.gr-input, .gr-dropdown {
    border-radius: 6px !important;
    border: 1px solid #ccc !important;
    background-color: #fcfcfc !important;
}

/* Tabla de Resultados */
.table-wrap { 
    border: 2px solid #b0965d !important; 
    border-radius: 10px !important; 
    overflow: hidden !important;
}
thead th { 
    background-color: #f7f1e3 !important; 
    color: #000000 !important; 
    font-weight: 700 !important;
    text-transform: uppercase;
}
"""

marcas_list = sorted(df_global['brand'].unique().tolist())

with gr.Blocks(css=css_custom, title="Perfumería Real") as demo:
    
    # Header (Logotipo simulado y Menú)
    with gr.Row():
        gr.HTML("""
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 10px 0; border-bottom: 1px solid #eee;">
                <div style="font-size: 24px; font-weight: 800; color: #000;">
                    <span style="color: #b0965d; font-size: 32px;">P</span>R Perfumeria Real
                </div>
                <div style="display: flex; gap: 20px; color: #666; font-weight: 400;">
                    <span>Inicio</span><span>Productos</span><span>Contacto</span>
                </div>
            </div>
        """)

    with gr.Column(elem_classes="title_container"):
        gr.Markdown("# Descubre tu Aroma Perfecto")
        gr.Markdown("Encuentra la fragancia ideal en nuestra colección exclusiva")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Selecciona la marca")
            drop_marca = gr.Dropdown(choices=marcas_list, label=None, container=False)
            
            gr.Markdown("### 2. Elige el perfume")
            drop_perfume = gr.Dropdown(choices=[], label=None, container=False, interactive=True)
            
            btn = gr.Button("BUSCAR SIMILARES", elem_classes="btn_gold")
        
        with gr.Column(scale=2):
            out_table = gr.Dataframe(
                headers=["Marca", "Perfume", "Similitud", "Notas"],
                interactive=False,
                wrap=True
            )

    # Lógica de interacción
    drop_marca.change(fn=update_perfumes, inputs=drop_marca, outputs=drop_perfume)
    btn.click(fn=recomendar, inputs=[drop_marca, drop_perfume], outputs=out_table)

if __name__ == "__main__":
    demo.launch()
