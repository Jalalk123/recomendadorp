import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import gradio as gr
import warnings

# Ignorar advertencias de Sklearn sobre el tokenizador
warnings.filterwarnings("ignore", category=UserWarning)

# --- 1. CARGA Y LIMPIEZA DE DATOS ---

# Cargar los archivos (Asegúrate de subirlos a la carpeta lateral de Colab)
try:
    df_global = pd.read_csv('fra_cleaned_prueba_sincomas.csv')
    df_stock_ref = pd.read_csv('stock_de_la_perfumeria.xlsx - Completo (1).csv')
except FileNotFoundError:
    print("⚠️ Error: Sube 'fra_cleaned_prueba_sincomas.csv' y 'stock_de_la_perfumeria.xlsx - Completo (1).csv' a Colab.")

# Función de limpieza (Lógica exacta de tu notebook)
def limpiar_notas(text):
    if pd.isna(text): return ""
    text = str(text).lower()
    # Eliminamos caracteres especiales y etiquetas de pirámide olfativa
    items_to_remove = ['[', ']', '"', '{', '}', 'middle: ', 'top: ', 'base: ', 'null']
    for item in items_to_remove:
        text = text.replace(item, "")
    return text.strip()

df_global['notes_clean'] = df_global['notes'].apply(limpiar_notas)

# Normalización para el cruce de datos
df_global['brand'] = df_global['brand'].str.strip()
df_global['perfume'] = df_global['perfume'].str.strip()
df_global['key'] = df_global['brand'].str.lower() + "|" + df_global['perfume'].str.lower()

df_stock_ref['key'] = df_stock_ref['brand'].str.strip().str.lower() + "|" + df_stock_ref['perfume'].str.strip().str.lower()

# Identificamos los índices de la base GLOBAL que están en STOCK
indices_en_stock = df_global[df_global['key'].isin(df_stock_ref['key'])].index.tolist()

# --- 2. VECTORIZACIÓN (IA) ---

def custom_tokenizer(text):
    # Separamos por espacios (las notas se limpian a un string separado por espacios)
    return text.split(' ')

# Creamos la bolsa de palabras (Bag of Words)
cv = CountVectorizer(tokenizer=custom_tokenizer, token_pattern=None)
matrix_bow = cv.fit_transform(df_global['notes_clean'])

# --- 3. FUNCIONES DE LA INTERFAZ ---

def update_perfumes(marca):
    """Actualiza la lista de perfumes cuando se selecciona una marca."""
    nombres = sorted(df_global[df_global['brand'] == marca]['perfume'].unique().tolist())
    return gr.Dropdown(choices=nombres, value=None, interactive=True)

def recomendar(marca_sel, perfume_sel):
    """Calcula similitud y filtra por stock."""
    if not marca_sel or not perfume_sel:
        return pd.DataFrame({"Aviso": ["Selecciona una marca y un perfume primero"]})

    try:
        # 1. Índice del perfume seleccionado en el catálogo mundial
        idx_base = df_global[(df_global['brand'] == marca_sel) & 
                             (df_global['perfume'] == perfume_sel)].index[0]
        
        vector_base = matrix_bow[idx_base]
        
        # 2. Calcular similitud contra TODO el catálogo mundial
        sims = cosine_similarity(vector_base, matrix_bow).flatten()
        
        # 3. Filtrar para mostrar SOLO resultados que estén EN STOCK
        recomendaciones = []
        for idx in indices_en_stock:
            # No recomendar el mismo perfume que se buscó
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

        # 4. Crear DataFrame, ordenar y formatear (Top 5)
        df_res = pd.DataFrame(recomendaciones)
        df_res = df_res.sort_values(by='Similitud_Val', ascending=False).head(5)
        
        # Formato de porcentaje (%)
        df_res['Similitud'] = df_res['Similitud_Val'].map("{:.2%}".format)
        # Reemplazar espacios por comas para lectura humana
        df_res['Notas'] = df_res['Notas'].str.replace(' ', ', ')
        
        return df_res[['Marca', 'Perfume', 'Similitud', 'Notas']]

    except Exception as e:
        return pd.DataFrame({"Error": [str(e)]})

# --- 4. CONSTRUCCIÓN DE GRADIO ---

marcas_list = sorted(df_global['brand'].unique().tolist())

with gr.Blocks(theme='freddyaboulton/dracula_revamped') as demo:
    gr.Markdown("# 🧪 Recomendador de Perfumes: Mundo ➔ Stock")
    gr.Markdown("Busca cualquier perfume del mundo y encuentra los más parecidos que tenemos **en stock**.")
    
    with gr.Row():
        drop_marca = gr.Dropdown(choices=marcas_list, label="1. Elige la Marca")
        drop_perfume = gr.Dropdown(choices=[], label="2. Elige el Perfume", interactive=True)
    
    # Evento para actualizar perfumes cuando cambie la marca (CORREGIDO)
    drop_marca.change(fn=update_perfumes, inputs=drop_marca, outputs=drop_perfume)
    
    btn = gr.Button("🔍 Buscar Parecidos en Stock", variant="primary")
    
    with gr.Row():
        out_table = gr.Dataframe(
            headers=["Marca", "Perfume", "Similitud", "Notas"],
            interactive=False
        )
    
    btn.click(fn=recomendar, inputs=[drop_marca, drop_perfume], outputs=out_table)

# Lanzar aplicación
if __name__ == "__main__":
    demo.launch(share=True)