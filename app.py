import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Diplomado IA: Visión Artificial",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Estilos CSS ---
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    h1 {color: #2c3e50;}
    .stTabs [data-baseweb="tab-list"] {gap: 10px;}
    .stTabs [data-baseweb="tab"] {background-color: #ffffff; border-radius: 5px;}
    .stTabs [aria-selected="true"] {background-color: #e3f2fd;}
    /* Ajuste para centrar imágenes */
    div[data-testid="stImage"] {
        display: flex;
        justify_content: center;
    }
    /* Estilo para las explicaciones */
    .explanation-box {
        background-color: #e8f4f8;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #007bff;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Funciones Auxiliares ---

def load_local_image(filename):
    """Carga una imagen local desde la misma carpeta del script."""
    if not os.path.exists(filename):
        st.sidebar.error(f"❌ Error: No se encuentra '{filename}' en la carpeta.")
        return None
    
    # Cargar con OpenCV
    image = cv2.imread(filename)
    if image is None:
        st.sidebar.error(f"❌ Error: '{filename}' no es una imagen válida.")
        return None
        
    # Convertir BGR a RGB
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def add_salt_and_pepper_noise(image, prob):
    output = np.copy(image)
    num_salt = np.ceil(prob * image.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    output[coords[0], coords[1], :] = 255
    num_pepper = np.ceil(prob * image.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    output[coords[0], coords[1], :] = 0
    return output

def plot_histogram(image, mode='gray'):
    fig, ax = plt.subplots(figsize=(6, 2.5))
    if mode == 'rgb':
        colors = ('r', 'g', 'b')
        for i, color in enumerate(colors):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            ax.plot(hist, color=color)
        ax.set_title("Histograma RGB")
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        ax.plot(hist, color='black')
        ax.fill_between(range(256), hist.flatten(), color='gray', alpha=0.3)
        ax.set_title("Histograma de Intensidad")
    ax.set_xlim([0, 256])
    plt.tight_layout()
    return fig

def show_kernel_latex(name):
    kernels = {
        "Promedio": r"K = \frac{1}{k^2} \begin{bmatrix} 1 & \dots & 1 \\ \vdots & \ddots & \vdots \\ 1 & \dots & 1 \end{bmatrix}",
        "Sobel X": r"G_x = \begin{bmatrix} -1 & 0 & +1 \\ -2 & 0 & +2 \\ -1 & 0 & +1 \end{bmatrix}",
        "Sobel Y": r"G_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ +1 & +2 & +1 \end{bmatrix}",
        "Laplaciano": r"K = \begin{bmatrix} 0 & 1 & 0 \\ 1 & -4 & 1 \\ 0 & 1 & 0 \end{bmatrix}"
    }
    if name in kernels:
        st.latex(kernels[name])

# --- Interfaz Principal ---

st.title("👁️ Módulo 3: Visión Artificial - Preprocesamiento")
st.markdown("Herramienta didáctica para visualizar algoritmos de procesamiento de imágenes.")

# --- BARRA LATERAL (Configuración) ---
st.sidebar.header("1. Configuración de Entrada")
img_source = st.sidebar.radio("Fuente de Imagen:", ["Ejemplo: Lena", "Ejemplo: Monedas", "Subir Imagen"])

original_image = None

if img_source == "Subir Imagen":
    uploaded = st.sidebar.file_uploader("Carga tu imagen aquí:", type=["jpg", "png", "jpeg"])
    if uploaded:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        original_image = cv2.imdecode(file_bytes, 1)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    else:
        st.info("👈 Por favor sube una imagen en el menú lateral.")
        st.stop()

elif img_source == "Ejemplo: Lena":
    # CARGA LOCAL
    original_image = load_local_image("lena.png")

elif img_source == "Ejemplo: Monedas":
    # CARGA LOCAL
    original_image = load_local_image("coins.jpg")

if original_image is None:
    st.stop()

# --- Pestañas de Contenido ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Análisis y Color", 
    "🧹 Filtrado y Ruido", 
    "⚡ Detección de Bordes", 
    "📐 Transformaciones", 
    "🧬 Morfología"
])

# --- TAB 1: ANÁLISIS ---
with tab1:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(original_image, caption="Imagen Original", width=400)
    
    with col2:
        st.subheader("Análisis de la Imagen")
        analysis_mode = st.selectbox("Herramienta:", ["Histograma", "Canales RGB", "Espacio HSV"])
        
        if analysis_mode == "Histograma":
            hist_mode = st.radio("Modo:", ["Escala de Grises", "RGB"])
            fig = plot_histogram(original_image, 'rgb' if hist_mode == "RGB" else 'gray')
            st.pyplot(fig)
            explanation = "**Histograma:** Gráfico que muestra cuántos píxeles hay de cada intensidad. Si los picos están a la izquierda, la imagen es oscura; a la derecha, es brillante. Es fundamental para corregir la exposición."
            
        elif analysis_mode == "Canales RGB":
            r, g, b = cv2.split(original_image)
            zeros = np.zeros_like(r)
            img_red = cv2.merge([r, zeros, zeros])
            img_green = cv2.merge([zeros, g, zeros])
            img_blue = cv2.merge([zeros, zeros, b])
            
            c1, c2, c3 = st.columns(3)
            c1.image(img_red, caption="Rojo", use_container_width=True)
            c2.image(img_green, caption="Verde", use_container_width=True)
            c3.image(img_blue, caption="Azul", use_container_width=True)
            explanation = "**Canales RGB:** Una imagen digital a color está formada por tres matrices superpuestas (Rojo, Verde, Azul). Al combinarlas, el ojo humano percibe todos los colores."
            
        elif analysis_mode == "Espacio HSV":
            hsv = cv2.cvtColor(original_image, cv2.COLOR_RGB2HSV)
            h, s, v = cv2.split(hsv)
            
            c1, c2, c3 = st.columns(3)            
            fig_h, ax_h = plt.subplots(figsize=(3,3))
            # ax_h.imshow(h, cmap='hsv') 
            # ax_h.axis('off')
            # ax_h.set_title("Hue (Matiz)")
            #st.pyplot(fig_h)
            with c1: st.image(h, caption="Matiz", use_container_width=True,channels="RGB")
            with c2: st.image(s, caption="Saturación", use_container_width=True,channels="RGB")
            with c3: st.image(v, caption="Valor (Brillo)", use_container_width=True,channels="RGB")
            explanation = "**Espacio HSV:** Separa la información de color (Hue) de la iluminación (Value). Es mucho más robusto que RGB para detectar objetos por color en condiciones de luz variable."

    st.info(f"💡 **Nota Didáctica:** {explanation}")

# --- TAB 2: FILTRADO ---
with tab2:
    st.subheader("Técnicas de Filtrado")
    
    col_controls, col_display = st.columns([1, 3])
    
    with col_controls:
        add_noise = st.checkbox("Agregar Ruido")
        noise_amount = st.slider("Nivel Ruido", 0.01, 0.2, 0.05) if add_noise else 0
        st.markdown("---")
        filter_type = st.selectbox("Filtro", ["Promedio (Mean)", "Gaussiano", "Mediana", "Bilateral"])
        k_size = st.slider("Kernel (k)", 3, 21, 5, step=2)
    
    if add_noise:
        input_image = add_salt_and_pepper_noise(original_image, noise_amount)
        input_caption = "Entrada (Con Ruido)"
    else:
        input_image = original_image.copy()
        input_caption = "Entrada (Original)"
        
    if filter_type == "Promedio (Mean)":
        output_image = cv2.blur(input_image, (k_size, k_size))
        kernel_name = "Promedio"
        expl = "Calcula el promedio simple de los vecinos. Es rápido pero difumina mucho los bordes."
    elif filter_type == "Gaussiano":
        output_image = cv2.GaussianBlur(input_image, (k_size, k_size), 0)
        kernel_name = None
        expl = "Usa una campana de Gauss (pesos mayores al centro). Elimina ruido gaussiano (grano) mejor que el promedio."
    elif filter_type == "Mediana":
        output_image = cv2.medianBlur(input_image, k_size)
        kernel_name = None
        expl = "Reemplaza el píxel por la mediana estadística. **Es el mejor para eliminar ruido 'Sal y Pimienta'** (puntos blancos y negros) sin borrar los bordes."
    elif filter_type == "Bilateral":
        output_image = cv2.bilateralFilter(input_image, 9, 75, 75)
        kernel_name = None
        expl = "Filtro avanzado que suaviza las texturas planas pero **respeta los bordes fuertes**. Es computacionalmente más costoso."

    c1, c2 = col_display.columns(2)
    with c1: st.image(input_image, caption=input_caption, width=350)
    with c2: st.image(output_image, caption=f"Salida ({filter_type})", width=350)
    
    if kernel_name:
        st.markdown("**Kernel:**")
        show_kernel_latex(kernel_name)
    
    st.info(f"💡 **Nota Didáctica:** {expl}")

# --- TAB 3: BORDES ---
with tab3:
    st.subheader("Detección de Bordes")
    
    col_controls, col_imgs = st.columns([1, 3])
    
    with col_controls:
        edge_method = st.selectbox("Método", ["Canny", "Sobel", "Laplaciano"])
        gray_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
        
        if edge_method == "Canny":
            t1 = st.slider("Umbral 1", 0, 255, 50)
            t2 = st.slider("Umbral 2", 0, 255, 150)
            edges = cv2.Canny(gray_image, t1, t2)
            k_name = None
            expl = "**Canny:** Es el algoritmo 'estado del arte' clásico. No es un simple filtro, sino un proceso de múltiples etapas que limpia ruido, encuentra gradientes y conecta líneas discontinuas."
        elif edge_method == "Sobel":
            axis = st.radio("Eje", ["X", "Y", "XY"])
            k = st.slider("Kernel", 3, 7, 3, step=2)
            if axis == "X":
                edges = cv2.convertScaleAbs(cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=k))
                k_name = "Sobel X"
            elif axis == "Y":
                edges = cv2.convertScaleAbs(cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=k))
                k_name = "Sobel Y"
            else:
                gx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=k)
                gy = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=k)
                edges = cv2.addWeighted(cv2.convertScaleAbs(gx), 0.5, cv2.convertScaleAbs(gy), 0.5, 0)
                k_name = None
            expl = "**Sobel:** Calcula la primera derivada (gradiente). Es útil para detectar cambios de intensidad en direcciones específicas (vertical u horizontal)."
        elif edge_method == "Laplaciano":
            edges = cv2.convertScaleAbs(cv2.Laplacian(gray_image, cv2.CV_64F))
            k_name = "Laplaciano"
            expl = "**Laplaciano:** Calcula la segunda derivada. Detecta bordes en todas direcciones a la vez, pero es extremadamente sensible al ruido."

        if k_name:
            st.markdown("---")
            show_kernel_latex(k_name)

    with col_imgs:
        c1, c2 = st.columns(2)
        c1.image(original_image, caption="Original", width=350)
        c2.image(edges, caption=f"Bordes ({edge_method})", width=350)
    
    st.info(f"💡 **Nota Didáctica:** {expl}")

# --- TAB 4: GEOMETRÍA ---
with tab4:
    st.subheader("Transformaciones")
    
    col_controls, col_imgs = st.columns([1, 3])
    rows, cols = original_image.shape[:2]
    
    with col_controls:
        geo_type = st.radio("Tipo", ["Rotación", "Escalado", "Perspectiva"])
        
        if geo_type == "Rotación":
            angle = st.slider("Ángulo", -180, 180, 0)
            M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
            res = cv2.warpAffine(original_image, M, (cols, rows))
            expl = "**Rotación:** Gira la imagen alrededor de un punto (usualmente el centro). Utiliza una matriz de rotación de 2x3."
        elif geo_type == "Escalado":
            scale = st.slider("Escala", 0.1, 2.0, 1.0)
            res = cv2.resize(original_image, None, fx=scale, fy=scale)
            expl = "**Escalado:** Cambia el tamaño de la imagen. Si se agranda, el algoritmo debe 'inventar' píxeles nuevos (interpolación)."
        elif geo_type == "Perspectiva":
            dx = st.slider("Deformar X", 0, 100, 20)
            pts1 = np.float32([[0,0], [cols,0], [0,rows], [cols,rows]])
            pts2 = np.float32([[dx,0], [cols-dx,0], [0,rows], [cols,rows]])
            M = cv2.getPerspectiveTransform(pts1, pts2)
            res = cv2.warpPerspective(original_image, M, (cols, rows))
            expl = "**Perspectiva:** Simula un cambio en el punto de vista de la cámara. Es fundamental para rectificar documentos escaneados o leer patentes de autos en ángulo."

    with col_imgs:
        c1, c2 = st.columns(2)
        c1.image(original_image, caption="Original", width=350)
        
        # --- CORRECCIÓN DEFINITIVA DEL ESCALADO ---
        if geo_type == "Escalado":
            # NO PASAMOS WIDTH para que Streamlit use el tamaño real de la imagen
            c2.image(res, caption="Transformada")
        else:
            # Fijamos el ancho para mantener el orden en Rotación y Perspectiva
            c2.image(res, caption="Transformada", width=350)
    
    st.info(f"💡 **Nota Didáctica:** {expl}")

# --- TAB 5: MORFOLOGÍA ---
with tab5:
    st.subheader("Morfología (Binaria)")
    
    col_controls, col_imgs = st.columns([1, 3])
    
    with col_controls:
        st.markdown("**1. Binarizar**")
        thresh = st.slider("Umbral", 0, 255, 127)
        st.markdown("**2. Operar**")
        op = st.selectbox("Operación", ["Erosión", "Dilatación", "Apertura", "Cierre"])
        k_size = st.slider("Kernel", 3, 15, 5, step=2)
        
        gray = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
        kernel = np.ones((k_size, k_size), np.uint8)
        
        if op == "Erosión": 
            res = cv2.erode(binary, kernel, iterations=1)
            expl = "**Erosión:** 'Come' los bordes de los objetos blancos. Sirve para separar objetos que están tocándose levemente."
        elif op == "Dilatación": 
            res = cv2.dilate(binary, kernel, iterations=1)
            expl = "**Dilatación:** Expande los objetos blancos. Sirve para rellenar pequeños huecos o conectar partes rotas de un objeto."
        elif op == "Apertura": 
            res = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            expl = "**Apertura:** Es una Erosión seguida de Dilatación. Es perfecta para **eliminar ruido** (puntos blancos pequeños) sin cambiar el tamaño del objeto principal."
        elif op == "Cierre": 
            res = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            expl = "**Cierre:** Es una Dilatación seguida de Erosión. Sirve para **cerrar agujeros negros** dentro de los objetos blancos."

    with col_imgs:
        c1, c2 = st.columns(2)
        c1.image(binary, caption="Binaria (Entrada)", width=350)
        c2.image(res, caption=f"Resultado ({op})", width=350)
    
    st.info(f"💡 **Nota Didáctica:** {expl}")