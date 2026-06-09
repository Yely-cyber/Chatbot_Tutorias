import string
import random
import nltk
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords

# Descargas NLTK
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab')
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('omw-1.4', quiet=True)

# ─────────────────────────────────────────────────────────────
# CONFIGURACIÓN DE LA PÁGINA
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UNSAAC - Universidad Nacional de San Antonio Abad del Cusco",
    page_icon="🏛️",
    layout="wide"
)

# ─────────────────────────────────────────────────────────────
# CSS PERSONALIZADO - Estilo UNSAAC
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

/* Ocultar elementos de Streamlit */
header[data-testid="stHeader"] {
    background: #003366;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

.main .block-container {
    padding: 0;
    max-width: 100%;
}

/* === BARRA SUPERIOR AZUL === */
.top-bar {
    background: #003366;
    color: white;
    padding: 8px 120px;
    font-size: 12px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.top-bar a {
    color: white;
    text-decoration: none;
    margin-left: 20px;
    font-size: 12px;
}

.top-bar-left {
    display: flex;
    gap: 20px;
}

.top-bar-right {
    display: flex;
    gap: 15px;
}

/* === HEADER PRINCIPAL === */
.main-header {
    background: white;
    padding: 20px 120px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
}

.logo-area {
    display: flex;
    align-items: center;
    gap: 15px;
}

.logo-icon {
    font-size: 48px;
}

.logo-text h1 {
    color: #003366;
    font-size: 22px;
    font-weight: 700;
    margin: 0;
}

.logo-text p {
    color: #666;
    font-size: 11px;
    margin: 0;
}

/* === NAVEGACIÓN === */
.nav-bar {
    background: #004d99;
    padding: 0 120px;
}

.nav-bar ul {
    display: flex;
    list-style: none;
    gap: 30px;
    margin: 0;
    padding: 0;
}

.nav-bar li {
    padding: 15px 0;
    color: white;
    font-weight: 500;
    font-size: 14px;
    cursor: pointer;
    position: relative;
}

.nav-bar li:hover {
    background: #003366;
    padding: 15px 0;
}

/* === SLIDER / BANNER === */
.slider {
    background: linear-gradient(135deg, #003366 0%, #004d99 100%);
    padding: 60px 120px;
    color: white;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.slider-text h2 {
    font-size: 36px;
    font-weight: 700;
    margin-bottom: 15px;
}

.slider-text p {
    font-size: 16px;
    opacity: 0.9;
}

.slider-image {
    font-size: 80px;
}

/* === SECCIÓN DE NOTICIAS === */
.news-section {
    padding: 50px 120px;
    background: #f5f7fa;
}

.section-title {
    color: #003366;
    font-size: 28px;
    font-weight: 700;
    margin-bottom: 30px;
    border-left: 5px solid #ffcc00;
    padding-left: 15px;
}

.news-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 30px;
}

.news-card {
    background: white;
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 3px 15px rgba(0,0,0,0.08);
    transition: transform 0.3s;
}

.news-card:hover {
    transform: translateY(-5px);
}

.news-img {
    background: #004d99;
    height: 180px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 50px;
    color: white;
}

.news-content {
    padding: 20px;
}

.news-date {
    color: #ffcc00;
    font-size: 12px;
    font-weight: 600;
    margin-bottom: 8px;
}

.news-title {
    color: #003366;
    font-size: 16px;
    font-weight: 600;
    margin-bottom: 10px;
}

.news-desc {
    color: #666;
    font-size: 13px;
    line-height: 1.5;
}

/* === ENLACES RÁPIDOS === */
.quick-links {
    padding: 50px 120px;
    background: white;
}

.links-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 20px;
}

.link-item {
    text-align: center;
    padding: 25px 15px;
    background: #f5f7fa;
    border-radius: 10px;
    transition: all 0.3s;
    cursor: pointer;
}

.link-item:hover {
    background: #003366;
    color: white;
}

.link-icon {
    font-size: 32px;
    margin-bottom: 10px;
}

.link-text {
    font-size: 13px;
    font-weight: 500;
}

/* === FOOTER === */
.footer {
    background: #002244;
    color: #aaa;
    padding: 40px 120px 20px;
}

.footer-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 40px;
    margin-bottom: 30px;
}

.footer-col h4 {
    color: white;
    font-size: 16px;
    margin-bottom: 15px;
}

.footer-col p, .footer-col a {
    color: #aaa;
    font-size: 12px;
    line-height: 1.8;
    text-decoration: none;
    display: block;
}

.footer-bottom {
    text-align: center;
    padding-top: 20px;
    border-top: 1px solid #004466;
    font-size: 11px;
}

/* === CHATBOT BUTTON FLOTANTE === */
.chatbot-btn-container {
    position: fixed;
    bottom: 30px;
    right: 30px;
    z-index: 1000;
}

.chatbot-btn {
    background: linear-gradient(135deg, #003366 0%, #004d99 100%);
    border: none;
    border-radius: 50px;
    padding: 15px 25px;
    color: white;
    font-family: 'Poppins', sans-serif;
    font-weight: 600;
    font-size: 16px;
    cursor: pointer;
    box-shadow: 0 5px 20px rgba(0,51,102,0.4);
    transition: all 0.3s;
    display: flex;
    align-items: center;
    gap: 10px;
}

.chatbot-btn:hover {
    transform: scale(1.05);
    background: linear-gradient(135deg, #004d99 0%, #0066cc 100%);
}

/* === ESTILOS DEL CHAT === */
.chat-container {
    max-width: 900px;
    margin: 0 auto;
    padding: 20px;
    min-height: 100vh;
}

.header-box {
    background: linear-gradient(135deg, #003366 0%, #004d99 100%);
    border-radius: 16px;
    padding: 20px 24px;
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 24px;
}

.header-title {
    color: white;
    font-size: 22px;
    font-weight: 600;
    margin: 0;
}

.header-sub {
    color: #ffcc00;
    font-size: 13px;
    margin: 4px 0 0;
}

.bubble-bot {
    background: #ffffff;
    border: 1px solid #e0e0e0;
    border-radius: 16px 16px 16px 4px;
    padding: 12px 16px;
    font-size: 14px;
    color: #1a1a1a;
    max-width: 80%;
    margin-bottom: 4px;
    line-height: 1.6;
}

.bubble-user {
    background: #003366;
    border-radius: 16px 16px 4px 16px;
    padding: 12px 16px;
    font-size: 14px;
    color: white;
    max-width: 80%;
    margin-left: auto;
    margin-bottom: 4px;
    line-height: 1.6;
}

.row-bot {
    display: flex;
    justify-content: flex-start;
    margin-bottom: 8px;
}

.row-user {
    display: flex;
    justify-content: flex-end;
    margin-bottom: 8px;
}

.avatar-bot {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    background: #004d99;
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
    flex-shrink: 0;
    margin-right: 8px;
    margin-top: 2px;
}

.back-btn {
    background: #003366;
    color: white;
    border: none;
    border-radius: 25px;
    padding: 10px 20px;
    cursor: pointer;
    font-family: 'Poppins', sans-serif;
    margin-bottom: 20px;
}

.back-btn:hover {
    background: #004d99;
}

/* Sidebar del chat */
[data-testid="stSidebar"] {
    background: #f5f7fa;
}

.sidebar-title {
    font-size: 12px;
    color: #5F5E5A;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 12px;
    font-weight: 600;
}

div.stButton > button {
    background: #E1F5EE;
    color: #003366;
    border: 1px solid #004d99;
    border-radius: 20px;
    font-size: 12px;
    padding: 4px 14px;
    font-family: 'Poppins', sans-serif;
}

div.stButton > button:hover {
    background: #004d99;
    color: white;
    border-color: #003366;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# CARGA DEL CORPUS
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def cargar_corpus():
    try:
        with open("Corpus_tutorias.txt", "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read().lower()
        return nltk.sent_tokenize(raw)
    except FileNotFoundError:
        corpus_basico = """
        La tutoría académica es un proceso de acompañamiento personalizado.
        El tutor ayuda al estudiante en su desarrollo académico y personal.
        Bienestar Universitario ofrece servicios de apoyo psicológico y social.
        Las técnicas de estudio incluyen subrayado, resúmenes y mapas conceptuales.
        La matrícula regular se realiza al inicio de cada semestre académico.
        """
        return nltk.sent_tokenize(corpus_basico)

# ─────────────────────────────────────────────────────────────
# NLP
# ─────────────────────────────────────────────────────────────
lemmer = nltk.stem.WordNetLemmatizer()

remove_punct_dict = dict(
    (ord(p), None) for p in string.punctuation
)

def lem_tokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

def lem_normalize(text):
    return lem_tokens(
        nltk.word_tokenize(
            text.lower().translate(remove_punct_dict)
        )
    )

SALUDOS_IN = (
    "hola",
    "buenas",
    "saludos",
    "qué tal",
    "hey",
    "buenos días"
)

SALUDOS_OUT = [
    "¡Hola! ¿En qué puedo ayudarte?",
    "¡Hola! Soy el asistente de Tutorías UNSAAC.",
    "¡Bienvenido! Estoy aquí para ayudarte con información académica.",
    "¡Buenas! Puedes consultarme sobre tutorías, bienestar universitario y técnicas de estudio."
]

def saludo(sentence):
    for word in sentence.split():
        if word.lower() in SALUDOS_IN:
            return random.choice(SALUDOS_OUT)

def respuesta_corpus(user_response, sent_tokens):
    tokens_temp = sent_tokens.copy()
    tokens_temp.append(user_response)
    
    vectorizer = TfidfVectorizer(
        tokenizer=lem_normalize,
        stop_words=stopwords.words("spanish")
    )
    
    tfidf = vectorizer.fit_transform(tokens_temp)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    
    if req_tfidf == 0:
        return "Lo siento, no encontré información sobre ese tema en el material de Tutorías Académicas UNSAAC. Intenta reformular tu pregunta."
    return tokens_temp[idx]

def obtener_respuesta(user_input, sent_tokens):
    texto = user_input.lower().strip()
    
    if texto in ("salir", "adios", "chau"):
        return "¡Hasta pronto! Éxitos en tus estudios. 🎓"
    if texto in ("gracias", "muchas gracias"):
        return "¡Con mucho gusto! ¿Hay algo más en lo que pueda ayudarte?"
    
    s = saludo(texto)
    if s:
        return s
    
    return respuesta_corpus(texto, sent_tokens)

# ─────────────────────────────────────────────────────────────
# FUNCIÓN PARA MOSTRAR LA PÁGINA PRINCIPAL
# ─────────────────────────────────────────────────────────────
def mostrar_pagina_principal():
    # Barra superior
    st.markdown("""
    <div class="top-bar">
        <div class="top-bar-left">
            <span>📞 (084) 123456</span>
            <span>✉️ informes@unsaac.edu.pe</span>
        </div>
        <div class="top-bar-right">
            <a href="#">Intranet</a>
            <a href="#">Correo Institucional</a>
            <a href="#">Biblioteca Virtual</a>
            <a href="#">Aula Virtual</a>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Header principal
    st.markdown("""
    <div class="main-header">
        <div class="logo-area">
            <div class="logo-icon">🏛️</div>
            <div class="logo-text">
                <h1>UNSAAC</h1>
                <p>Universidad Nacional de San Antonio Abad del Cusco</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Navegación
    st.markdown("""
    <div class="nav-bar">
        <ul>
            <li>🏠 Inicio</li>
            <li>📖 La Universidad</li>
            <li>📚 Facultades</li>
            <li>🎓 Admisión</li>
            <li>📝 Investigación</li>
            <li>🌍 Internacional</li>
            <li>📢 Transparencia</li>
            <li>📞 Contacto</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Slider/Banner
    st.markdown("""
    <div class="slider">
        <div class="slider-text">
            <h2>Excelencia Académica <br>con Tradición e Innovación</h2>
            <p>Formamos profesionales líderes con valores éticos y compromiso social</p>
        </div>
        <div class="slider-image">
            🎓🏛️
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sección de noticias
    st.markdown("""
    <div class="news-section">
        <div class="section-title">📰 Últimas Noticias</div>
        <div class="news-grid">
            <div class="news-card">
                <div class="news-img">📢</div>
                <div class="news-content">
                    <div class="news-date">15 DE ENERO, 2025</div>
                    <div class="news-title">Convocatoria a exámenes de admisión 2025-I</div>
                    <div class="news-desc">La Oficina Central de Admisión informa sobre el cronograma de exámenes para el próximo ciclo académico.</div>
                </div>
            </div>
            <div class="news-card">
                <div class="news-img">🎓</div>
                <div class="news-content">
                    <div class="news-date">10 DE ENERO, 2025</div>
                    <div class="news-title">Graduación de nuevos profesionales</div>
                    <div class="news-desc">Más de 500 estudiantes recibieron su título profesional en ceremonia realizada en el Paraninfo Universitario.</div>
                </div>
            </div>
            <div class="news-card">
                <div class="news-img">🔬</div>
                <div class="news-content">
                    <div class="news-date">05 DE ENERO, 2025</div>
                    <div class="news-title">Convenio internacional con Universidad de Salamanca</div>
                    <div class="news-desc">Estudiantes UNSAAC podrán realizar intercambios académicos en España.</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Enlaces rápidos
    st.markdown("""
    <div class="quick-links">
        <div class="section-title">🔗 Enlaces de Interés</div>
        <div class="links-grid">
            <div class="link-item">
                <div class="link-icon">📝</div>
                <div class="link-text">Trámites<br>Documentarios</div>
            </div>
            <div class="link-item">
                <div class="link-icon">📚</div>
                <div class="link-text">Biblioteca<br>Central</div>
            </div>
            <div class="link-item">
                <div class="link-icon">💻</div>
                <div class="link-text">Campus<br>Virtual</div>
            </div>
            <div class="link-item">
                <div class="link-icon">🎓</div>
                <div class="link-text">Becas y<br>Créditos</div>
            </div>
            <div class="link-item">
                <div class="link-icon">📊</div>
                <div class="link-text">Investigación<br>+</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <div class="footer-grid">
            <div class="footer-col">
                <h4>UNSAAC</h4>
                <p>Universidad Nacional de San Antonio Abad del Cusco</p>
                <p>Av. de la Cultura Nro. 733</p>
                <p>Cusco - Perú</p>
            </div>
            <div class="footer-col">
                <h4>Enlaces útiles</h4>
                <a href="#">Portal de Transparencia</a>
                <a href="#">Defensoría Universitaria</a>
                <a href="#">Bolsa de Trabajo</a>
                <a href="#">Calendario Académico</a>
            </div>
            <div class="footer-col">
                <h4>Servicios</h4>
                <a href="#">Correo Institucional</a>
                <a href="#">Aula Virtual</a>
                <a href="#">Biblioteca Virtual</a>
                <a href="#">Solicitudes en Línea</a>
            </div>
            <div class="footer-col">
                <h4>Síguenos</h4>
                <a href="#">📘 Facebook</a>
                <a href="#">🐦 Twitter</a>
                <a href="#">📸 Instagram</a>
                <a href="#">▶️ YouTube</a>
            </div>
        </div>
        <div class="footer-bottom">
            © 2025 Universidad Nacional de San Antonio Abad del Cusco - Todos los derechos reservados
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# FUNCIÓN PARA MOSTRAR EL CHAT
# ─────────────────────────────────────────────────────────────
def mostrar_chat():
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Botón para volver a la página principal
    col1, col2, col3 = st.columns([1, 4, 1])
    with col1:
        if st.button("← Volver al inicio", key="back_btn", use_container_width=True):
            st.session_state.pagina_actual = "principal"
            st.rerun()
    
    # Header del chat
    st.markdown("""
    <div class="header-box">
        <span style="font-size:36px;">🎓</span>
        <div>
            <p class="header-title">Asistente de Tutorías UNSAAC</p>
            <p class="header-sub">Resuelve tus dudas académicas y universitarias</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar del chat
    with st.sidebar:
        st.markdown("### 🎓 Asistente de Tutorías UNSAAC")
        st.markdown("---")
        st.markdown('<div class="sidebar-title">Consultas frecuentes</div>', unsafe_allow_html=True)
        
        temas = {
            "🎓 Tutoría": "¿Qué es la tutoría académica?",
            "👨‍🏫 Tutor": "¿Cuáles son las funciones del tutor académico?",
            "📚 Técnicas de estudio": "¿Qué técnicas de estudio existen?",
            "🧠 Bienestar": "¿Qué servicios ofrece Bienestar Universitario?",
            "📈 Rendimiento": "¿Qué hacer si tengo bajo rendimiento académico?",
            "📝 Matrícula": "¿Cómo se realiza la matrícula?",
            "💼 Perfil profesional": "¿Cómo elaborar un currículum vitae?"
        }
        
        for label, pregunta in temas.items():
            if st.button(label, key=f"sidebar_{label}", use_container_width=True):
                st.session_state.pregunta_rapida = pregunta
        
        st.markdown("---")
        if st.button("🗑️ Limpiar conversación", use_container_width=True):
            st.session_state.mensajes = []
            st.session_state.pregunta_rapida = ""
            st.rerun()
        
        st.markdown("---")
        st.caption("Powered by TF-IDF · NLTK")
    
    # Estado inicial del chat
    if "mensajes" not in st.session_state:
        st.session_state.mensajes = [{
            "rol": "bot",
            "texto": "¡Hola! Soy el asistente virtual de Tutorías Académicas de la UNSAAC. Puedes preguntarme sobre tutoría académica, bienestar universitario, técnicas de estudio, matrícula, rendimiento académico y más. ¿En qué puedo ayudarte?"
        }]
    
    if "pregunta_rapida" not in st.session_state:
        st.session_state.pregunta_rapida = ""
    
    # Cargar corpus
    sent_tokens = cargar_corpus()
    
    # Mostrar historial
    chat_html = ""
    for msg in st.session_state.mensajes:
        if msg["rol"] == "bot":
            chat_html += f"""
            <div class="row-bot">
                <div class="avatar-bot">🤖</div>
                <div class="bubble-bot">{msg["texto"]}</div>
            </div>
            """
        else:
            chat_html += f"""
            <div class="row-user">
                <div class="bubble-user">{msg["texto"]}</div>
            </div>
            """
    st.markdown(chat_html, unsafe_allow_html=True)
    
    # Preguntas sugeridas
    if len(st.session_state.mensajes) <= 1:
        st.markdown("**💡 Preguntas sugeridas:**")
        sugerencias = [
            "¿Qué es la tutoría académica?",
            "¿Qué es la matrícula condicionada?",
            "¿Qué técnicas de estudio existen?",
            "¿Cuándo debo acudir a Bienestar Universitario?"
        ]
        cols = st.columns(len(sugerencias))
        for col, sug in zip(cols, sugerencias):
            with col:
                if st.button(sug, key=f"sug_{sug}"):
                    st.session_state.pregunta_rapida = sug
    
    # Input del chat
    user_input = st.chat_input("Escribe tu pregunta aquí...")
    
    if st.session_state.pregunta_rapida:
        user_input = st.session_state.pregunta_rapida
        st.session_state.pregunta_rapida = ""
    
    if user_input:
        st.session_state.mensajes.append({"rol": "user", "texto": user_input})
        respuesta = obtener_respuesta(user_input, sent_tokens)
        st.session_state.mensajes.append({"rol": "bot", "texto": respuesta})
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# CONTROL DE NAVEGACIÓN PRINCIPAL
# ─────────────────────────────────────────────────────────────
# Inicializar estado de página
if "pagina_actual" not in st.session_state:
    st.session_state.pagina_actual = "principal"

# Mostrar la página correspondiente
if st.session_state.pagina_actual == "chat":
    mostrar_chat()
else:
    mostrar_pagina_principal()

# Botón flotante de chatbot (solo visible en la página principal)
if st.session_state.pagina_actual == "principal":
    st.markdown("""
    <div class="chatbot-btn-container">
        <div class="chatbot-btn" onclick="parent.document.querySelector('button[data-testid=\"baseButton-secondary\"]').click()">
            💬 Asistente Virtual
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Botón oculto para capturar el clic
    if st.button("", key="abrir_chat", help="Abrir chatbot", use_container_width=False):
        st.session_state.pagina_actual = "chat"
        st.rerun()
    
    # CSS para ocultar el botón real y mostrar el estilo flotante
    st.markdown("""
    <style>
    button[data-testid="baseButton-secondary"][kind="secondary"] {
        position: fixed;
        bottom: 30px;
        right: 30px;
        z-index: 1000;
        background: linear-gradient(135deg, #003366 0%, #004d99 100%);
        border: none;
        border-radius: 50px;
        padding: 15px 25px;
        color: white;
        font-weight: 600;
        font-size: 16px;
        box-shadow: 0 5px 20px rgba(0,51,102,0.4);
        transition: all 0.3s;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    button[data-testid="baseButton-secondary"][kind="secondary"]:hover {
        transform: scale(1.05);
        background: linear-gradient(135deg, #004d99 0%, #0066cc 100%);
    }
    button[data-testid="baseButton-secondary"][kind="secondary"]::before {
        content: "💬";
        margin-right: 8px;
    }
    </style>
    """, unsafe_allow_html=True)
