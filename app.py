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
    page_title="Escuela Profesional de Ingeniería Informática y Sistemas - UNSAAC",
    page_icon="💻",
    layout="wide"
)

# ─────────────────────────────────────────────────────────────
# CSS PERSONALIZADO - Estilo Escuela de Ingeniería Informática
# Paleta de colores: Rojo institucional (#8B0000, #A52A2A) + Dorado (#DAA520, #FFD700)
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;500;600;700;800&family=Open+Sans:wght@300;400;500;600;700&display=swap');

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

html, body, [class*="css"] {
    font-family: 'Open Sans', sans-serif;
}

/* Ocultar elementos de Streamlit */
header[data-testid="stHeader"] {
    background: #8B0000;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

.main .block-container {
    padding: 0;
    max-width: 100%;
}

/* === BARRA SUPERIOR === */
.top-bar {
    background: #6B0000;
    color: #f5f5f5;
    padding: 8px 120px;
    font-size: 12px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
}

.top-bar a {
    color: #FFD700;
    text-decoration: none;
    margin-left: 20px;
    font-size: 12px;
    transition: color 0.3s;
}

.top-bar a:hover {
    color: #ffffff;
    text-decoration: underline;
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
    background: linear-gradient(135deg, #8B0000 0%, #A52A2A 50%, #8B0000 100%);
    padding: 20px 120px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
}

.logo-area {
    display: flex;
    align-items: center;
    gap: 20px;
}

.logo-icon {
    font-size: 52px;
    filter: drop-shadow(2px 2px 4px rgba(0,0,0,0.2));
}

.logo-text h1 {
    color: #FFD700;
    font-size: 24px;
    font-weight: 700;
    margin: 0;
    font-family: 'Montserrat', sans-serif;
    letter-spacing: -0.5px;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
}

.logo-text h1 span {
    color: white;
}

.logo-text p {
    color: rgba(255,255,255,0.85);
    font-size: 11px;
    margin: 5px 0 0;
}

/* Botón del Chatbot en el Header - ESTÁTICO EN ESQUINA SUPERIOR DERECHA */
.chatbot-header-wrapper {
    display: flex;
    justify-content: flex-end;
}

.chatbot-header-btn {
    background: linear-gradient(135deg, #DAA520 0%, #FFD700 100%);
    border: none;
    border-radius: 40px;
    padding: 12px 28px;
    color: #8B0000;
    font-family: 'Montserrat', sans-serif;
    font-weight: 700;
    font-size: 14px;
    cursor: pointer;
    transition: all 0.3s;
    display: flex;
    align-items: center;
    gap: 10px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
}

.chatbot-header-btn:hover {
    transform: translateY(-2px);
    background: linear-gradient(135deg, #FFD700 0%, #FFC107 100%);
    box-shadow: 0 6px 20px rgba(0,0,0,0.25);
    color: #6B0000;
}

/* === NAVEGACIÓN === */
.nav-bar {
    background: #7A0000;
    padding: 0 120px;
    overflow-x: auto;
    border-bottom: 3px solid #FFD700;
}

.nav-bar ul {
    display: flex;
    list-style: none;
    gap: 35px;
    margin: 0;
    padding: 0;
}

.nav-bar li {
    padding: 15px 0;
    color: white;
    font-weight: 500;
    font-size: 13px;
    cursor: pointer;
    white-space: nowrap;
    transition: all 0.3s;
    letter-spacing: 0.5px;
}

.nav-bar li:hover {
    color: #FFD700;
    border-bottom: 2px solid #FFD700;
    margin-bottom: -2px;
}

/* === SLIDER / BANNER === */
.slider {
    background: linear-gradient(135deg, #FFF8E7 0%, #FFF3D6 100%);
    padding: 50px 120px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
}

.slider-text h2 {
    color: #8B0000;
    font-size: 32px;
    font-weight: 700;
    margin-bottom: 15px;
    font-family: 'Montserrat', sans-serif;
}

.slider-text p {
    color: #555;
    font-size: 16px;
    opacity: 0.9;
}

.slider-image {
    font-size: 90px;
}

/* === SECCIÓN DE INFORMACIÓN === */
.info-section {
    padding: 50px 120px;
    background: #FAFAFA;
}

.section-title {
    color: #8B0000;
    font-size: 28px;
    font-weight: 700;
    margin-bottom: 30px;
    border-left: 5px solid #DAA520;
    padding-left: 20px;
    font-family: 'Montserrat', sans-serif;
}

.info-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 30px;
}

.info-card {
    background: white;
    border-radius: 15px;
    overflow: hidden;
    box-shadow: 0 5px 20px rgba(0,0,0,0.08);
    transition: transform 0.3s, box-shadow 0.3s;
    text-align: center;
    padding: 30px 20px;
    border-top: 4px solid #DAA520;
}

.info-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 15px 35px rgba(0,0,0,0.12);
}

.info-icon {
    font-size: 48px;
    margin-bottom: 15px;
}

.info-title {
    color: #8B0000;
    font-size: 18px;
    font-weight: 700;
    margin-bottom: 10px;
}

.info-desc {
    color: #666;
    font-size: 13px;
    line-height: 1.6;
}

/* === SECCIÓN DE NOTICIAS === */
.news-section {
    padding: 50px 120px;
    background: white;
}

.news-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
    gap: 30px;
}

.news-card {
    background: #FAFAFA;
    border-radius: 15px;
    overflow: hidden;
    box-shadow: 0 3px 15px rgba(0,0,0,0.05);
    transition: transform 0.3s;
}

.news-card:hover {
    transform: translateY(-5px);
}

.news-img {
    background: linear-gradient(135deg, #8B0000 0%, #A52A2A 100%);
    height: 160px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 55px;
    color: #FFD700;
}

.news-content {
    padding: 20px;
}

.news-date {
    color: #DAA520;
    font-size: 12px;
    font-weight: 600;
    margin-bottom: 8px;
}

.news-title {
    color: #8B0000;
    font-size: 16px;
    font-weight: 700;
    margin-bottom: 10px;
}

.news-desc {
    color: #666;
    font-size: 13px;
    line-height: 1.5;
}

/* === FOOTER === */
.footer {
    background: #4A0000;
    color: #ccc;
    padding: 50px 120px 25px;
}

.footer-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 40px;
    margin-bottom: 30px;
}

.footer-col h4 {
    color: #FFD700;
    font-size: 16px;
    margin-bottom: 20px;
    font-weight: 600;
}

.footer-col p, .footer-col a {
    color: #ccc;
    font-size: 12px;
    line-height: 1.8;
    text-decoration: none;
    display: block;
    transition: color 0.3s;
}

.footer-col a:hover {
    color: #FFD700;
}

.footer-bottom {
    text-align: center;
    padding-top: 20px;
    border-top: 1px solid #7A0000;
    font-size: 11px;
}

/* === ESTILOS DEL CHAT MODERNO === */
.chat-container {
    max-width: 1000px;
    margin: 0 auto;
    padding: 20px;
    min-height: 100vh;
    background: #FAFAFA;
}

.header-box {
    background: linear-gradient(135deg, #8B0000 0%, #A52A2A 100%);
    border-radius: 20px;
    padding: 20px 28px;
    display: flex;
    align-items: center;
    gap: 18px;
    margin-bottom: 25px;
    box-shadow: 0 8px 25px rgba(139,0,0,0.2);
}

.header-title {
    color: #FFD700;
    font-size: 22px;
    font-weight: 700;
    margin: 0;
    font-family: 'Montserrat', sans-serif;
}

.header-sub {
    color: rgba(255,255,255,0.85);
    font-size: 13px;
    margin: 6px 0 0;
}

.bubble-bot {
    background: white;
    border-radius: 18px 18px 18px 6px;
    padding: 14px 18px;
    font-size: 14px;
    color: #1a1a2e;
    max-width: 80%;
    margin-bottom: 6px;
    line-height: 1.6;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    border-left: 3px solid #DAA520;
}

.bubble-user {
    background: linear-gradient(135deg, #8B0000 0%, #A52A2A 100%);
    border-radius: 18px 18px 6px 18px;
    padding: 14px 18px;
    font-size: 14px;
    color: white;
    max-width: 80%;
    margin-left: auto;
    margin-bottom: 6px;
    line-height: 1.6;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

.row-bot {
    display: flex;
    justify-content: flex-start;
    margin-bottom: 12px;
}

.row-user {
    display: flex;
    justify-content: flex-end;
    margin-bottom: 12px;
}

.avatar-bot {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    background: linear-gradient(135deg, #8B0000 0%, #A52A2A 100%);
    color: #FFD700;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    flex-shrink: 0;
    margin-right: 10px;
    margin-top: 2px;
}

.back-btn-container {
    margin-bottom: 20px;
}

.back-btn-container button {
    background: linear-gradient(135deg, #8B0000 0%, #A52A2A 100%);
    color: #FFD700;
    border: none;
    border-radius: 30px;
    padding: 10px 25px;
    cursor: pointer;
    font-family: 'Montserrat', sans-serif;
    font-weight: 500;
    transition: all 0.3s;
}

.back-btn-container button:hover {
    background: linear-gradient(135deg, #A52A2A 0%, #8B0000 100%);
    transform: translateX(-3px);
}

/* Estilo del input del chat */
.stChatInputContainer > div {
    border-radius: 30px !important;
    border: 2px solid #E8E8E8 !important;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05) !important;
}

.stChatInputContainer > div:focus-within {
    border-color: #DAA520 !important;
    box-shadow: 0 2px 15px rgba(218,165,32,0.1) !important;
}

/* Sidebar del chat */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #FAFAFA 0%, #F0F0F0 100%);
}

[data-testid="stSidebar"] button {
    border-radius: 12px !important;
    transition: all 0.3s !important;
}

[data-testid="stSidebar"] button:hover {
    background: #8B0000 !important;
    color: #FFD700 !important;
    transform: translateX(5px);
}

/* Botones de sugerencias */
.stButton button {
    background: white !important;
    border: 1px solid #E0E0E0 !important;
    border-radius: 25px !important;
    padding: 8px 16px !important;
    font-size: 12px !important;
    transition: all 0.3s !important;
    color: #333 !important;
}

.stButton button:hover {
    background: #DAA520 !important;
    border-color: #DAA520 !important;
    color: #8B0000 !important;
}

/* Responsive */
@media (max-width: 768px) {
    .top-bar, .main-header, .nav-bar, .slider, .info-section, .news-section, .footer {
        padding-left: 20px;
        padding-right: 20px;
    }
    
    .slider-text h2 {
        font-size: 22px;
    }
    
    .slider-image {
        font-size: 45px;
    }
    
    .chatbot-header-btn {
        padding: 8px 18px;
        font-size: 12px;
    }
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# CARGA DEL CORPUS - Usando el archivo Corpus_tutorias.txt
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def cargar_corpus():
    try:
        with open("Corpus_tutorias.txt", "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read().lower()
        return nltk.sent_tokenize(raw)
    except FileNotFoundError:
        # Corpus de respaldo si no encuentra el archivo
        corpus_basico = """
        La tutoría académica en la Escuela Profesional de Ingeniería Informática y de Sistemas de la UNSAAC es un proceso de acompañamiento permanente.
        Existen tres tipos de tutoría: académica, personal y profesional.
        La tutoría académica se enfoca en el rendimiento en cursos como Algoritmos, Estructura de Datos e Ingeniería de Software.
        La tutoría personal aborda aspectos como autoestima, manejo del estrés y comunicación.
        La tutoría profesional orienta sobre el futuro laboral y elaboración de CV.
        El tutor es un docente de la Escuela que apoya la formación integral del estudiante.
        La matrícula condicionada requiere obligatoriamente un tutor académico.
        Bienestar Universitario ofrece apoyo psicológico, social y económico.
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
    "hola", "buenas", "saludos", "qué tal", "hey", "buenos días", 
    "buenas tardes", "buenas noches", "holi", "holita"
)

SALUDOS_OUT = [
    "¡Hola! 👋 Soy el asistente de tutorías de la Escuela de Ingeniería Informática y Sistemas. ¿En qué puedo ayudarte?",
    "¡Hola! Bienvenido al sistema de tutorías. Estoy aquí para resolver tus dudas sobre acompañamiento académico, personal y profesional.",
    "¡Buenas! Soy el asistente virtual de tutorías. Puedes preguntarme sobre el proceso tutorial, funciones del tutor, técnicas de estudio, bienestar universitario y más.",
    "¡Hola! ¿Necesitas orientación sobre tutorías académicas? Cuéntame cómo puedo ayudarte."
]

def saludo(sentence):
    for word in sentence.split():
        if word.lower() in SALUDOS_IN:
            return random.choice(SALUDOS_OUT)

def respuesta_corpus(user_response, sent_tokens):
    # Agregar la respuesta del usuario al corpus temporal
    tokens_temp = sent_tokens.copy()
    tokens_temp.append(user_response)
    
    # Crear vectorizador TF-IDF
    vectorizer = TfidfVectorizer(
        tokenizer=lem_normalize,
        stop_words=stopwords.words("spanish")
    )
    
    # Calcular similitud coseno
    tfidf = vectorizer.fit_transform(tokens_temp)
    vals = cosine_similarity(tfidf[-1], tfidf)
    
    # Encontrar la mejor coincidencia
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    
    # Obtener el índice de la respuesta más similar
    idx = vals.argsort()[0][-2]
    
    if req_tfidf == 0:
        return "Lo siento, no encontré información específica sobre ese tema en el material de tutorías. Por favor, reformula tu pregunta o consulta directamente con el Comité Tutorial de nuestra Escuela."
    
    return tokens_temp[idx]

def obtener_respuesta(user_input, sent_tokens):
    texto = user_input.lower().strip()
    
    # Palabras de despedida
    if texto in ("salir", "adios", "chau", "hasta luego", "nos vemos"):
        return "¡Hasta pronto! 🎓 Recuerda que tu tutor está disponible para apoyarte durante toda tu carrera. ¡Muchos éxitos!"
    
    # Palabras de agradecimiento
    if texto in ("gracias", "muchas gracias", "te lo agradezco", "gracias por tu ayuda"):
        return "¡Con mucho gusto! 😊 ¿Hay algo más en lo que pueda ayudarte sobre el proceso de tutorías?"
    
    # Verificar saludo
    s = saludo(texto)
    if s:
        return s
    
    # Buscar respuesta en el corpus
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
            <span>✉️ informatica@unsaac.edu.pe</span>
        </div>
        <div class="top-bar-right">
            <a href="#">Intranet</a>
            <a href="#">Correo Institucional</a>
            <a href="#">Laboratorios Virtuales</a>
            <a href="#">Repositorio</a>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Header principal con el botón del chatbot estático en la esquina superior derecha
    st.markdown("""
    <div class="main-header">
        <div class="logo-area">
            <div class="logo-icon">💻</div>
            <div class="logo-text">
                <h1>Escuela Profesional de <span>Ingeniería Informática y de Sistemas</span></h1>
                <p>UNSAAC - Universidad Nacional de San Antonio Abad del Cusco</p>
            </div>
        </div>
        <div class="chatbot-header-wrapper">
            <button class="chatbot-header-btn" id="chatbotBtn">
                💬 Asistente Virtual
            </button>
        </div>
    </div>
    
    <script>
    document.getElementById('chatbotBtn').addEventListener('click', function() {
        const streamlitButton = window.parent.document.querySelector('button[key="header_chat_btn"]');
        if (streamlitButton) {
            streamlitButton.click();
        }
    });
    </script>
    """, unsafe_allow_html=True)
    
    # Botón oculto de Streamlit para el chatbot
    if st.button("", key="header_chat_btn", help="Abrir asistente virtual"):
        st.session_state.pagina = "chat"
        st.rerun()
    
    # Navegación
    st.markdown("""
    <div class="nav-bar">
        <ul>
            <li>🏠 Inicio</li>
            <li>📖 Presentación</li>
            <li>📚 Plan de Estudios</li>
            <li>👨‍🏫 Docentes</li>
            <li>🔬 Investigación</li>
            <li>💼 Bolsa Laboral</li>
            <li>📢 Eventos</li>
            <li>📞 Contacto</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Slider/Banner
    st.markdown("""
    <div class="slider">
        <div class="slider-text">
            <h2>Sistema de Tutoría Académica<br>Acompañamiento para tu Éxito</h2>
            <p>Orientación personalizada para potenciar tu rendimiento y formación integral</p>
        </div>
        <div class="slider-image">
            🎓📚🤝
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sección de información sobre tutorías
    st.markdown("""
    <div class="info-section">
        <div class="section-title">🎓 ¿Qué es la Tutoría Académica?</div>
        <div class="info-grid">
            <div class="info-card">
                <div class="info-icon">📚</div>
                <div class="info-title">Tutoría Académica</div>
                <div class="info-desc">Enfoque en tu rendimiento en cursos como Algoritmos, Estructura de Datos e Ingeniería de Software. Estrategias de estudio y organización del tiempo para programar.</div>
            </div>
            <div class="info-card">
                <div class="info-icon">🧠</div>
                <div class="info-title">Tutoría Personal</div>
                <div class="info-desc">Abordamos aspectos como autoestima, manejo del estrés por carga de trabajos y comunicación con compañeros. Derivación a Bienestar Universitario si es necesario.</div>
            </div>
            <div class="info-card">
                <div class="info-icon">💼</div>
                <div class="info-title">Tutoría Profesional</div>
                <div class="info-desc">Orientación sobre tu futuro como ingeniero informático: elaboración de CV, preparación para entrevistas y vinculación con el mercado laboral.</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sección de noticias/tips
    st.markdown("""
    <div class="news-section">
        <div class="section-title">📝 Tips para tu Éxito Académico</div>
        <div class="news-grid">
            <div class="news-card">
                <div class="news-img">⏰</div>
                <div class="news-content">
                    <div class="news-date">TÉCNICA RECOMENDADA</div>
                    <div class="news-title">Técnica Pomodoro</div>
                    <div class="news-desc">Estudia 25 minutos, descansa 5. Ideal para depurar código o estudiar algoritmos. Evita la fatiga mental.</div>
                </div>
            </div>
            <div class="news-card">
                <div class="news-img">🗂️</div>
                <div class="news-content">
                    <div class="news-date">TÉCNICA RECOMENDADA</div>
                    <div class="news-title">Fichas de Estudio</div>
                    <div class="news-desc">Crea flashcards con preguntas sobre programación. Usa aplicaciones como Anki o Flashcards World.</div>
                </div>
            </div>
            <div class="news-card">
                <div class="news-img">👥</div>
                <div class="news-content">
                    <div class="news-date">RECOMENDACIÓN</div>
                    <div class="news-title">Grupos de Estudio</div>
                    <div class="news-desc">Forma equipos para proyectos grandes. Usa Trello o Notion para organizar tareas y Google Calendar para reuniones.</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <div class="footer-grid">
            <div class="footer-col">
                <h4>📚 Sistema de Tutorías</h4>
                <p>Escuela de Ingeniería Informática y Sistemas</p>
                <p>Comité Tutorial - Director de Escuela</p>
                <p>📞 (084) 123456 anexo 1234</p>
                <p>✉️ tutorias.informatica@unsaac.edu.pe</p>
            </div>
            <div class="footer-col">
                <h4>Enlaces útiles</h4>
                <a href="#">Reglamento de Tutoría</a>
                <a href="#">Horarios de Tutoría</a>
                <a href="#">Bienestar Universitario</a>
                <a href="#">Calendario Académico</a>
            </div>
            <div class="footer-col">
                <h4>Recursos</h4>
                <a href="#">Técnicas de Estudio</a>
                <a href="#">Gestión del Tiempo</a>
                <a href="#">Manejo del Estrés</a>
                <a href="#">Preparación para Entrevistas</a>
            </div>
            <div class="footer-col">
                <h4>Síguenos</h4>
                <a href="#">📘 Facebook - EP Informática</a>
                <a href="#">🐦 Twitter - @informaticaUNSAAC</a>
                <a href="#">📸 Instagram - ep.informatica.unsaac</a>
                <a href="#">💻 GitHub - Labs</a>
            </div>
        </div>
        <div class="footer-bottom">
            © 2025 Escuela Profesional de Ingeniería Informática y de Sistemas - UNSAAC | Sistema de Tutorías Académicas
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# FUNCIÓN PARA MOSTRAR EL CHAT MODERNO
# ─────────────────────────────────────────────────────────────
def mostrar_chat():
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Botón para volver a la página principal
    col1, col2 = st.columns([1, 10])
    with col1:
        if st.button("← Volver", key="back_btn"):
            st.session_state.pagina = "principal"
            st.rerun()
    
    # Header del chat
    st.markdown("""
    <div class="header-box">
        <span style="font-size:40px;">📚🤖</span>
        <div>
            <p class="header-title">Asistente de Tutorías - Ingeniería Informática</p>
            <p class="header-sub">Resuelve tus dudas sobre acompañamiento académico, personal y profesional</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar del chat
    with st.sidebar:
        st.markdown("### 📚 Asistente de Tutorías")
        st.markdown("**Escuela de Ingeniería Informática y Sistemas**")
        st.markdown("---")
        
        temas = {
            "🎓 ¿Qué es la tutoría?": "¿Qué es la tutoría académica?",
            "📚 Tipos de tutoría": "¿Cuáles son los tipos de tutoría?",
            "👨‍🏫 Funciones del tutor": "¿Cuáles son las funciones del tutor académico?",
            "📝 Matrícula condicionada": "¿Qué es matrícula condicionada?",
            "🧠 Bienestar Universitario": "¿Qué servicios ofrece Bienestar Universitario?",
            "⏰ Técnicas de estudio": "¿Qué técnicas de estudio existen para programación?",
            "💼 Perfil profesional": "¿Cómo preparar mi CV como ingeniero informático?"
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
        st.caption("💡 Powered by TF-IDF · NLTK")
        st.caption("📖 Basado en el Reglamento de Tutoría UNSAAC")
        st.caption("© EP Ingeniería Informática - UNSAAC")
    
    # Estado inicial del chat con mensaje de bienvenida
    if "mensajes" not in st.session_state:
        st.session_state.mensajes = [{
            "rol": "bot",
            "texto": "¡Hola! 👋 Bienvenido al asistente virtual de tutorías de la Escuela Profesional de Ingeniería Informática y de Sistemas de la UNSAAC.\n\nEstoy aquí para ayudarte con información sobre:\n• Proceso de tutoría académica, personal y profesional\n• Funciones del tutor y Comité Tutorial\n• Técnicas de estudio para programación\n• Bienestar Universitario y apoyo psicológico\n• Matrícula condicionada y rendimiento académico\n• Elaboración de CV y preparación para entrevistas\n\n¿En qué puedo ayudarte hoy? 🎓"
        }]
    
    if "pregunta_rapida" not in st.session_state:
        st.session_state.pregunta_rapida = ""
    
    # Cargar corpus
    sent_tokens = cargar_corpus()
    
    # Mostrar historial
    for msg in st.session_state.mensajes:
        if msg["rol"] == "bot":
            col1, col2 = st.columns([1, 10])
            with col1:
                st.markdown('<div class="avatar-bot">📚</div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div class="bubble-bot">{msg["texto"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="row-user"><div class="bubble-user">{msg["texto"]}</div></div>', unsafe_allow_html=True)
    
    # Preguntas sugeridas
    if len(st.session_state.mensajes) <= 1:
        st.markdown("**💡 Preguntas frecuentes sobre tutorías:**")
        sugerencias = [
            "¿Qué es la tutoría académica?",
            "¿Cuáles son los tipos de tutoría?",
            "¿Qué hacer si tengo bajo rendimiento en programación?",
            "¿Cómo solicitar cambio de tutor?"
        ]
        cols = st.columns(len(sugerencias))
        for col, sug in zip(cols, sugerencias):
            with col:
                if st.button(sug, key=f"sug_{sug}"):
                    st.session_state.pregunta_rapida = sug
    
    # Input del chat
    user_input = st.chat_input("Escribe tu pregunta sobre tutorías aquí...")
    
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
# CONTROL PRINCIPAL
# ─────────────────────────────────────────────────────────────
if "pagina" not in st.session_state:
    st
