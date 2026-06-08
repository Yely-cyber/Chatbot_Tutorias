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
    page_title="Asistente de Tutorías UNSAAC",
    page_icon="🎓",
    layout="centered"
)

# ─────────────────────────────────────────────────────────────
# CSS PERSONALIZADO
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
}

.header-box {
    background: linear-gradient(135deg, #0a5c46 0%, #1D9E75 100%);
    border-radius: 16px;
    padding: 20px 24px;
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 24px;
}

.header-title {
    color: #E1F5EE;
    font-size: 22px;
    font-weight: 600;
    margin: 0;
}

.header-sub {
    color: #9FE1CB;
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
    background: #1D9E75;
    border-radius: 16px 16px 4px 16px;
    padding: 12px 16px;
    font-size: 14px;
    color: #E1F5EE;
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
    background: #0F6E56;
    color: #E1F5EE;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
    flex-shrink: 0;
    margin-right: 8px;
    margin-top: 2px;
}

div.stButton > button {
    background: #E1F5EE;
    color: #085041;
    border: 1px solid #5DCAA5;
    border-radius: 20px;
    font-size: 12px;
    padding: 4px 14px;
    font-family: 'Sora', sans-serif;
}

div.stButton > button:hover {
    background: #9FE1CB;
    color: #04342C;
    border-color: #1D9E75;
}

[data-testid="stSidebar"] {
    background: #f7fdf9;
}

.sidebar-title {
    font-size: 12px;
    color: #5F5E5A;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 12px;
    font-weight: 600;
}

[data-testid="stChatInput"] textarea {
    font-family: 'Sora', sans-serif;
    border-radius: 24px;
}

footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# CARGA DEL CORPUS
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def cargar_corpus():
    with open("Corpus_tutorias.txt", "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read().lower()

    return nltk.sent_tokenize(raw)

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

    vals = cosine_similarity(
        tfidf[-1],
        tfidf
    )

    idx = vals.argsort()[0][-2]

    flat = vals.flatten()
    flat.sort()

    req_tfidf = flat[-2]

    if req_tfidf == 0:
        return (
            "Lo siento, no encontré información sobre ese tema "
            "en el material de Tutorías Académicas UNSAAC. "
            "Intenta reformular tu pregunta."
        )

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
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:

    st.markdown("### 🎓 Asistente de Tutorías UNSAAC")

    st.markdown("---")

    st.markdown(
        '<div class="sidebar-title">Consultas frecuentes</div>',
        unsafe_allow_html=True
    )

    temas = {
        "🎓 Tutoría":
            "¿Qué es la tutoría académica?",

        "👨‍🏫 Tutor":
            "¿Cuáles son las funciones del tutor académico?",

        "📚 Técnicas de estudio":
            "¿Qué técnicas de estudio existen?",

        "🧠 Bienestar":
            "¿Qué servicios ofrece Bienestar Universitario?",

        "📈 Rendimiento":
            "¿Qué hacer si tengo bajo rendimiento académico?",

        "📝 Matrícula":
            "¿Cómo se realiza la matrícula?",

        "💼 Perfil profesional":
            "¿Cómo elaborar un currículum vitae?"
    }

    for label, pregunta in temas.items():

        if st.button(
            label,
            key=f"sidebar_{label}",
            use_container_width=True
        ):
            st.session_state.pregunta_rapida = pregunta

    st.markdown("---")

    if st.button(
        "🗑️ Limpiar conversación",
        use_container_width=True
    ):
        st.session_state.mensajes = []
        st.session_state.pregunta_rapida = ""
        st.rerun()

    st.markdown("---")
    st.caption("Powered by TF-IDF · NLTK")

# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-box">
    <span style="font-size:36px;">🎓</span>
    <div>
        <p class="header-title">
            Asistente de Tutorías UNSAAC
        </p>
        <p class="header-sub">
            Resuelve tus dudas académicas y universitarias
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# ESTADO INICIAL
# ─────────────────────────────────────────────────────────────
if "mensajes" not in st.session_state:

    st.session_state.mensajes = [
        {
            "rol": "bot",
            "texto":
            "¡Hola! Soy el asistente virtual de Tutorías Académicas de la UNSAAC. "
            "Puedes preguntarme sobre tutoría académica, bienestar universitario, "
            "técnicas de estudio, matrícula, rendimiento académico y más. "
            "¿En qué puedo ayudarte?"
        }
    ]

if "pregunta_rapida" not in st.session_state:
    st.session_state.pregunta_rapida = ""

# ─────────────────────────────────────────────────────────────
# CARGAR CORPUS
# ─────────────────────────────────────────────────────────────
sent_tokens = cargar_corpus()

# ─────────────────────────────────────────────────────────────
# MOSTRAR HISTORIAL
# ─────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────
# PREGUNTAS SUGERIDAS
# ─────────────────────────────────────────────────────────────
if len(st.session_state.mensajes) <= 1:

    st.markdown("**Preguntas sugeridas:**")

    sugerencias = [
        "¿Qué es la tutoría académica?",
        "¿Qué es la matrícula condicionada?",
        "¿Qué técnicas de estudio existen?",
        "¿Cuándo debo acudir a Bienestar Universitario?"
    ]

    cols = st.columns(len(sugerencias))

    for col, sug in zip(cols, sugerencias):

        with col:

            if st.button(
                sug,
                key=f"sug_{sug}"
            ):
                st.session_state.pregunta_rapida = sug

# ─────────────────────────────────────────────────────────────
# CHAT INPUT
# ─────────────────────────────────────────────────────────────
user_input = st.chat_input(
    "Escribe tu pregunta aquí..."
)

if st.session_state.pregunta_rapida:

    user_input = st.session_state.pregunta_rapida
    st.session_state.pregunta_rapida = ""

if user_input:

    st.session_state.mensajes.append(
        {
            "rol": "user",
            "texto": user_input
        }
    )

    respuesta = obtener_respuesta(
        user_input,
        sent_tokens
    )

    st.session_state.mensajes.append(
        {
            "rol": "bot",
            "texto": respuesta
        }
    )

    st.rerun()
