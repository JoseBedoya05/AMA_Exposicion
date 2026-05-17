from pathlib import Path

import streamlit as st

from src.agent import LLMAgent
from src.config import load_config
from src.openai_client import get_openai_client
from src.retriever import SemanticRetriever, load_markdown_chunks
from src.token_utils import estimate_tokens


APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "data" / "knowledge_base_llm.md"
CSS_PATH = APP_DIR / "assets" / "styles.css"


st.set_page_config(
    page_title="LLM Agent Demo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_css() -> None:
    if CSS_PATH.exists():
        st.markdown(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def get_chunks():
    return load_markdown_chunks(DATA_PATH)


def init_messages() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Hola. Soy un agente especializado en LLM. Puedes preguntarme sobre "
                    "Transformers, atención, embeddings, RAG, modelos preentrenados o API keys."
                ),
            }
        ]


def render_header(config) -> None:
    st.markdown(
        f"""
        <div class="hero">
            <h1>🧠 LLM Agent Demo</h1>
            <p>
                Aplicación educativa para mostrar cómo se implementa un agente con un modelo preentrenado,
                embeddings, recuperación semántica y API key segura en Streamlit.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f"<div class='metric-card'><h3>LLM</h3><p><b>{config.llm_model}</b><br/>Modelo generativo para responder.</p></div>",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"<div class='metric-card'><h3>Embeddings</h3><p><b>{config.embedding_model}</b><br/>Vectores para búsqueda semántica.</p></div>",
            unsafe_allow_html=True,
        )
    with col3:
        key_status = "Configurada" if config.api_key else "No configurada"
        st.markdown(
            f"<div class='metric-card'><h3>API Key</h3><p><b>{key_status}</b><br/>Leída desde Streamlit Secrets.</p></div>",
            unsafe_allow_html=True,
        )


def render_sidebar(config):
    st.sidebar.title("⚙️ Configuración")
    st.sidebar.caption("Los modelos se definen en `.streamlit/secrets.toml` o en Streamlit Cloud Secrets.")

    use_retrieval = st.sidebar.toggle("Usar recuperación con embeddings", value=True)
    temperature = st.sidebar.slider("Temperatura", 0.0, 1.0, config.default_temperature, 0.05)

    st.sidebar.divider()
    st.sidebar.subheader("🔐 Secrets esperados")
    st.sidebar.code(
        'OPENAI_API_KEY = "sk-..."\nOPENAI_MODEL = "gpt-5.5"\nOPENAI_EMBEDDING_MODEL = "text-embedding-3-small"',
        language="toml",
    )

    st.sidebar.divider()
    st.sidebar.subheader("🧩 Flujo de la demo")
    for step in [
        "1. Pregunta del usuario",
        "2. Embedding de la pregunta",
        "3. Búsqueda semántica",
        "4. Contexto recuperado",
        "5. Prompt al LLM",
        "6. Respuesta generada",
    ]:
        st.sidebar.markdown(f"<div class='pipeline-step'>{step}</div>", unsafe_allow_html=True)

    return use_retrieval, temperature


def build_agent(config, temperature):
    client = get_openai_client(config.api_key)
    if client is None:
        return None

    chunks = get_chunks()
    retriever = SemanticRetriever(client=client, embedding_model=config.embedding_model, chunks=chunks)
    return LLMAgent(
        client=client,
        llm_model=config.llm_model,
        retriever=retriever,
        temperature=temperature,
    )


def render_chat(agent, use_retrieval: bool):
    st.subheader("💬 Pregúntale al agente")
    st.caption("Ejemplos: ¿Qué es self-attention? ¿Cómo se usa una API key? ¿Qué diferencia hay entre embeddings y LLM?")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Escribe una pregunta sobre LLM...")

    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    if agent is None:
        with st.chat_message("assistant"):
            st.error(
                "No encontré `OPENAI_API_KEY`. Configúrala en `.streamlit/secrets.toml` o en los Secrets de Streamlit Cloud."
            )
        return

    with st.chat_message("assistant"):
        with st.spinner("Generando respuesta con el agente LLM..."):
            try:
                result = agent.answer(question, use_retrieval=use_retrieval)
                st.markdown(result.answer)

                with st.expander("Ver contexto recuperado y trazabilidad"):
                    if result.retrieved_titles:
                        st.write("Fragmentos recuperados:")
                        for title in result.retrieved_titles:
                            st.markdown(f"- {title}")
                    else:
                        st.write("La recuperación semántica no se usó o no devolvió fragmentos.")

                    st.text_area(
                        "Contexto enviado al modelo",
                        value=result.context or "Sin contexto recuperado.",
                        height=220,
                    )

                st.session_state.messages.append({"role": "assistant", "content": result.answer})
            except Exception as exc:
                st.error("Ocurrió un error llamando a OpenAI. Revisa la API key, el modelo configurado y los límites de uso.")
                st.exception(exc)


def render_learning_tabs():
    st.divider()
    tab1, tab2, tab3 = st.tabs(["📌 Conceptos clave", "🧮 Tokens", "🛡️ Seguridad"])

    with tab1:
        st.markdown(
            """
            **Idea central:** un LLM recibe texto, lo tokeniza, lo convierte en embeddings internos,
            procesa el contexto con bloques Transformer y genera una respuesta prediciendo tokens probables.

            **En esta app:** usamos embeddings externos para recuperar contexto y luego enviamos ese contexto al LLM.
            Eso es una versión pequeña del patrón RAG.
            """
        )

    with tab2:
        text = st.text_area("Pega un texto para estimar tokens", "Los LLM procesan texto en tokens.")
        st.info(f"Estimación didáctica: aproximadamente {estimate_tokens(text)} tokens.")
        st.caption("Esta estimación usa una regla simple. En producción conviene usar herramientas oficiales o tokenizadores compatibles con el modelo.")

    with tab3:
        st.markdown(
            """
            Buenas prácticas para la API key:

            - No escribir la clave directamente en `app.py`.
            - No subir `.streamlit/secrets.toml` a GitHub.
            - Usar Streamlit Secrets en despliegue.
            - Rotar la clave si se expone accidentalmente.
            - No enviar datos sensibles sin revisión de privacidad.
            """
        )


def main():
    load_css()
    config = load_config()
    init_messages()
    render_header(config)
    use_retrieval, temperature = render_sidebar(config)
    agent = build_agent(config, temperature)
    render_chat(agent, use_retrieval)
    render_learning_tabs()


if __name__ == "__main__":
    main()
