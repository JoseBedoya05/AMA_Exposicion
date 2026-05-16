# LLM Agent Demo con OpenAI + Streamlit

Aplicación educativa en Streamlit para explicar cómo se implementa un **agente especializado en Large Language Models (LLM)** usando modelos preentrenados de OpenAI, embeddings y una pequeña base de conocimiento local.

La app está pensada para una exposición corta de Ciencia de Datos. Permite mostrar:

- Uso de un modelo LLM preentrenado vía API.
- Uso de embeddings para recuperación semántica simple.
- Separación entre interfaz, agente, cliente OpenAI y retriever.
- Manejo seguro de la API key mediante `st.secrets`.
- Ejemplo de flujo tipo RAG: pregunta → embeddings → contexto relevante → respuesta del LLM.

---

## 1. Estructura del repositorio

```text
llm_streamlit_agent_demo/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   └── secrets.toml.example
├── assets/
│   └── styles.css
├── data/
│   └── knowledge_base_llm.md
└── src/
    ├── __init__.py
    ├── agent.py
    ├── config.py
    ├── openai_client.py
    ├── prompts.py
    ├── retriever.py
    └── token_utils.py
```

---

## 2. Configuración local

### Crear entorno virtual

```bash
python -m venv .venv
```

### Activar entorno

En Windows PowerShell:

```powershell
.venv\Scripts\activate
```

En macOS/Linux:

```bash
source .venv/bin/activate
```

### Instalar dependencias

```bash
pip install -r requirements.txt
```

---

## 3. Configurar la API key

Crea un archivo local:

```text
.streamlit/secrets.toml
```

Con este contenido:

```toml
OPENAI_API_KEY = "sk-REEMPLAZA_CON_TU_API_KEY"
OPENAI_MODEL = "gpt-5.5"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
```

> Importante: `secrets.toml` no debe subirse a GitHub. El repositorio solo incluye `secrets.toml.example`.

---

## 4. Ejecutar la app

```bash
streamlit run app.py
```

---

## 5. Despliegue en Streamlit Community Cloud

1. Sube este repositorio a GitHub.
2. Entra a Streamlit Community Cloud.
3. Crea una nueva app apuntando al repositorio.
4. Define como archivo principal: `app.py`.
5. En la sección **Secrets**, agrega:

```toml
OPENAI_API_KEY = "sk-REEMPLAZA_CON_TU_API_KEY"
OPENAI_MODEL = "gpt-5.5"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
```

6. Despliega la aplicación.

---

## 6. Modelos usados

Por defecto se propone:

- **LLM:** `gpt-5.5`, por su capacidad para razonamiento, explicación técnica y generación de respuestas pedagógicas.
- **Embeddings:** `text-embedding-3-small`, porque ofrece buen equilibrio entre calidad, costo y velocidad para una demo educativa.

Puedes cambiarlos desde `st.secrets` sin modificar el código.

---

## 7. Buenas prácticas implementadas

- La API key no aparece quemada en el código.
- Se usa `st.secrets` para Streamlit Cloud.
- Se mantiene separación modular del proyecto.
- El agente tiene un prompt de sistema especializado en LLM.
- El retriever usa embeddings para recuperar fragmentos relevantes.
- La interfaz permite explicar visualmente el flujo de un LLM aplicado.

---

## 8. Advertencia académica

Esta app es una demo educativa. Para producción se recomienda:

- Controlar costos y límites de uso.
- Agregar autenticación de usuarios.
- Usar una base vectorial persistente como ChromaDB, FAISS, Pinecone, Weaviate o pgvector.
- Registrar métricas, trazabilidad y evaluación de respuestas.
- Validar privacidad y tratamiento de datos antes de enviar información sensible a una API externa.
