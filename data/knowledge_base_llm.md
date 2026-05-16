# Base de conocimiento: LLM para Ciencia de Datos

## ¿Qué es un LLM?
Un Large Language Model es un modelo de aprendizaje profundo entrenado con grandes volúmenes de texto para modelar la probabilidad de secuencias lingüísticas. En modelos autoregresivos, el objetivo central es predecir el siguiente token dado el contexto anterior: P(token_t | token_1, token_2, ..., token_{t-1}). Esa predicción permite generar texto, responder preguntas, resumir documentos, escribir código o asistir procesos de análisis.

## Tokenización y embeddings
Antes de entrar al modelo, el texto se divide en tokens. Un token puede ser una palabra, parte de una palabra o un símbolo. Luego cada token se transforma en un vector numérico llamado embedding. Los embeddings capturan relaciones semánticas y permiten operaciones como búsqueda por similitud, clasificación, clustering y recuperación de contexto para RAG.

## Arquitectura Transformer
La arquitectura Transformer reemplazó las redes recurrentes tradicionales en muchas tareas de NLP. Su fortaleza está en procesar secuencias mediante mecanismos de atención, permitiendo que cada token se relacione con otros tokens del contexto. Un bloque Transformer incluye self-attention, multi-head attention, capas feed-forward, normalización y conexiones residuales.

## Mecanismo de atención
En self-attention, cada token se proyecta en tres vectores: Query, Key y Value. Query representa lo que un token busca; Key representa lo que cada token ofrece para ser encontrado; Value contiene la información que se transfiere cuando un token recibe atención. La fórmula general es Attention(Q,K,V)=softmax(QK^T / sqrt(d_k))V. El producto QK^T mide compatibilidad, softmax convierte compatibilidades en pesos, y esos pesos se aplican sobre V.

## Modelos preentrenados
Un modelo preentrenado ya aprendió patrones generales del lenguaje a partir de grandes corpus. Los desarrolladores pueden usarlo mediante API sin entrenarlo desde cero. En un desarrollo propio, el modelo puede adaptarse mediante prompting, RAG, herramientas, function calling o fine-tuning. Para muchos casos empresariales, RAG es una primera estrategia recomendable porque permite conectar el modelo con documentos propios sin modificar sus pesos.

## RAG con embeddings
Retrieval-Augmented Generation combina recuperación de información y generación. Primero se convierten documentos en embeddings y se almacenan en un índice. Cuando el usuario pregunta, la pregunta también se convierte en embedding. Luego se recuperan los fragmentos más similares y se envían al LLM como contexto. Esto reduce respuestas genéricas y mejora la trazabilidad porque el modelo responde con base en información recuperada.

## Uso de API key de OpenAI
Para usar modelos de OpenAI desde código se requiere una API key. Esta clave funciona como credencial de acceso y no debe escribirse directamente en el código ni subirse a GitHub. En Streamlit Cloud se recomienda guardarla en Secrets y leerla con st.secrets. En local puede guardarse en .streamlit/secrets.toml o en variables de entorno. Si una clave se expone, debe revocarse o rotarse.

## Buenas prácticas para una app educativa
Una app sencilla debe separar interfaz, agente, cliente de API y recuperación de documentos. También debe evitar exponer secretos, manejar errores, mostrar el flujo de datos y dejar claro cuándo la respuesta viene del modelo y cuándo usa contexto recuperado. Para una exposición, es útil mostrar el pipeline: pregunta del usuario, embedding de la pregunta, búsqueda semántica, construcción del prompt y generación de respuesta.
