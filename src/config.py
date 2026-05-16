import os
from dataclasses import dataclass

import streamlit as st


@dataclass(frozen=True)
class AppConfig:
    """Configuración central de la aplicación."""

    api_key: str | None
    llm_model: str
    embedding_model: str
    default_temperature: float = 0.2
    top_k: int = 3


def _read_secret(name: str, default: str | None = None) -> str | None:
    """Lee un secreto desde Streamlit Cloud o desde variables de entorno."""
    try:
        value = st.secrets.get(name, None)
    except Exception:
        value = None
    return value or os.getenv(name, default)


def load_config() -> AppConfig:
    """Carga configuración desde st.secrets o variables de entorno."""
    return AppConfig(
        api_key=_read_secret("OPENAI_API_KEY"),
        llm_model=_read_secret("OPENAI_MODEL", "gpt-5.5") or "gpt-5.5",
        embedding_model=(
            _read_secret("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
            or "text-embedding-3-small"
        ),
    )
