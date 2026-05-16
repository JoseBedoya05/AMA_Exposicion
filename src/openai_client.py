from openai import OpenAI


def get_openai_client(api_key: str | None) -> OpenAI | None:
    """Construye el cliente oficial de OpenAI si existe API key."""
    if not api_key:
        return None
    return OpenAI(api_key=api_key)
