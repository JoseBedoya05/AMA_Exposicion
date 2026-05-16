def estimate_tokens(text: str) -> int:
    """Estimación sencilla para fines didácticos: ~1 token por cada 4 caracteres."""
    if not text:
        return 0
    return max(1, round(len(text) / 4))
