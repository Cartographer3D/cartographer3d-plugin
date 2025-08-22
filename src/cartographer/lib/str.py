def removesuffix(s: str, suffix: str) -> str:
    """Remove the specified suffix from the string if it exists."""
    if s.endswith(suffix):
        return s[: -len(suffix)]
    return s
