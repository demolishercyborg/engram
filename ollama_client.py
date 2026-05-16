import requests

OLLAMA_URL = "http://localhost:11434/api/chat"


def make_llm_fn(model: str):
    """Returns a callable llm_fn(messages, tools) -> response dict."""
    def llm_fn(messages: list[dict], tools: list[dict] = None) -> dict:
        payload = {"model": model, "messages": messages, "stream": False}
        if tools:
            payload["tools"] = tools
        r = requests.post(OLLAMA_URL, json=payload, timeout=120)
        r.raise_for_status()
        return r.json()
    return llm_fn
