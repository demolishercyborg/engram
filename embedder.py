import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False


class Embedder:
    """
    Primary: sentence-transformers all-MiniLM-L6-v2 (downloads once, cached).
    Fallback: deterministic bag-of-chars vector — keeps everything functional
    without a network connection (e.g. first run before model is cached).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = None
        if _ST_AVAILABLE:
            try:
                print(f"[Embedder] Loading {model_name} ...")
                self.model = SentenceTransformer(model_name)
                print("[Embedder] Ready (semantic mode).")
            except Exception as e:
                print(f"[Embedder] Load failed ({e}). Using fallback.")
        else:
            print("[Embedder] sentence-transformers not found. Using fallback.")

    def embed(self, text: str) -> np.ndarray:
        if self.model:
            return self.model.encode(text, convert_to_numpy=True)
        return self._fallback(text)

    def _fallback(self, text: str, dim: int = 128) -> np.ndarray:
        vec = np.zeros(dim)
        for ch in text.lower():
            vec[ord(ch) % dim] += 1
        n = np.linalg.norm(vec)
        return vec / (n + 1e-10)

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))
