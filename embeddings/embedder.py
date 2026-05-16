import numpy as np
import torch

try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False


class Embedder:
    """
    Primary: sentence-transformers all-MiniLM-L6-v2.
    Fallback: deterministic bag-of-chars vector.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        if _ST_AVAILABLE:
            try:
                print(f"[Embedder] Loading {model_name} on {self._device} ...")
                self.model = SentenceTransformer(model_name, device=self._device)
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
