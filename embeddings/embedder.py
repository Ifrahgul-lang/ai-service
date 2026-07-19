"""
3.2 Embeddings
IMPORTANT: this MUST use the exact same model Abdul-haq used to embed the
textbook chunks. Switched to sentence-transformers (open-source, runs
locally, no API key needed) since an OpenAI key wasn't available on
Abdul-haq's side. Model: all-MiniLM-L6-v2 -> 384-dimensional vectors.
Any mismatch here (different model, different dimension) silently
breaks retrieval quality.
"""
from sentence_transformers import SentenceTransformer
from config import settings

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.EMBEDDING_MODEL)
    return _model


def embed_text(text: str) -> list[float]:
    """Returns the embedding vector for a single string of text."""
    if not text or not text.strip():
        raise ValueError("Cannot embed empty text.")

    model = _get_model()
    vector = model.encode(text, convert_to_numpy=True)
    return vector.tolist()


def embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Batch version — useful if Abdul-haq's ingestion script (or future
    re-indexing) needs to embed many textbook chunks at once.
    """
    texts = [t for t in texts if t and t.strip()]
    if not texts:
        return []

    model = _get_model()
    vectors = model.encode(texts, convert_to_numpy=True)
    return [v.tolist() for v in vectors]
