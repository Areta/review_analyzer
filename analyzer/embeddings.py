"""Turn texts into vectors (embeddings)."""
import numpy as np


def get_embeddings(texts, backend="ollama", model="nomic-embed-text"):
    """Return a 2D array: one vector per text.

    backend="ollama": real semantic embeddings from a local model (needs Ollama running).
    backend="tfidf":  lightweight fallback based on word statistics. It works offline and
                      is used for quick tests, but it is NOT semantic: it only sees shared words.
    """
    if backend == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return np.array(OllamaEmbeddings(model=model).embed_documents(list(texts)))

    if backend == "tfidf":
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        tfidf = TfidfVectorizer(stop_words="english").fit_transform(texts)
        n_components = max(2, min(30, tfidf.shape[1] - 1, tfidf.shape[0] - 1))
        return TruncatedSVD(n_components=n_components, random_state=42).fit_transform(tfidf)

    raise ValueError(f"Unknown backend: {backend}")