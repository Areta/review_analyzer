"""Group embeddings into clusters and describe them."""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def find_best_k(vectors, k_min=2, k_max=8, seed=42):
    """Try several K values and keep the one with the best silhouette score.

    Returns (best_k, labels, scores) where scores is {k: silhouette}.
    """
    k_max = min(k_max, len(vectors) - 1)
    results = {}
    for k in range(k_min, k_max + 1):
        labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(vectors)
        results[k] = (silhouette_score(vectors, labels), labels)

    best_k = max(results, key=lambda k: results[k][0])
    scores = {k: round(score, 3) for k, (score, _) in results.items()}
    return best_k, results[best_k][1], scores


def representative_texts(texts, vectors, labels, per_cluster=3):
    """For each cluster, return the texts closest to the cluster center."""
    texts = np.array(list(texts))
    out = {}
    for cluster_id in sorted(set(labels)):
        mask = labels == cluster_id
        center = vectors[mask].mean(axis=0)
        distances = np.linalg.norm(vectors[mask] - center, axis=1)
        closest = np.argsort(distances)[:per_cluster]
        out[int(cluster_id)] = texts[mask][closest].tolist()
    return out