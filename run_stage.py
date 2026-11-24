"""Stage 1: load reviews, embed them, find clusters. No LLM yet.

Usage:
    python run_stage.py                                  # sample data, Ollama embeddings
    python run_stage.py --backend tfidf                  # no Ollama needed (quick test)
    python run_stage.py --data data/my_reviews.csv --column review_text
"""
import argparse

import pandas as pd

from analyzer.clustering import find_best_k, representative_texts
from analyzer.embeddings import get_embeddings


def load_reviews(path, column="text"):
    df = pd.read_csv(path)
    if column not in df.columns:
        raise SystemExit(f"Column '{column}' not found. Available: {list(df.columns)}")
    df = df.rename(columns={column: "text"})
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() >= 10]   # drop empty / too-short reviews
    return df.drop_duplicates(subset="text").reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/sample_reviews.csv")
    parser.add_argument("--column", default="text")
    parser.add_argument("--backend", default="ollama", choices=["ollama", "tfidf"])
    parser.add_argument("--kmin", type=int, default=2)
    parser.add_argument("--kmax", type=int, default=8)
    args = parser.parse_args()

    df = load_reviews(args.data, args.column)
    print(f"Loaded {len(df)} reviews")

    vectors = get_embeddings(df["text"], backend=args.backend)
    best_k, labels, scores = find_best_k(vectors, args.kmin, args.kmax)
    print(f"Silhouette by K: {scores}")
    print(f"Best K = {best_k}\n")

    df["cluster"] = labels
    examples = representative_texts(df["text"], vectors, labels)
    for cluster_id, texts in examples.items():
        size = int((labels == cluster_id).sum())
        print(f"Cluster {cluster_id} ({size} reviews):")
        for t in texts:
            print(f"  - {t}")
        print()

    df.to_csv("results/clusters.csv", index=False)
    print("Saved results/clusters.csv")


if __name__ == "__main__":
    main()