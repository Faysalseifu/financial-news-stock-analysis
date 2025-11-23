"""Topic modeling and n-gram extraction for headlines.

Writes:
- outputs/top_ngrams.csv (ngram, count)
- outputs/topics.csv (topic_id, top_words)
- outputs/significant_phrases.csv (phrase, count)

Usage: python scripts/topic_modeling.py
"""
import os
import re
import sys
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


def ensure_outputs(outputs_dir: str):
    os.makedirs(outputs_dir, exist_ok=True)


def read_headlines(path: str) -> pd.Series:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    # detect headline/title column
    for c in ["headline", "title", "head"]:
        if c in df.columns:
            return df[c].fillna("").astype(str)
    # fallback: try second column if first is an index
    if df.shape[1] >= 2:
        return df.iloc[:, 1].fillna("").astype(str)
    return pd.Series([], dtype=str)


def top_ngrams(corpus, ngram_range=(1, 2), top_k=200):
    vec = CountVectorizer(stop_words="english", ngram_range=ngram_range, max_features=10000)
    X = vec.fit_transform(corpus)
    counts = X.sum(axis=0).A1
    features = vec.get_feature_names_out()
    ser = pd.Series(counts, index=features).sort_values(ascending=False)
    return ser.head(top_k)


def run_lda(corpus, n_topics=12, max_iter=8, random_state=0, n_top_words=12):
    vec = CountVectorizer(stop_words="english", max_features=20000)
    X = vec.fit_transform(corpus)
    lda = LatentDirichletAllocation(n_components=n_topics, max_iter=max_iter, random_state=random_state)
    lda.fit(X)
    feature_names = vec.get_feature_names_out()
    topics = []
    for idx, topic in enumerate(lda.components_):
        top_indices = topic.argsort()[-n_top_words:][::-1]
        top_words = [feature_names[i] for i in top_indices]
        topics.append((idx, " ".join(top_words)))
    return topics


def count_phrases(corpus, phrases):
    counts = {p: 0 for p in phrases}
    pat = re.compile("(" + "|".join(re.escape(p) for p in phrases) + ")", flags=re.IGNORECASE)
    for doc in corpus:
        for m in pat.findall(doc):
            counts[m.lower()] = counts.get(m.lower(), 0) + 1
    return pd.Series(counts).sort_values(ascending=False)


def main():
    repo_root = os.getcwd()
    data_path = os.path.join(repo_root, "data", "raw_analyst_ratings.csv")
    outputs_dir = os.path.join(repo_root, "outputs")
    ensure_outputs(outputs_dir)

    corpus = read_headlines(data_path).tolist()
    if len(corpus) == 0:
        print("No headlines found; exiting")
        sys.exit(0)

    # top unigrams and bigrams
    uni = top_ngrams(corpus, ngram_range=(1, 1), top_k=200)
    bi = top_ngrams(corpus, ngram_range=(2, 2), top_k=200)
    tri = top_ngrams(corpus, ngram_range=(3, 3), top_k=200)
    top_all = pd.concat([
        uni.rename("unigram_count"),
        bi.rename("bigram_count"),
        tri.rename("trigram_count"),
    ], axis=1).fillna(0).astype(int)
    top_all.to_csv(os.path.join(outputs_dir, "top_ngrams.csv"))

    # LDA topics
    try:
        topics = run_lda(corpus, n_topics=12)
        df_topics = pd.DataFrame(topics, columns=["topic_id", "top_words"])
        df_topics.to_csv(os.path.join(outputs_dir, "topics.csv"), index=False)
    except Exception as e:
        print("LDA failed:", e)

    # count specific significant phrases
    phrases = ["FDA approval", "price target", "earnings scheduled", "raises price", "lowers price", "upgrades", "downgrades", "earnings", "revenue"]
    phrase_counts = count_phrases(corpus, phrases)
    phrase_counts.to_csv(os.path.join(outputs_dir, "significant_phrases.csv"), header=["count"]) 

    print("Topic modeling complete. Outputs written to:", outputs_dir)


if __name__ == "__main__":
    main()
