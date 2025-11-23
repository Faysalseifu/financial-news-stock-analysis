import os
import re
import sys
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer


def ensure_outputs(outputs_dir: str):
    os.makedirs(outputs_dir, exist_ok=True)


def read_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")

    # Try reading with header first, then fallback to a default schema
    try:
        # Read without forcing date parsing here; we'll parse explicitly below
        df = pd.read_csv(path)
        return df
    except Exception:
        cols = ["id", "headline", "url", "author", "published_at", "ticker"]
        df = pd.read_csv(path, header=None, names=cols)
        return df


def extract_domain(author: str) -> str:
    if pd.isna(author):
        return ""
    m = re.search(r"@([A-Za-z0-9.-]+)", str(author))
    return m.group(1).lower() if m else ""


def run_eda(data_path: str, outputs_dir: str):
    ensure_outputs(outputs_dir)

    df = read_data(data_path)

    # HEADLINE LENGTH STATS
    df["headline"] = df["headline"].fillna("").astype(str)
    df["headline_len"] = df["headline"].str.len()
    headline_stats = df["headline_len"].describe()
    headline_stats.to_csv(os.path.join(outputs_dir, "headline_length_stats.csv"))

    # PUBLISHER COUNTS
    if "author" not in df.columns:
        df["author"] = ""
    df["author"] = df["author"].fillna("Unknown")
    publisher_counts = df["author"].value_counts()
    publisher_counts.to_csv(os.path.join(outputs_dir, "publisher_counts.csv"))

    # PUBLICATION DATE / TIME TRENDS
    if "published_at" in df.columns:
        # Robustly convert published_at to datetimes. Use utc=True to avoid mixed tz issues
        df["date"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)

        # Only operate on rows where date parsing succeeded
        valid = df["date"].notna()
        if valid.any():
            daily = df.loc[valid].groupby(df.loc[valid, "date"].dt.date).size().rename("count")
            daily.to_csv(os.path.join(outputs_dir, "daily_publication_counts.csv"))

            hour_counts = df.loc[valid, "date"].dt.hour.value_counts().sort_index()
            hour_counts.to_csv(os.path.join(outputs_dir, "hourly_publication_counts.csv"))
        else:
            # No parseable dates found; write empty files so downstream checks still work
            pd.Series(dtype=int).to_csv(os.path.join(outputs_dir, "daily_publication_counts.csv"))
            pd.Series(dtype=int).to_csv(os.path.join(outputs_dir, "hourly_publication_counts.csv"))
    else:
        # If published_at missing, create empty placeholders
        pd.Series(dtype=int).to_csv(os.path.join(outputs_dir, "daily_publication_counts.csv"))
        pd.Series(dtype=int).to_csv(os.path.join(outputs_dir, "hourly_publication_counts.csv"))

    # AUTHOR DOMAIN ANALYSIS
    df["author_domain"] = df["author"].apply(extract_domain)
    domain_counts = df["author_domain"].replace("", pd.NA).value_counts(dropna=True)
    domain_counts.to_csv(os.path.join(outputs_dir, "author_domain_counts.csv"))

    # SIMPLE TEXT ANALYSIS - TOP KEYWORDS
    corpus = df["headline"].fillna("").astype(str).tolist()
    vec = CountVectorizer(stop_words="english", ngram_range=(1, 2), max_features=200)
    X = vec.fit_transform(corpus)
    counts = X.sum(axis=0).A1
    features = vec.get_feature_names_out()
    top = pd.Series(counts, index=features).sort_values(ascending=False)
    top.to_csv(os.path.join(outputs_dir, "top_keywords.csv"))

    print("EDA complete. Outputs written to:", outputs_dir)


def main():
    repo_root = os.getcwd()
    data_path = os.path.join(repo_root, "data", "raw_analyst_ratings.csv")
    outputs_dir = os.path.join(repo_root, "outputs")

    try:
        run_eda(data_path, outputs_dir)
    except Exception as exc:
        print("EDA failed:", exc)
        sys.exit(2)


if __name__ == "__main__":
    main()
