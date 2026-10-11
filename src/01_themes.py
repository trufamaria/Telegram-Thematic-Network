"""Step 1: find common themes with topic modeling (TF-IDF + NMF).

Usage: python 01_themes.py FOLDER_WITH_CSVS [OUTPUT_FOLDER] [K]
K is the number of themes (default 12). Use K = auto to test several values and pick the one
with the best coherence.

Method
  1. Clean the text (no links, mentions, accents, stop words) and weight words by TF-IDF.
  2. Fit a non-negative matrix factorization (NMF) model. Each theme is a set of words that
     occur together; each message is assigned to its strongest theme.
  3. If one theme is much larger than the rest (a catch-all), split it again with a second,
     smaller model, so the sub-themes can be told apart.
  4. Name the themes by hand: read themes.csv (top words and example messages), then write
     the names in theme_names.csv (columns: theme, name). The script uses that file if it exists.

Topic models are a way to find candidate themes. Read the example messages and check that
each theme means what its name says before you use the results in a paper.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from common import load_messages

MIN_TOKENS = 6        # messages shorter than this (after cleaning) are too short to assign to a theme
TOP_N = 12            # words shown per theme
RANDOM_STATE = 0
SPLIT_LARGEST = True  # split the largest theme into sub-themes
SPLIT_IF_SHARE_ABOVE = 0.20  # ... only if it holds more than 20% of the messages
K_SUB = 5


def npmi_coherence(top_words_idx, binary_matrix) -> float:
    """Mean NPMI coherence of the top words of each topic (higher = the words appear together)."""
    n_docs = binary_matrix.shape[0]
    doc_freq = np.asarray(binary_matrix.sum(axis=0)).ravel()
    scores = []
    for words in top_words_idx:
        sub = binary_matrix[:, words]
        co = (sub.T @ sub).toarray()
        pair_scores = []
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                p_ij = co[i, j] / n_docs
                p_i, p_j = doc_freq[words[i]] / n_docs, doc_freq[words[j]] / n_docs
                pair_scores.append(-1.0 if p_ij == 0 else np.log(p_ij / (p_i * p_j)) / -np.log(p_ij))
        scores.append(np.mean(pair_scores))
    return float(np.mean(scores))


def fit_nmf(X, k: int) -> NMF:
    return NMF(n_components=k, init="nndsvd", random_state=RANDOM_STATE, max_iter=400).fit(X)


def sort_key(label: str):
    parts = label.split(".")
    return tuple(int(p) for p in parts)


def heatmap(table: pd.DataFrame, names: dict, path: Path) -> None:
    short = {"eduardobolsonaro": "Eduardo", "flaviobolsonaro": "Flávio",
             "jairbolsonaro": "Jair", "michellebolsonaroo": "Michelle"}
    labels = [f"{names[c]} ({c})" if c in names else str(c) for c in table.columns]
    fig, ax = plt.subplots(figsize=(10.5, 1.4 + 0.38 * len(labels)))
    im = ax.imshow(table.T.values, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(table))); ax.set_xticklabels([short.get(c, c) for c in table.index], fontsize=10)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=9)
    for i in range(table.shape[1]):
        for j in range(table.shape[0]):
            v = table.iloc[j, i]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8.5,
                    color="white" if v > table.values.max() * 0.55 else "#222222")
    fig.suptitle("Share of each channel's messages in each theme (%)", fontsize=11, x=0.5, y=0.99)
    fig.colorbar(im, ax=ax, shrink=0.6, label="%", pad=0.02)
    fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(path, dpi=200); plt.close(fig)


def main() -> None:
    folder = sys.argv[1]
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("results")
    k_arg = sys.argv[3] if len(sys.argv) > 3 else "12"
    out.mkdir(parents=True, exist_ok=True)

    data = load_messages(folder)
    docs = data[data["n_tokens"] >= MIN_TOKENS].copy().reset_index(drop=True)
    print(f"Messages with text: {len(data)}. Long enough for themes: {len(docs)}.")

    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=10, max_df=0.4, max_features=20000, sublinear_tf=True)
    X = tfidf.fit_transform(docs["text_clean"])
    terms = np.array(tfidf.get_feature_names_out())

    if k_arg == "auto":
        binary = CountVectorizer(vocabulary=tfidf.vocabulary_, binary=True, ngram_range=(1, 2)).fit_transform(docs["text_clean"])
        rows = []
        for k in (6, 8, 10, 12, 14, 16, 18):
            m = fit_nmf(X, k)
            top_idx = [np.argsort(c)[::-1][:10] for c in m.components_]
            rows.append({"k": k, "coherence_npmi": round(npmi_coherence(top_idx, binary), 4)})
        selection = pd.DataFrame(rows)
        selection.to_csv(out / "theme_model_selection.csv", index=False)
        print(selection.to_string(index=False))
        k = int(selection.sort_values("coherence_npmi", ascending=False).iloc[0]["k"])
    else:
        k = int(k_arg)
    print(f"Using k = {k}")

    model = fit_nmf(X, k)
    W = model.transform(X)
    docs["theme"] = W.argmax(axis=1).astype(str)
    docs["theme_strength"] = W.max(axis=1)
    top_words = {str(t): terms[np.argsort(model.components_[t])[::-1][:TOP_N]] for t in range(k)}

    # Split the catch-all theme, if there is one
    biggest = docs["theme"].value_counts().idxmax()
    if SPLIT_LARGEST and docs["theme"].value_counts(normalize=True).max() > SPLIT_IF_SHARE_ABOVE:
        idx = docs.index[docs["theme"] == biggest]
        sub_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=8, max_df=0.3, sublinear_tf=True)
        Xs = sub_vec.fit_transform(docs.loc[idx, "text_clean"])
        sub_terms = np.array(sub_vec.get_feature_names_out())
        sub_model = fit_nmf(Xs, K_SUB)
        Ws = sub_model.transform(Xs)
        docs.loc[idx, "theme"] = [f"{biggest}.{s}" for s in Ws.argmax(axis=1)]
        docs.loc[idx, "theme_strength"] = Ws.max(axis=1)
        del top_words[biggest]
        for s in range(K_SUB):
            top_words[f"{biggest}.{s}"] = sub_terms[np.argsort(sub_model.components_[s])[::-1][:TOP_N]]
        print(f"Theme {biggest} was the largest ({len(idx)} messages) and was split into {K_SUB} sub-themes.")

    names_path = out / "theme_names.csv"
    names = dict(pd.read_csv(names_path, dtype=str).values) if names_path.exists() else {}

    rows = []
    for label in sorted(top_words, key=sort_key):
        members = docs[docs["theme"] == label].sort_values("theme_strength", ascending=False)
        rows.append({
            "theme": label,
            "name": names.get(label, ""),
            "top_words": ", ".join(top_words[label]),
            "n_messages": len(members),
            "share_pct": round(100 * len(members) / len(docs), 1),
            "n_channels": members["channel"].nunique(),
            "example_messages": " || ".join(s.replace("\n", " ")[:220] for s in members["text"].head(4)),
        })
    themes = pd.DataFrame(rows)
    themes.to_csv(out / "themes.csv", index=False, encoding="utf-8-sig")

    order = sorted(docs["theme"].unique(), key=sort_key)
    by_channel = pd.crosstab(docs["channel"], docs["theme"], normalize="index").mul(100).round(1)[order]
    by_channel.to_csv(out / "themes_by_channel_pct.csv")
    by_year = pd.crosstab(docs["year"], docs["theme"], normalize="index").mul(100).round(1)[order]
    by_year.to_csv(out / "themes_by_year_pct.csv")

    keep = ["channel", "message_id", "date", "views", "theme", "theme_strength", "text"]
    docs[keep].to_csv(out / "message_themes.csv", index=False, encoding="utf-8-sig")
    heatmap(by_channel, names, out / "fig_themes_by_channel.png")
    print(themes[["theme", "name", "n_messages", "share_pct", "n_channels", "top_words"]].to_string(index=False))


if __name__ == "__main__":
    main()
