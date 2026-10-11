"""Step 2: find reposted messages (exact and near-duplicate text) and shared links.

Usage: python 02_reposts.py FOLDER_WITH_CSVS [OUTPUT_FOLDER] [SIMILARITY] [WINDOW_HOURS]
SIMILARITY (default 0.95) is the cosine similarity at or above which two messages count as
the same message. WINDOW_HOURS (default 168, one week) is the longest time between a message
and its copy. The window keeps recurring templates (daily vaccine counts, "watch the live")
from being chained into one group across years.
The script also prints a sensitivity table for several similarity values.

Outputs:
    repost_clusters.csv   one row per group of near-identical messages
    repost_messages.csv   one row per message that belongs to a group
    repost_summary.csv    share of each channel's messages that are reposts
    shared_links.csv      links that appear in more than one channel
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer

from common import URL_RE, load_messages, normalize

MIN_CHARS = 40  # shorter messages ("Bom dia!") match each other by chance and are not counted
CHUNK = 1500


def match_key(text: str) -> str:
    """Text used for matching: no links, no accents, lowercase, no punctuation.
    Digits are kept, so updates with different numbers do not count as the same message."""
    t = URL_RE.sub(" ", text)
    t = normalize(t)
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def clean_url(url: str) -> str:
    url = url.rstrip(".,;:)!?\"'»")
    url = re.sub(r"^https?://(www\.)?", "", url.lower())
    url = re.sub(r"[?#].*$", "", url)  # drop tracking parameters
    return url.rstrip("/")


def main() -> None:
    folder = sys.argv[1]
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("results")
    threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.95
    window_hours = float(sys.argv[4]) if len(sys.argv) > 4 else 168
    out.mkdir(parents=True, exist_ok=True)

    data = load_messages(folder)
    data["key"] = data["text"].apply(match_key)
    data = data[data["key"].str.len() >= MIN_CHARS].copy().reset_index(drop=True)
    print(f"Messages with at least {MIN_CHARS} characters of text (after removing links): {len(data)}")

    # Character n-grams are robust to small edits (typos, changed numbers, added hashtags)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(4, 5), min_df=2, sublinear_tf=True, dtype=np.float32)
    X = vec.fit_transform(data["key"])

    # Pairs of messages that are similar AND close in time (computed once at a low similarity,
    # then filtered for each threshold)
    floor = 0.80
    hours = data["date"].astype("int64").to_numpy() / 3.6e12  # nanoseconds to hours
    pi, pj, ps = [], [], []
    for start in range(0, X.shape[0], CHUNK):
        sims = (X[start:start + CHUNK] @ X.T).tocoo()
        i = sims.row + start
        mask = (sims.data >= floor) & (i < sims.col) & (np.abs(hours[i] - hours[sims.col]) <= window_hours)
        pi.append(i[mask]); pj.append(sims.col[mask]); ps.append(sims.data[mask])
    pi, pj, ps = np.concatenate(pi), np.concatenate(pj), np.concatenate(ps)

    def group_at(thr: float) -> pd.DataFrame:
        """Connected groups of messages at similarity thr. Returns the messages in groups of 2+."""
        keep = ps >= thr
        n = len(data)
        graph = csr_matrix((np.ones(keep.sum()), (pi[keep], pj[keep])), shape=(n, n))
        _, labels = connected_components(graph, directed=False)
        tmp = data.copy()
        tmp["cluster"] = labels
        size = tmp.groupby("cluster")["cluster"].transform("size")
        return tmp[size >= 2]

    # Sensitivity: how many copies in other channels at different similarity values?
    sens = []
    for thr in (0.80, 0.85, 0.90, 0.95, 0.99):
        g = group_at(thr).sort_values(["cluster", "date"])
        origin = g.groupby("cluster")["channel"].transform("first")
        later = g.groupby("cluster").cumcount() > 0
        sens.append({"similarity": thr, "groups": g["cluster"].nunique(), "messages_in_groups": len(g),
                     "later_copies": int(later.sum()),
                     "later_copies_in_other_channel": int((later & (g["channel"] != origin)).sum())})
    sens = pd.DataFrame(sens)
    sens.to_csv(out / "repost_sensitivity.csv", index=False)
    print(f"Window: {window_hours:.0f} hours. Sensitivity to the similarity value:")
    print(sens.to_string(index=False))

    grouped = group_at(threshold).copy()
    print(f"\nUsing similarity >= {threshold}: groups = {grouped['cluster'].nunique()}, "
          f"messages in groups = {len(grouped)}")

    # Origin = earliest message of the group. Everything after it is a repost.
    grouped = grouped.sort_values(["cluster", "date"])
    first = grouped.groupby("cluster").head(1).set_index("cluster")
    grouped["origin_channel"] = grouped["cluster"].map(first["channel"])
    grouped["origin_date"] = grouped["cluster"].map(first["date"])
    grouped["origin_message_id"] = grouped["cluster"].map(first["message_id"])
    grouped["is_origin"] = grouped.groupby("cluster").cumcount() == 0
    grouped["lag_hours"] = (grouped["date"] - grouped["origin_date"]).dt.total_seconds() / 3600
    grouped["cross_channel"] = grouped["channel"] != grouped["origin_channel"]

    # One row per group
    def summarize(g: pd.DataFrame) -> pd.Series:
        return pd.Series({
            "n_messages": len(g),
            "n_channels": g["channel"].nunique(),
            "channels": ", ".join(sorted(g["channel"].unique())),
            "origin_channel": g["origin_channel"].iloc[0],
            "first_date": g["date"].min(),
            "last_date": g["date"].max(),
            "median_lag_hours_cross_channel": g.loc[g["cross_channel"], "lag_hours"].median(),
            "example_text": g["text"].iloc[0].replace("\n", " ")[:300],
        })

    clusters = grouped.groupby("cluster").apply(summarize, include_groups=False).reset_index()
    clusters = clusters.sort_values("n_messages", ascending=False)
    clusters.to_csv(out / "repost_clusters.csv", index=False, encoding="utf-8-sig")

    cols_out = ["cluster", "channel", "message_id", "date", "views", "origin_channel", "origin_message_id",
                "origin_date", "is_origin", "cross_channel", "lag_hours", "text"]
    grouped[cols_out].to_csv(out / "repost_messages.csv", index=False, encoding="utf-8-sig")

    # Share of messages that are reposts, per channel
    total = data.groupby("channel").size().rename("messages_checked")
    in_group = grouped.groupby("channel").size().rename("in_repeated_group")
    reposts_any = grouped[~grouped["is_origin"]].groupby("channel").size().rename("later_copies")
    reposts_cross = grouped[~grouped["is_origin"] & grouped["cross_channel"]].groupby("channel").size().rename(
        "later_copies_of_another_channel")
    summary = pd.concat([total, in_group, reposts_any, reposts_cross], axis=1).fillna(0).astype(int)
    summary["later_copies_pct"] = (100 * summary["later_copies"] / summary["messages_checked"]).round(1)
    summary["copied_from_other_channel_pct"] = (
        100 * summary["later_copies_of_another_channel"] / summary["messages_checked"]).round(1)
    summary.to_csv(out / "repost_summary.csv", encoding="utf-8-sig")
    print(summary.to_string())

    # Links shared across channels
    all_msgs = load_messages(folder)
    link_rows = []
    for _, r in all_msgs.iterrows():
        for u in URL_RE.findall(r["text"]):
            link_rows.append((clean_url(u), r["channel"], r["date"]))
    links = pd.DataFrame(link_rows, columns=["url", "channel", "date"])
    link_stats = links.groupby("url").agg(
        n_posts=("channel", "size"), n_channels=("channel", "nunique"),
        channels=("channel", lambda s: ", ".join(sorted(set(s)))), first_date=("date", "min"))
    shared = link_stats[link_stats["n_channels"] >= 2].sort_values("n_posts", ascending=False)
    shared.to_csv(out / "shared_links.csv", encoding="utf-8-sig")
    print(f"\nLinks posted in 2+ channels: {len(shared)} (of {len(link_stats)} distinct links)")


if __name__ == "__main__":
    main()
