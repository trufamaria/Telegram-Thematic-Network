"""Step 3: network analysis of reposting between channels.

Usage: python 03_network.py [OUTPUT_FOLDER]
Reads repost_messages.csv from step 2 (and message_themes.csv from step 1, if present).

Network: each channel is a node. A directed edge A -> B means channel B posted a copy of a
message that channel A had posted first. Edge weight = number of copies.
Note: "posted first" means first among the channels in the data. A message can come from a
source outside these channels (for example a ministry), so the direction shows who posted
first in this data, not who wrote the message.

Outputs: network_edges.csv, network_nodes.csv, repost_network.gexf (opens in Gephi),
         network_by_period.csv, lag_summary.csv, themes_of_reposts.csv (if themes exist),
         fig_network.png, fig_lag.png, fig_copies_by_year.png
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch

SHORT = {"eduardobolsonaro": "Eduardo", "flaviobolsonaro": "Flávio",
         "jairbolsonaro": "Jair", "michellebolsonaroo": "Michelle"}
COLORS = {"Eduardo": "#2a6f97", "Flávio": "#c1121f", "Jair": "#2d6a4f", "Michelle": "#7b2cbf"}


def short(name: str) -> str:
    return SHORT.get(name, name)


def build_graph(copies: pd.DataFrame, sizes: pd.Series) -> nx.DiGraph:
    g = nx.DiGraph()
    for ch, n in sizes.items():
        g.add_node(short(ch), messages=int(n))
    edges = copies.groupby(["origin_channel", "channel"]).agg(
        weight=("message_id", "size"), median_lag_hours=("lag_hours", "median")).reset_index()
    for _, e in edges.iterrows():
        g.add_edge(short(e["origin_channel"]), short(e["channel"]),
                   weight=int(e["weight"]), median_lag_hours=round(float(e["median_lag_hours"]), 2))
    return g


def node_table(g: nx.DiGraph) -> pd.DataFrame:
    out_s = dict(g.out_degree(weight="weight"))
    in_s = dict(g.in_degree(weight="weight"))
    rows = []
    for n in g.nodes:
        rows.append({
            "channel": n, "messages_checked": g.nodes[n]["messages"],
            "copies_made_by_others_of_my_posts (out-strength)": out_s[n],
            "copies_I_made_of_others_posts (in-strength)": in_s[n],
            "net_flow (out minus in)": out_s[n] - in_s[n],
            "pagerank": round(nx.pagerank(g, weight="weight")[n], 3),
        })
    return pd.DataFrame(rows).sort_values("net_flow (out minus in)", ascending=False)


def draw_network(g: nx.DiGraph, path: Path) -> None:
    pos = {"Jair": (0.0, 0.0), "Flávio": (1.0, 0.0), "Eduardo": (0.5, 0.85), "Michelle": (0.5, -0.85)}
    pos = {n: pos.get(n, (np.random.rand(), np.random.rand())) for n in g.nodes}
    fig, ax = plt.subplots(figsize=(8, 7))
    max_w = max(d["weight"] for _, _, d in g.edges(data=True))
    max_n = max(d["messages"] for _, d in g.nodes(data=True))
    radius = {n: 0.115 + 0.10 * np.sqrt(d["messages"] / max_n) for n, d in g.nodes(data=True)}
    pts_per_unit = 140  # approximate, to stop arrows at the node border
    for u, v, d in g.edges(data=True):
        width = 0.8 + 8 * d["weight"] / max_w
        both = g.has_edge(v, u)
        rad = 0.22 if both else 0.0
        ax.add_patch(FancyArrowPatch(
            pos[u], pos[v], connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>,head_length=0.7,head_width=0.35",
            mutation_scale=16, linewidth=width, color="#444444", alpha=0.6,
            shrinkA=radius[u] * pts_per_unit, shrinkB=radius[v] * pts_per_unit, zorder=1))
        # label at the middle of the curve
        dx, dy = pos[v][0] - pos[u][0], pos[v][1] - pos[u][1]
        lx = (pos[u][0] + pos[v][0]) / 2 + 0.5 * rad * dy
        ly = (pos[u][1] + pos[v][1]) / 2 - 0.5 * rad * dx
        ax.text(lx, ly, str(d["weight"]), fontsize=9 if d["weight"] >= 10 else 8, ha="center", va="center",
                color="#111111", zorder=3,
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#bbbbbb", lw=0.5, alpha=0.95))
    for n in g.nodes:
        ax.add_patch(plt.Circle(pos[n], radius[n], color=COLORS.get(n, "#888888"), zorder=2))
        ax.text(pos[n][0], pos[n][1], n, ha="center", va="center", color="white",
                fontsize=11 if radius[n] > 0.14 else 8.5, fontweight="bold", zorder=4)
        label = f"{g.nodes[n]['messages']:,} messages"
        x, y = pos[n]
        if n == "Jair":
            ax.text(x - radius[n] - 0.04, y, label, ha="right", va="center", fontsize=8.5, color="#444444")
        elif n == "Flávio":
            ax.text(x + radius[n] + 0.04, y, label, ha="left", va="center", fontsize=8.5, color="#444444")
        elif n == "Eduardo":
            ax.text(x, y + radius[n] + 0.05, label, ha="center", va="bottom", fontsize=8.5, color="#444444")
        else:
            ax.text(x, y - radius[n] - 0.05, label, ha="center", va="top", fontsize=8.5, color="#444444")
    ax.set_xlim(-0.8, 1.8); ax.set_ylim(-1.3, 1.3); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("Who copies whom: reposts between four Telegram channels\n"
                 "Arrow: from the channel that posted first to the channel that posted the copy. Number: copies.",
                 fontsize=10.5)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")
    msgs = pd.read_csv(out / "repost_messages.csv", parse_dates=["date", "origin_date"])
    sizes = pd.read_csv(out / "repost_summary.csv", index_col=0)["messages_checked"]

    copies = msgs[(~msgs["is_origin"]) & (msgs["cross_channel"])].copy()
    print(f"Copies of a message first posted in another channel: {len(copies)}")

    g = build_graph(copies, sizes)
    edges = nx.to_pandas_edgelist(g).rename(columns={"source": "from_channel_posted_first", "target": "to_channel_copied"})
    edges = edges[["from_channel_posted_first", "to_channel_copied", "weight", "median_lag_hours"]]
    edges = edges.sort_values("weight", ascending=False)
    edges.to_csv(out / "network_edges.csv", index=False)
    nodes = node_table(g)
    nodes.to_csv(out / "network_nodes.csv", index=False)
    nx.write_gexf(g, out / "repost_network.gexf")

    # Network measures
    pairs_both_ways = sum(1 for u, v in g.edges if g.has_edge(v, u)) / 2
    print(f"Channels: {g.number_of_nodes()}, links: {g.number_of_edges()}, density: {nx.density(g):.2f}, "
          f"pairs with copies in both directions: {int(pairs_both_ways)}")
    print(nodes.to_string(index=False))
    print(edges.to_string(index=False))

    # By period: the network may change over time
    copies["year"] = copies["date"].dt.year
    copies["period"] = np.where(copies["year"] <= 2022, "2021-2022", "2023-2026")
    by_period = copies.groupby(["period", "origin_channel", "channel"]).agg(
        copies=("message_id", "size"), median_lag_hours=("lag_hours", "median")).reset_index()
    by_period["origin_channel"] = by_period["origin_channel"].map(short)
    by_period["channel"] = by_period["channel"].map(short)
    by_period.to_csv(out / "network_by_period.csv", index=False)

    # Timing: how fast are copies posted?
    lag = copies["lag_hours"]
    lag_summary = pd.DataFrame({
        "statistic": ["copies", "median hours", "mean hours", "within 1 hour %", "within 6 hours %",
                      "within 24 hours %", "within 7 days %"],
        "value": [len(lag), round(lag.median(), 2), round(lag.mean(), 1), round(100 * (lag <= 1).mean(), 1),
                  round(100 * (lag <= 6).mean(), 1), round(100 * (lag <= 24).mean(), 1),
                  round(100 * (lag <= 168).mean(), 1)]})
    lag_summary.to_csv(out / "lag_summary.csv", index=False)
    print(lag_summary.to_string(index=False))

    # What gets reposted? (uses the themes from step 1)
    themes_path = out / "message_themes.csv"
    if themes_path.exists():
        th = pd.read_csv(themes_path, dtype={"theme": str})[["channel", "message_id", "theme"]]
        copied_ids = set(zip(msgs.loc[msgs["cross_channel"] | ~msgs["is_origin"], "channel"],
                             msgs.loc[msgs["cross_channel"] | ~msgs["is_origin"], "message_id"]))
        origin_ids = set(zip(msgs.loc[msgs["is_origin"], "channel"], msgs.loc[msgs["is_origin"], "message_id"]))
        th["in_repost_group"] = [(c, m) in copied_ids or (c, m) in origin_ids for c, m in zip(th["channel"], th["message_id"])]
        tab = th.groupby("theme")["in_repost_group"].agg(["size", "sum"]).rename(
            columns={"size": "messages", "sum": "in_repost_group"})
        tab["pct_in_repost_group"] = (100 * tab["in_repost_group"] / tab["messages"]).round(1)
        tab.to_csv(out / "themes_of_reposts.csv")
        print(tab.to_string())

    # Figures
    draw_network(g, out / "fig_network.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(np.clip(lag, 0.01, None), bins=np.logspace(-2, np.log10(168), 40), color="#2d6a4f")
    ax.set_xscale("log"); ax.set_xlabel("Hours between the first post and the copy (log scale)")
    ax.set_ylabel("Copies"); ax.set_title("Copies appear quickly"); ax.axvline(lag.median(), color="#c1121f", ls="--")
    ax.text(lag.median() * 1.1, ax.get_ylim()[1] * 0.9, f"median {lag.median():.1f} h", color="#c1121f")
    fig.tight_layout(); fig.savefig(out / "fig_lag.png", dpi=200); plt.close(fig)

    yearly = copies.assign(pair=copies["origin_channel"].map(short) + " → " + copies["channel"].map(short)) \
        .groupby(["year", "pair"]).size().unstack(fill_value=0)
    top = yearly.sum().sort_values(ascending=False).index[:3]
    yearly["Other pairs"] = yearly.drop(columns=top).sum(axis=1)
    yearly[list(top) + ["Other pairs"]].plot(kind="bar", stacked=True, figsize=(7.5, 4.2), color=["#2d6a4f", "#c1121f", "#2a6f97", "#999999"])
    plt.xlabel("Year"); plt.ylabel("Copies"); plt.title("Copies in other channels, by year"); plt.xticks(rotation=0)
    plt.tight_layout(); plt.savefig(out / "fig_copies_by_year.png", dpi=200); plt.close()
    print("Done.")


if __name__ == "__main__":
    main()
