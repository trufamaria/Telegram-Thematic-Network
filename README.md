# Themes and Reposts in Four Telegram Channels

Python code that finds common themes in Telegram posts, checks whether the same messages are reposted across channels, and maps the reposting as a network.

I used it on public channels of four Brazilian right-wing political figures (Jair, Flávio, Eduardo, and Michelle Bolsonaro), January 2021 to October 2026. The code works with any Telegram export in the same format.

**Status:** work in progress. The themes are candidate themes from a topic model. I read example messages to name them, and the names are mine.

## What the code does

| Step | Script | Output |
|------|--------|--------|
| 1. Themes | `src/01_themes.py` | Topic model (TF-IDF and NMF). 16 themes, each with top words, example messages, and its share in each channel and year. |
| 2. Reposts | `src/02_reposts.py` | Groups of near-identical messages, with the channel and time of the first post. Also links shared across channels. |
| 3. Network | `src/03_network.py` | Directed network of who copies whom, network measures, timing of copies, three figures, and a `.gexf` file for Gephi. |

## Data format

One CSV per channel in a folder. Each file needs these columns: `message_id`, `date`, `text`, `views`, `channel_id`. The file name becomes the channel label.

If two files hold the same channel (same `channel_id`), the script keeps each message once.

The raw data and message-level results are not in this repository. Both contain the full text of posts. The `.gitignore` file keeps them out.

## How to run

```
pip install -r requirements.txt
bash run_all.sh path/to/folder_with_csvs
```

Run time is about one minute. Results go in `results/`.

## Method

**Themes.** Links, mentions, accents, numbers, and Portuguese stop words are removed. Words are weighted with TF-IDF (single words and pairs). A non-negative matrix factorization (NMF) model finds 12 sets of words that occur together. Each message goes to its strongest theme. One theme held 30% of the messages (general political rhetoric), so a second model split it into five sub-themes. Coherence was highest and nearly equal at 10 and 12 themes (about 0.37 for both). I used 12 because it separated the government topics more clearly. Run `python src/01_themes.py DATA results auto` to test 6 to 18 themes. Messages with fewer than six words after cleaning are left out (16,522 of 20,839 messages are used).

**Reposts.** Messages with at least 40 characters are compared with character n-gram TF-IDF vectors and cosine similarity. Two messages count as the same message when similarity is at least 0.95 and they were posted within 7 days of each other. The time window stops recurring templates (daily vaccine counts, "watch the live") from being joined across years. Digits are kept, so two updates with different numbers are not treated as one message. The earliest message in each group is the origin, and later messages are copies.

**Network.** Each channel is a node. An edge from A to B means B posted a copy of a message that A had posted first. Edge weight is the number of copies.

## Main results

- **Reposting is common and mostly one way.** 765 messages are copies of a message first posted in another channel. 676 of them (88%) are Jair → Flávio. About 7% of the messages in Flávio's channel are copies of Jair's.
- **Copies are fast.** The median time between the first post and the copy is 1.7 hours. 96% appear within 24 hours.
- **Most reposting is from 2021 and 2022.** 93% of copies fall in those two years.
- **The result holds when the similarity setting changes.** Between 695 and 911 cross-channel copies are found at similarity values from 0.99 down to 0.80 (`results/repost_sensitivity.csv`).
- **Reposted content is government communication.** About 85% of the copies that have a theme belong to themes about ministries, vaccines, works, and social programs. Messages attacking Lula or promoting the 2026 campaign are almost never in a repost group (under 1% of each of those themes).
- **The channels differ.** Government-communication themes make up 78.5% of Jair's messages, 39.4% of Flávio's, 30.2% of Michelle's, and 18.7% of Eduardo's. Eduardo's and Michelle's channels lean on anti-left rhetoric, free speech, and Moraes.
- **The themes change over time.** Government-communication themes drop from 72% of messages in 2021 to 6% in 2026. In 2026 the "path to prosperity" campaign theme is about 32% of the messages.

## Limits

- "Posted first" means first among these four channels. A message can come from a source outside them, such as a ministry, so the direction shows who posted first in this data, not who wrote the message.
- Jair's channel ends in July 2025 and Michelle's in April 2025, so later years have fewer channels.
- The topic model joins messages by shared words. Some messages sit in a theme only because of one word (for example, "Rio" in the state-by-state theme). Read the examples in `results/themes.csv` before you cite a theme.
- Only messages with text are used. Photos, videos, and deleted posts are not counted.

## Results files

- `results/themes.csv`: top words, size, and example messages for each theme
- `results/themes_by_channel_pct.csv` and `themes_by_year_pct.csv`: theme shares
- `results/repost_summary.csv`, `repost_sensitivity.csv`: how many messages are reposts
- `results/network_edges.csv`, `network_nodes.csv`, `network_by_period.csv`: the network
- `results/repost_network.gexf`: the network for Gephi
- `results/fig_*.png`: figures

## Author

Marina Petric, Ph.D., Media and Communication, Texas Tech University
[marinapetric.com](https://marinapetric.com)
