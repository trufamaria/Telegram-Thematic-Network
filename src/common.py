"""Shared helpers: load the Telegram CSV exports, clean the text, Portuguese stop words."""
import re
import unicodedata
from pathlib import Path

import pandas as pd

URL_RE = re.compile(r"https?://\S+|www\.\S+")
HASHTAG_RE = re.compile(r"#\w+")
MENTION_RE = re.compile(r"@\w+")

# Portuguese stop words (no external download needed) plus words that carry no theme
# in Telegram channels (calls to share, link shorteners, and so on).
STOPWORDS = set("""
a à ao aos aquela aquelas aquele aqueles aquilo as até com como da das de dela delas dele deles depois do dos e é ela elas ele eles em entre era eram essa essas esse esses esta estas este estes eu foi foram há isso isto já la lhe lhes mais mas me mesmo meu meus minha minhas muito na nas nem no nos nós nossa nossas nosso nossos num numa não o os ou para pela pelas pelo pelos por qual quando que quem se sem ser seu seus só sua suas também te tem têm tinha tu tua tuas um uma você vocês vos vai vão ainda onde sobre ser são ter está estão estava será serão pode podem deve devem fazer fez feito faz sendo ficar contra cada todo toda todos todas outro outra outros outras ano anos dia dias hoje agora aqui ali bem mal sim novo nova novos novas segundo primeira primeiro nesta neste nessa nesse dessa desse deste desta pois porque então assim ainda sempre nunca apenas além antes durante após desde sob sobre tanto tão quanto quanta quantos quantas
compartilhe compartilhar inscreva inscrevase canal link links acesse acessar clique confira veja assista assistir aovivo ao vivo video vídeo vídeos live youtube youtu instagram twitter tiktok telegram whatsapp bit ly bityli https http www com br gov via junte participe grupo grupos siga sigam siga-nos segue seguir
""".split())


def normalize(text: str) -> str:
    """Lowercase and remove accents."""
    text = unicodedata.normalize("NFD", str(text).lower())
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")


STOPWORDS_NORM = {normalize(w) for w in STOPWORDS}


def clean_for_topics(text: str) -> str:
    """Remove links, mentions, and punctuation; keep words (hashtags become plain words)."""
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = text.replace("#", " ")
    text = normalize(text)
    text = re.sub(r"[^a-z\s]", " ", text)  # letters only (drops numbers and emojis)
    tokens = [t for t in text.split() if len(t) > 2 and t not in STOPWORDS_NORM]
    return " ".join(tokens)


def load_messages(folder: str) -> pd.DataFrame:
    """Read every CSV in `folder`, drop duplicated messages (the same channel exported
    twice), and keep messages that have text."""
    frames = []
    for path in sorted(Path(folder).glob("*.csv")):
        df = pd.read_csv(path)
        df["file"] = path.stem
        frames.append(df)
    # Largest files first, so a channel that appears twice keeps the larger export's label
    frames.sort(key=len, reverse=True)
    data = pd.concat(frames, ignore_index=True)
    data = data.drop_duplicates(subset=["channel_id", "message_id"]).copy()

    data["text"] = data["text"].fillna("").astype(str)
    data = data[data["text"].str.strip() != ""].copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce", utc=True)
    data = data.dropna(subset=["date"])
    data["channel"] = data["file"]
    data["year"] = data["date"].dt.year
    data["text_clean"] = data["text"].apply(clean_for_topics)
    data["n_tokens"] = data["text_clean"].str.split().str.len()
    return data.sort_values("date").reset_index(drop=True)
