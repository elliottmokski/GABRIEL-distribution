import json
from pathlib import Path
import pandas as pd

# Unified output schema
SCHEMA = ["dataset", "split", "id", "text", "label", "label_desc", "meta"]


def _to_json(obj) -> str:
    """Dump object to JSON string using UTF-8."""
    return json.dumps(obj, ensure_ascii=False)


def _bundle_meta(df: pd.DataFrame, exclude: list[str]) -> pd.Series:
    """Bundle remaining columns into JSON meta field."""
    cols = [c for c in exclude if c in df.columns]
    meta = df.drop(columns=cols)
    return meta.apply(lambda r: _to_json(r.dropna().to_dict()), axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset-specific cleaners
# ─────────────────────────────────────────────────────────────────────────────

def clean_anthropic_persuasion(base: Path) -> pd.DataFrame:
    df = pd.read_csv(base / "Anthropic Persuasion.csv")
    df["dataset"] = "Anthropic Persuasion"
    df["split"] = "orig"
    df["id"] = df.get("Unnamed: 0", df.index).astype(str)
    df["text"] = df["argument"].fillna("")
    df["label"] = df["rating_final"]
    df["label_desc"] = (
        df["rating_final"].astype(str).str.split("-", n=1, expand=True)[1].str.strip()
    )
    df["meta"] = _bundle_meta(df, ["Unnamed: 0", "argument", "rating_final", "label_desc", "text"] + SCHEMA)
    return df[SCHEMA]

def clean_convincingness(base: Path) -> pd.DataFrame:
    df = pd.read_csv(base / "UKPConvArg1Strict_all.csv")
    df["dataset"] = "Convincingness"
    df["split"] = "orig"
    df["id"] = df["id"].astype(str)
    df["text"] = df.apply(lambda r: _to_json({"a1": r["sentence_a1"], "a2": r["sentence_a2"]}), axis=1)
    df["label"] = df["label"]
    df["label_desc"] = None
    df["meta"] = _bundle_meta(df, ["sentence_a1", "sentence_a2", "label"] + SCHEMA)
    return df[SCHEMA]

def clean_emobank(base: Path) -> pd.DataFrame:
    df = pd.read_csv(base / "emobank_with_reader_columns.csv")
    df["dataset"] = "EmoBank"
    df["split"] = df["split"].fillna("orig")
    df["id"] = df["id"].astype(str)
    vad_cols = ["V_x", "A_x", "D_x", "V_y", "A_y", "D_y"]
    df["text"] = df["text"]
    df["label"] = df[vad_cols].apply(lambda r: _to_json(r.to_dict()), axis=1)
    df["label_desc"] = None
    df["meta"] = _bundle_meta(df, vad_cols + ["text"] + SCHEMA)
    return df[SCHEMA]

def clean_formality_scores(base: Path) -> pd.DataFrame:
    df = pd.read_csv(base / "formality_scores.csv")
    df["dataset"] = "Formality Scores"
    df["split"] = df.get("group", "orig")
    df["id"] = df.get("Unnamed: 0", df.index).astype(str)
    df["text"] = df["sentence"]
    df["label"] = df["avg_score"]
    df["label_desc"] = None
    df["meta"] = _bundle_meta(df, ["sentence", "avg_score", "group", "Unnamed: 0"] + SCHEMA)
    return df[SCHEMA]

def clean_global_populism(base: Path) -> pd.DataFrame:
    csv_path = base / "gpd_v2_20220427.csv"
    if not csv_path.exists():
        csv_path = base / "GPD_20190625.csv"
    df = pd.read_csv(csv_path)
    text_dir = base / "speeches_20220427_unzipped" / "speeches_20220427"
    if not text_dir.exists():
        zip_path = base / "speeches_20220427" / "speeches_20220427.zip"
        if zip_path.exists():
            import zipfile
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(base / "speeches_20220427_unzipped")
            text_dir = base / "speeches_20220427_unzipped" / "speeches_20220427"
    def read_text(fn: str) -> str:
        fp = text_dir / fn
        if fp.exists():
            return fp.read_text(encoding="utf-8", errors="ignore")
        return ""
    df["dataset"] = "Global Populism"
    df["split"] = "orig"
    df["id"] = df.index.astype(str)
    if "merging_variable" in df.columns:
        df["text"] = df["merging_variable"].astype(str).apply(read_text)
    else:
        df["text"] = df["speechtype"].astype(str)
    label_col = "rubricgrade" if "rubricgrade" in df.columns else None
    df["label"] = df[label_col] if label_col else None
    df["label_desc"] = None
    exclude = ["merging_variable"] + ([label_col] if label_col else []) + SCHEMA
    df["meta"] = _bundle_meta(df, exclude)
    return df[SCHEMA]

def clean_go_emotion(base: Path) -> pd.DataFrame:
    df = pd.read_csv(base / "go_emotion.csv")
    df["dataset"] = "Go Emotion"
    df["split"] = df["split"].fillna("orig")
    df["id"] = df["id"].astype(str)
    df["text"] = df["text"]
    df["label"] = df["labels"]
    df["label_desc"] = None
    df["meta"] = _bundle_meta(df, ["labels", "text"] + SCHEMA)
    return df[SCHEMA]

def clean_good_news(base: Path) -> pd.DataFrame:
    df = pd.read_csv(base / "gne-release-v1.0.tsv", sep="\t")
    df["dataset"] = "GoodNewsEveryone"
    df["split"] = "orig"
    df["id"] = df["id"].astype(str)
    df["text"] = df["headline"]
    df["label"] = df["dominant_emotion"]
    df["label_desc"] = None
    df["meta"] = _bundle_meta(df, ["headline", "dominant_emotion"] + SCHEMA)
    return df[SCHEMA]

def clean_humicroedit(base: Path) -> pd.DataFrame:
    df = pd.read_csv(base / "humicroedit.csv")
    df["dataset"] = "Humicroedit"
    df["split"] = df.get("split", "orig")
    df["id"] = df["id"].astype(str)
    df["text"] = df["edit"]
    df["label"] = df["meanGrade"]
    df["label_desc"] = None
    df["meta"] = _bundle_meta(df, ["edit", "meanGrade"] + SCHEMA)
    return df[SCHEMA]

def clean_mbic(base: Path) -> pd.DataFrame:
    df = pd.read_excel(base / "labeled_dataset.xlsx")
    df["dataset"] = "MBIC"
    df["split"] = "orig"
    df["id"] = df.get("group_id", df.index).astype(str)
    df["text"] = df["sentence"]
    df["label"] = df[["Label_bias", "Label_opinion"]].apply(lambda r: _to_json(r.to_dict()), axis=1)
    df["label_desc"] = None
    df["meta"] = _bundle_meta(df, ["sentence", "Label_bias", "Label_opinion"] + SCHEMA)
    return df[SCHEMA]

def clean_mint(base: Path) -> pd.DataFrame:
    df = pd.read_csv(base / "preprocess_train.csv")
    df["dataset"] = "MINT"
    df["split"] = "train"
    df["id"] = df.index.astype(str)
    df["text"] = df["text"]
    df["label"] = df["label"]
    df["label_desc"] = df["emo_label"] if "emo_label" in df.columns else None
    df["meta"] = _bundle_meta(df, ["label", "emo_label", "text"] + SCHEMA)
    return df[SCHEMA]

def clean_persuade(base: Path) -> pd.DataFrame:
    df = pd.read_csv(base / "persuade_corpus_2.0_sample.csv")
    df["dataset"] = "PERSUADE 2.0"
    df["split"] = "orig"
    df["id"] = df["essay_id"].astype(str)
    df["text"] = df["full_text"]
    df["label"] = df["holistic_essay_score"]
    df["label_desc"] = df.get("discourse_effectiveness")
    df["meta"] = _bundle_meta(df, ["full_text", "holistic_essay_score", "essay_id", "discourse_effectiveness"] + SCHEMA)
    return df[SCHEMA]

def _parse_sst_line(line: str):
    label = None
    tokens = []
    i = 0
    while i < len(line):
        ch = line[i]
        if ch == '(':  # start of subtree with label
            i += 1
            num = ''
            while i < len(line) and line[i].isdigit():
                num += line[i]
                i += 1
            if label is None and num:
                label = int(num)
        elif ch == ')':
            i += 1
        else:
            j = i
            while j < len(line) and line[j] not in '()':
                j += 1
            token = line[i:j].strip()
            if token:
                tokens.append(token)
            i = j
            continue
        if i < len(line) and line[i] == ' ':
            i += 1
    return label, ' '.join(tokens)

def clean_sst(base: Path) -> pd.DataFrame:
    rows = []
    for split in ["train", "dev", "test"]:
        path = base / f"{split}.txt"
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                label, text = _parse_sst_line(line)
                rows.append({
                    "dataset": "Sentiment Treebank",
                    "split": split,
                    "id": f"{split}-{idx}",
                    "text": text,
                    "label": label,
                    "label_desc": None,
                    "meta": _to_json({})
                })
    return pd.DataFrame(rows, columns=SCHEMA)

def clean_unify_emotion(base: Path) -> pd.DataFrame:
    df = pd.read_csv(base / "unified-dataset-sample.csv")
    df["dataset"] = "Unify Emotion"
    df["split"] = df["split"].fillna("orig")
    df["id"] = df["id"].astype(str)
    df["text"] = df["text"]
    df["label"] = df["emotions"]
    df["label_desc"] = None
    df["meta"] = _bundle_meta(df, ["text", "emotions"] + SCHEMA)
    return df[SCHEMA]

def clean_wassa(base: Path) -> pd.DataFrame:
    df = pd.read_csv(base / "emotion_intensity_all.tsv", sep="\t")
    df["dataset"] = "WASSA"
    df["split"] = df["split"].fillna("orig")
    df["id"] = df["tweet_id"].astype(str)
    df["text"] = df["tweet"]
    df["label"] = df["gold_label"]
    df["label_desc"] = df["emotion"]
    df["meta"] = _bundle_meta(df, ["tweet", "gold_label", "emotion"] + SCHEMA)
    return df[SCHEMA]

def clean_politeness(base: Path) -> pd.DataFrame:
    path = base / "utterances.jsonl"
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            meta = obj.get("meta", {})
            rows.append({
                "dataset": "politeness_corpus",
                "split": "orig",
                "id": obj.get("id"),
                "text": obj.get("text", ""),
                "label": meta.get("Normalized Score"),
                "label_desc": meta.get("Binary"),
                "meta": _to_json({k: v for k, v in obj.items() if k not in {"id", "text", "meta"}}),
            })
    return pd.DataFrame(rows, columns=SCHEMA)

# Mapping from folder name to cleaner
CLEANERS = {
    "Anthropic Persuasion": clean_anthropic_persuasion,
    "Convincingness": clean_convincingness,
    "EmoBank": clean_emobank,
    "Formality Scores": clean_formality_scores,
    "Global Populism": clean_global_populism,
    "Go Emotion": clean_go_emotion,
    "GoodNewsEveryone": clean_good_news,
    "Humicroedit": clean_humicroedit,
    "MBIC": clean_mbic,
    "MINT": clean_mint,
    "PERSUADE 2.0": clean_persuade,
    "Sentiment Treebank": clean_sst,
    "Unify Emotion": clean_unify_emotion,
    "WASSA": clean_wassa,
    "politeness_corpus": clean_politeness,
}

def main() -> None:
    base = Path("Datasets")
    for name, func in CLEANERS.items():
        d = base / name
        if not d.exists():
            continue
        print(f"Cleaning {name}...")
        try:
            df = func(d)
            df.to_csv(d / "clean.csv", index=False)
            print(f"Saved {len(df)} rows to {d/'clean.csv'}")
        except Exception as e:
            print(f"Failed to clean {name}: {e}")

if __name__ == "__main__":
    main()
