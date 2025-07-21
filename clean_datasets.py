import json
import ast
import re
from pathlib import Path
import pandas as pd

# Unified output schema
SCHEMA = ["dataset", "split", "id", "text", "meta"]

def _to_json(obj) -> str:
    """Dump object to JSON string using UTF-8."""
    return json.dumps(obj, ensure_ascii=False)


def _bundle_meta(df: pd.DataFrame, exclude: list[str]) -> pd.Series:
    """Bundle remaining columns into JSON meta field."""
    cols = [c for c in exclude if c in df.columns]
    meta = df.drop(columns=cols)
    return meta.apply(lambda r: _to_json(r.dropna().to_dict()), axis=1)


def _add_attribute_multiindex(df: pd.DataFrame) -> pd.DataFrame:
    """Wrap attribute columns in a MultiIndex for saving."""
    attr_cols = [c for c in df.columns if c not in SCHEMA]
    if not attr_cols:
        return df
    tuples = []
    for col in df.columns:
        if col in attr_cols:
            tuples.append(("attributes", col))
        else:
            tuples.append((col, ""))
    df.columns = pd.MultiIndex.from_tuples(tuples)
    return df

# ─────────────────────────────────────────────────────────────────────────────
# Dataset-specific cleaners
# ─────────────────────────────────────────────────────────────────────────────


def clean_anthropic_persuasion(base: Path) -> pd.DataFrame:
    df = pd.read_csv(base / "Anthropic Persuasion.csv")
    df["dataset"] = "Anthropic Persuasion"
    df["split"] = "orig"
    df["id"] = df.get("Unnamed: 0", df.index).astype(str)
    df["text"] = df["argument"].fillna("")
    df["persuasiveness"] = df["rating_final"]
    # df["label"] = None
    df["persuasiveness_desc"] = (
        df["rating_final"].astype(str).str.split("-", n=1, expand=True)[1].str.strip()
    )
    df["meta"] = _bundle_meta(
        df,
        [
            "Unnamed: 0",
            "argument",
            "rating_final",
            "label_desc",
            "text",
            "persuasiveness",
        ]
        + SCHEMA,
    )
    return df[SCHEMA + ["persuasiveness"]]


def clean_convincingness(base: Path) -> pd.DataFrame:
    df = pd.read_csv(base / "UKPConvArg1Strict_all.csv")
    df["dataset"] = "Convincingness"
    df["split"] = "orig"
    df["id"] = df["id"].astype(str)
    df["text"] = df.apply(
        lambda r: _to_json({"a1": r["sentence_a1"], "a2": r["sentence_a2"]}), axis=1
    )
    df["convincingness"] = df["label"]
    # df["label_desc"] = None
    df["meta"] = _bundle_meta(
        df,
        [
            "sentence_a1",
            "sentence_a2",
            "label",
            "convincingness",
        ]
        + SCHEMA,
    )
    return df[SCHEMA + ["convincingness"]]

def clean_emobank(base: Path) -> pd.DataFrame:
    df = pd.read_csv(base / "emobank_with_reader_columns.csv")
    df["dataset"] = "EmoBank"
    df["split"] = df["split"].fillna("orig")
    df["id"] = df["id"].astype(str)
    df = df.rename(
        columns={
            "V_x": "valence_writer",
            "A_x": "arousal_writer",
            "D_x": "dominance_writer",
            "V_y": "valence_reader",
            "A_y": "arousal_reader",
            "D_y": "dominance_reader",
        }
    )
    vad_cols = [
        "valence_writer",
        "arousal_writer",
        "dominance_writer",
        "valence_reader",
        "arousal_reader",
        "dominance_reader",
    ]
    df["text"] = df["text"]
    # df["label"] = None
    # df["label_desc"] = None
    df["meta"] = _bundle_meta(df, vad_cols + ["text"] + SCHEMA)
    return df[SCHEMA + vad_cols]

def clean_formality_scores(base: Path) -> pd.DataFrame:
    df = pd.read_csv(base / "formality_scores.csv")
    df["dataset"] = "Formality Scores"
    df["split"] = df.get("group", "orig")
    df["id"] = df.get("Unnamed: 0", df.index).astype(str)
    df["text"] = df["sentence"]
    df["formality"] = df["avg_score"]
    # df["label"] = None
    # df["label_desc"] = None
    df["meta"] = _bundle_meta(
        df, ["sentence", "avg_score", "group", "Unnamed: 0", "formality"] + SCHEMA
    )
    return df[SCHEMA + ["formality"]]

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
    df["populism"] = df[label_col] if label_col else None
    # df["label"] = None
    df["populism_desc"] = None
    exclude = (
        ["merging_variable", "populism"] + ([label_col] if label_col else []) + SCHEMA
    )
    df["meta"] = _bundle_meta(df, exclude)
    return df[SCHEMA + ["populism"]]

def clean_go_emotion(base: Path) -> pd.DataFrame:
    df = pd.read_csv(base / "go_emotion.csv")
    df["dataset"] = "Go Emotion"
    df["split"] = df["split"].fillna("orig")
    df["id"] = df["id"].astype(str)
    df["text"] = df["text"]

    def parse_labels(val: str) -> list[int]:
        return [int(x) for x in re.findall(r"\d+", str(val))]

    df["_labels_list"] = df["labels"].apply(parse_labels)
    emotions = [
        "admiration",
        "amusement",
        "anger",
        "annoyance",
        "approval",
        "caring",
        "confusion",
        "curiosity",
        "desire",
        "disappointment",
        "disapproval",
        "disgust",
        "embarrassment",
        "excitement",
        "fear",
        "gratitude",
        "grief",
        "joy",
        "love",
        "nervousness",
        "optimism",
        "pride",
        "realization",
        "relief",
        "remorse",
        "sadness",
        "surprise",
        "neutral",
    ]
    for idx, emo in enumerate(emotions):
        df[emo] = df["_labels_list"].apply(lambda ls, i=idx: int(i in ls))
    # df["label"] = None
    # df["label_desc"] = None
    df["meta"] = _bundle_meta(
        df, ["labels", "_labels_list", "text"] + SCHEMA + emotions
    )
    return df[SCHEMA + emotions]


def clean_good_news(base: Path) -> pd.DataFrame:
    df = pd.read_csv(base / "gne-release-v1.0.tsv", sep="\t")
    df["dataset"] = "GoodNewsEveryone"
    df["split"] = "orig"
    df["id"] = df["id"].astype(str)
    df["text"] = df["headline"]
    df["dominant_emotion"] = df["dominant_emotion"]
    # df["label_desc"] = None
    df["meta"] = _bundle_meta(df, ["headline", "dominant_emotion"] + SCHEMA)
    return df[SCHEMA + ["dominant_emotion"]]

def clean_humicroedit(base: Path) -> pd.DataFrame:
    df = pd.read_csv(base / "humicroedit.csv")
    df["dataset"] = "Humicroedit"
    df["split"] = df.get("split", "orig")
    df["id"] = df["id"].astype(str)
    df["text"] = df["edit"]
    df["funniness"] = df["meanGrade"]
    # df["label_desc"] = None
    df["meta"] = _bundle_meta(df, ["edit", "meanGrade", "funniness"] + SCHEMA)
    return df[SCHEMA + ["funniness"]]

def clean_mbic(base: Path) -> pd.DataFrame:
    df = pd.read_excel(base / "labeled_dataset.xlsx")
    df["dataset"] = "MBIC"
    df["split"] = "orig"
    df["id"] = df.get("group_id", df.index).astype(str)
    df["text"] = df["sentence"]
    # df["label"] = None
    # df["label_desc"] = None
    label_cols = ["Label_bias", "Label_opinion"]
    df["meta"] = _bundle_meta(df, ["sentence"] + label_cols + SCHEMA)
    return df[SCHEMA + label_cols].rename(columns={"Label_bias": "bias", "Label_opinion": "opinion"})

# def clean_mbic(base: Path

def clean_mint(base: Path) -> pd.DataFrame:
    df = pd.read_csv(base / "preprocess_train.csv")
    df["dataset"] = "MINT"
    df["split"] = "train"
    df["id"] = df.index.astype(str)
    df["text"] = df["text"]
    df["intimacy"] = df["label"]
    df["intimacy_desc"] = df["emo_label"] if "emo_label" in df.columns else None
    df["meta"] = _bundle_meta(
        df, ["label", "emo_label", "text", "intimacy", "intimacy_desc"] + SCHEMA
    )
    return df[SCHEMA + ["intimacy"]]

def clean_persuade(base: Path) -> pd.DataFrame:
    df = pd.read_csv(base / "persuade_corpus_2.0_train.csv", index_col=0)
    df["dataset"] = "PERSUADE 2.0"
    df["split"] = "orig"
    df["id"] = df["essay_id_comp"].astype(str)
    df["text"] = df["full_text"]
    df["discourse_effectiveness"] = df["holistic_essay_score"]
    df["meta"] = _bundle_meta(
        df,
        [
            "full_text",
            "holistic_essay_score",
            "essay_id",
            "essay_id_comp",
            "discourse_effectiveness",
        ]
        + SCHEMA,
    )
    return df[SCHEMA + ["discourse_effectiveness"]]


def _parse_sst_line(line: str):
    label = None
    tokens = []
    i = 0
    while i < len(line):
        ch = line[i]
        if ch == "(":  # start of subtree with label
            i += 1
            num = ""
            while i < len(line) and line[i].isdigit():
                num += line[i]
                i += 1
            if label is None and num:
                label = int(num)
        elif ch == ")":
            i += 1
        else:
            j = i
            while j < len(line) and line[j] not in "()":
                j += 1
            token = line[i:j].strip()
            if token:
                tokens.append(token)
            i = j
            continue
        if i < len(line) and line[i] == " ":
            i += 1
    return label, " ".join(tokens)


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
                rows.append(
                    {
                        "dataset": "Sentiment Treebank",
                        "split": split,
                        "id": f"{split}-{idx}",
                        "text": text,
                        "sentiment_valence": label,
                        # "label_desc": None,
                        "meta": _to_json({}),
                    }
                )
    df = pd.DataFrame(rows)
    return df[SCHEMA + ["sentiment_valence"]]


def clean_unify_emotion(base: Path) -> pd.DataFrame:
    df = pd.read_csv(base / "unified-dataset.csv")
    df["dataset"] = "Unify Emotion"
    df["split"] = df["split"].fillna("orig")
    df["id"] = df["id"].astype(str)
    df["text"] = df["text"]
    df["emotions"] = df["emotions"].apply(ast.literal_eval)
    df["VAD"] = df["VAD"].apply(ast.literal_eval)

    all_emotions = sorted({k for d in df["emotions"] for k in d})
    for emo in all_emotions:
        df[emo] = df["emotions"].apply(lambda d, e=emo: d.get(e))

    df["valence"] = df["VAD"].apply(lambda d: d.get("valence"))
    df["arousal"] = df["VAD"].apply(lambda d: d.get("arousal"))
    df["dominance"] = df["VAD"].apply(lambda d: d.get("dominance"))

    # df["label"] = None
    # df["label_desc"] = None
    extra_cols = all_emotions + ["valence", "arousal", "dominance"]
    df["meta"] = _bundle_meta(
        df,
        [
            "text",
            "emotions",
            "VAD",
            "source",
            "emotion_model",
            "domain",
            "labeled",
            "annotation_procedure",
        ]
        + SCHEMA
        + extra_cols,
    )
    return df[SCHEMA + extra_cols]


def clean_wassa(base: Path) -> pd.DataFrame:
    df = pd.read_csv(base / "emotion_intensity_all.tsv", sep="\t")
    df["dataset"] = "WASSA"
    df["split"] = df["split"].fillna("orig")
    df["id"] = df["tweet_id"].astype(str)
    df["text"] = df["tweet"]

    wide = (
        df.pivot_table(
            index=["dataset", "split", "id", "text"],
            columns="emotion",
            values="gold_label",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )

    emotions = ["anger", "fear", "joy", "sadness"]
    for emo in emotions:
        if emo not in wide.columns:
            wide[emo] = None

    # wide["label"] = None
    # wide["label_desc"] = None
    wide["meta"] = _bundle_meta(wide, emotions + SCHEMA)
    return wide[SCHEMA + emotions]


def clean_politeness(base: Path) -> pd.DataFrame:
    path = base / "utterances.jsonl"
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            meta = obj.get("meta", {})
            rows.append(
                {
                    "dataset": "politeness_corpus",
                    "split": "orig",
                    "id": obj.get("id"),
                    "text": obj.get("text", ""),
                    "politeness": meta.get("Normalized Score"),
                    "politeness_desc": meta.get("Binary"),
                    "meta": _to_json(
                        {
                            k: v
                            for k, v in obj.items()
                            if k not in {"id", "text", "meta"}
                        }
                    ),
                }
            )
    df = pd.DataFrame(rows)
    return df[SCHEMA + ["politeness"]]


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
            df_to_save = _add_attribute_multiindex(df.copy())
            df_to_save.to_csv(d / "clean.csv", index=False)
            print(f"Saved {len(df)} rows to {d/'clean.csv'}")
        except Exception as e:
            print(f"Failed to clean {name}: {e}")


if __name__ == "__main__":
    main()
