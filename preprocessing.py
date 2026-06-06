import uuid

import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path
from sklearn.model_selection import train_test_split

# Each label maps to a list of substrings matched case-insensitively against the chamber name.
# "correct" catches both "correctionnelle" and abbreviated "Correct." forms.
_HF_PATTERNS: dict[str, list[str]] = {
    "soc":  ["soc"],
    "civ":  ["civ"],
    "crim": ["crim", "correct", "mineur", "tpe", "detention"],
    "com":  ["com"],
}


def load_data(
    drop_uncategorized: bool = True,
    court_types: list[str] | None = None,
) -> pd.DataFrame:
    from datasets import load_dataset

    if court_types is not None:
        # Passing a list to split= returns a list[Dataset] in the same order,
        # avoiding downloading unused splits entirely.
        datasets_list = load_dataset("antoinejeannot/jurisprudence", split=court_types)
        ds = dict(zip(court_types, datasets_list))
    else:
        ds = load_dataset("antoinejeannot/jurisprudence")

    frames = []
    for court_type, split in ds.items():
        df = split.to_pandas()[["text", "chamber"]]
        chamber_lc = df["chamber"].str.lower()

        flags = pd.DataFrame({
            label: chamber_lc.str.contains("|".join(patterns), na=False)
            for label, patterns in _HF_PATTERNS.items()
        })
        n_matches = flags.sum(axis=1)
        single_match = n_matches == 1

        if drop_uncategorized:
            mask = single_match
        else:
            mask = pd.Series(True, index=df.index)

        df = df[mask].copy()
        df["label"] = flags[single_match].idxmax(axis=1)
        df["court_type"] = court_type
        frames.append(df)

    result = pd.concat(frames, ignore_index=True)
    result.index = [str(uuid.uuid4()) for _ in range(len(result))]
    return result


def parse_html(text: pd.Series) -> pd.Series:
    def _strip(val: str) -> str:
        if not isinstance(val, str):
            return ""
        return " ".join(BeautifulSoup(val, "html.parser").get_text().split())

    return text.map(_strip)


def stratified_split(
    df: pd.DataFrame,
    class_col: str,
    test_pct: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train, test = train_test_split(
        df,
        test_size=test_pct,
        stratify=df[class_col],
        random_state=random_state,
    )
    return train, test


def load_word_list(path: str | Path) -> list[str]:
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


