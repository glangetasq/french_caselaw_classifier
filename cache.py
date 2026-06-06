from pathlib import Path

import joblib
import numpy as np
import pandas as pd

CACHE_DIR = Path("cache").resolve()


def load_or_run(
    filename: str,
    fn,
    index: pd.Index | None = None,
    force: bool = False,
):
    """
    Load an artifact from CACHE_DIR/<filename> if it exists, otherwise call fn() and save.

    filename : includes the extension — .parquet for DataFrames, .pkl for anything else,
               .npz for numpy embedding arrays.
    index    : only required for .npz files. The DataFrame index used when the embeddings
               were saved; rows are reordered on load to match it, guarding against
               shuffles between runs.
    force    : if True, ignore any existing cache and recompute.
    """
    path = CACHE_DIR / filename

    if path.exists() and not force:
        if path.suffix == ".parquet":
            result = pd.read_parquet(path)
        elif path.suffix == ".npz":
            import embedding as _emb
            result = _emb.load_embeddings(path, index)
        else:
            result = joblib.load(path)
        print(f"[cache] loaded {path}")
        return result

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    result = fn()
    if path.suffix == ".parquet":
        result.to_parquet(path)
    elif path.suffix == ".npz":
        import embedding as _emb
        _emb.save_embeddings(result, index, path)
    else:
        joblib.dump(result, path)
    print(f"[cache] saved {path}")
    return result
