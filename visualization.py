import math

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer

# Plotly D3 palette — one color per legal domain
LABEL_COLORS = {
    "civ":  "#1f77b4",  # blue
    "soc":  "#2ca02c",  # green
    "com":  "#ff7f0e",  # orange
    "crim": "#d62728",  # red
}


def _draw_imbalance(ax: plt.Axes, labels: pd.Series, subtitle: str) -> None:
    counts = labels.value_counts().sort_values(ascending=False)
    classes = counts.index.tolist()
    values = counts.values.tolist()
    min_count = min(values)
    max_count = max(values)
    palette = [LABEL_COLORS.get(cls, "#7f7f7f") for cls in classes]

    bars = ax.bar(range(len(classes)), values, color=palette, width=0.6)

    for bar, val in zip(bars, values):
        cx = bar.get_x() + bar.get_width() / 2
        h = bar.get_height()
        is_short = h / max_count < 0.15

        ax.text(
            cx,
            h / 2 if not is_short else h + max_count * 0.01,
            f"{val:,}",
            ha="center",
            va="center" if not is_short else "bottom",
            fontsize=13, fontweight="bold",
            color="white" if not is_short else "#333333",
        )

        if val != min_count:
            ratio = round(val / min_count)
            ratio_y = h + max_count * (0.07 if is_short else 0.015)
            ax.text(
                cx, ratio_y,
                f"{ratio:,}x min",
                ha="center", va="bottom",
                fontsize=10, color="#666666",
            )

    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(subtitle, fontsize=12, fontweight="bold")
    ax.set_ylim(0, max_count * 1.18)

    patches = [mpatches.Patch(color=LABEL_COLORS.get(cls, "#7f7f7f"), label=cls) for cls in classes]
    ax.legend(handles=patches, title="Label", frameon=True, loc="upper right")


def plot_class_imbalance(labels: pd.Series, ood_labels: pd.Series | None = None) -> None:
    if ood_labels is None:
        fig, ax = plt.subplots(figsize=(max(7, labels.nunique() * 2), 6))
        _draw_imbalance(ax, labels, "In Sample")
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(14, labels.nunique() * 4), 6))
        _draw_imbalance(ax1, labels, "In Sample")
        _draw_imbalance(ax2, ood_labels, "Out of Distribution")

    fig.suptitle("Class Distribution", fontsize=14, fontweight="bold")
    sns.despine()
    plt.tight_layout()
    plt.show()


def plot_length_distributions(
    word_count: pd.Series,
    char_count: pd.Series,
    groupby: pd.Series | None = None,
) -> None:
    if groupby is None:
        classes = ["all"]
        masks   = {"all": pd.Series(True, index=word_count.index)}
    else:
        class_counts = groupby.value_counts()
        classes      = class_counts.index.tolist()
        masks        = {cls: groupby == cls for cls in classes}

    n_rows = len(classes)
    x_lim  = {
        "word": word_count.quantile(0.99),
        "char": char_count.quantile(0.99),
    }
    series = {"word": word_count, "char": char_count}
    labels = {"word": "word count", "char": "char count"}

    fig, axes = plt.subplots(
        n_rows, 2,
        figsize=(10, 3 * n_rows),
        sharey="row",
        squeeze=False,
    )

    for i, cls in enumerate(classes):
        color = LABEL_COLORS.get(cls, "#7f7f7f") if cls != "all" else "#7f7f7f"
        mask  = masks[cls]
        for j, (key, s) in enumerate(series.items()):
            ax   = axes[i][j]
            data = s[mask]
            clip = (0, x_lim[key])
            sns.histplot(
                data, ax=ax, color=color, stat="density", kde=True,
                kde_kws={"clip": clip, "bw_adjust": 0.5},
                line_kws={"linewidth": 2},
            )
            ax.set_xlim(0, x_lim[key])
            ax.set_xlabel(labels[key])
            ax.set_ylabel("Density" if j == 0 else "")
            ax.set_title(f"{cls} — {labels[key]}", fontsize=11, fontweight="bold")
            ax.tick_params(labelbottom=True)
            ax.xaxis.set_major_formatter(
                mticker.FuncFormatter(lambda x, _, k=key: f"{x/1000:.0f}k" if (k == "char" and x >= 1000) else f"{x:.0f}")
            )

    sns.despine()
    plt.tight_layout()
    plt.show()


def top_tfidf_per_class(
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
    vec: TfidfVectorizer,
    n: int = 20,
) -> pd.DataFrame:
    cols = {}
    for cls, group in df.groupby(label_col):
        cls_vec = clone(vec).fit(group[text_col])
        terms = np.array(cls_vec.get_feature_names_out())
        mean_scores = np.asarray(cls_vec.transform(group[text_col]).mean(axis=0)).flatten()
        top_idx = mean_scores.argsort()[::-1][:n]
        cols[cls] = terms[top_idx]
    return pd.DataFrame(cols)


def top_nb_features(pipeline, n: int = 20) -> pd.DataFrame:
    """Top features per class from a fitted NB pipeline (by log-probability)."""
    vec   = pipeline["tfidf"]
    clf   = pipeline["clf"]
    terms = vec.get_feature_names_out()
    return pd.DataFrame({
        cls: terms[clf.feature_log_prob_[i].argsort()[::-1][:n]]
        for i, cls in enumerate(clf.classes_)
    })


def top_logreg_features(pipeline, n: int = 20) -> pd.DataFrame:
    """Top features per class from a fitted LogReg pipeline (by coefficient)."""
    vec   = pipeline["tfidf"]
    clf   = pipeline["clf"]
    terms = vec.get_feature_names_out()
    return pd.DataFrame({
        cls: terms[clf.coef_[i].argsort()[::-1][:n]]
        for i, cls in enumerate(clf.classes_)
    })


def summary_table(**results):
    """keyword arguments: model_name=EvalMulticlassResult"""
    def _extract(result):
        m     = result.metrics
        ood_m = result.ood_metrics
        f_col = next((c for c in m.columns if c.startswith("f")), "f1")
        class_rows = [r for r in m.index if r != "macro"]
        label = f_col.upper()
        return {
            f"Macro {label}": m.loc["macro", f_col],
            f"Min {label}":   m.loc[class_rows, f_col].min(),
            f"OOD {label}":   ood_m.loc["macro", f_col] if ood_m is not None else np.nan,
            "PR-AUC":         m.loc["macro", "pr_auc"] if "pr_auc" in m.columns else np.nan,
            "Inference ms":   result.inf_time,
        }

    rows = {name: _extract(result) for name, result in results.items()}
    df = pd.DataFrame(rows).T

    def _highlight(s):
        valid = s.dropna()
        if valid.empty:
            return [""] * len(s)
        tol = 0.001
        if s.name == "Inference ms":
            best_val = valid.min()
            is_best  = s.apply(lambda v: pd.notna(v) and abs(v - best_val) <= tol)
        else:
            best_val = valid.max()
            is_best  = s.apply(lambda v: pd.notna(v) and abs(v - best_val) <= tol)
        return ["background-color: #ffd700; font-weight: bold" if is_best[i] else "" for i in s.index]

    f_cols = [c for c in df.columns if c.startswith("Macro") or c.startswith("Min") or c.startswith("OOD")]
    fmt = {c: "{:.1%}" for c in f_cols}
    fmt["PR-AUC"] = "{:.1%}"
    fmt["Inference ms"] = "{:.1f} ms"

    return df.style.apply(_highlight, axis=0).format(fmt, na_rep="—")


def ngram_comparison_table(results: dict[tuple[int, int], "EvalMulticlassResult"]):
    """Compare char TF-IDF n-gram ranges. Rows sorted by ngram_range."""
    from evaluation import _TBL_STYLES

    rows = {}
    for ngram_range, result in sorted(results.items()):
        f_col = f"f{result.beta:g}"
        oos = result.metrics.loc["macro", f_col]
        ood = result.ood_metrics.loc["macro", f_col] if result.ood_metrics is not None else float("nan")
        try:
            vocab = len(result.model[0].vocabulary_)
        except Exception:
            vocab = float("nan")
        rows[str(ngram_range)] = {
            "OOS F2": oos,
            "OOD F2": ood,
            "Δ (OOD−OOS)": ood - oos,
            "Vocab": vocab,
        }

    df = pd.DataFrame(rows).T
    best_ood_row = df["OOD F2"].idxmax()
    _GOLD = "background-color: #ffd700; font-weight: bold; color: #333"

    def _row_style(row):
        if row.name != best_ood_row:
            return [""] * len(row)
        out = ["background-color: #f0f4ff"] * len(row)
        out[list(row.index).index("OOD F2")] = _GOLD
        return out

    def _col_highlight(col):
        if col.name not in ("OOS F2", "Δ (OOD−OOS)"):
            return [""] * len(col)
        best = col.idxmax()
        return [_GOLD if i == best else "" for i in col.index]

    fmt = {"OOS F2": "{:.1%}", "OOD F2": "{:.1%}", "Δ (OOD−OOS)": "{:+.1%}", "Vocab": "{:,.0f}"}
    return (
        df.style
        .apply(_row_style, axis=1)
        .apply(_col_highlight, axis=0)
        .format(fmt, na_rep="—")
        .set_caption("Char TF-IDF — n-gram range comparison")
        .set_table_styles(_TBL_STYLES)
    )


def stratified_split_table(
    df: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    label_col: str = "label",
) -> pd.DataFrame:
    def _density(d):
        counts = d[label_col].value_counts().sort_index()
        return counts / counts.sum()

    tbl = pd.concat([_density(df).rename("all"), _density(train).rename("train"), _density(test).rename("test")], axis=1)
    return tbl.map(lambda v: f"{v:.2%}")
