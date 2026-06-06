import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    fbeta_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import label_binarize

@dataclass
class EvalMulticlassResult:
    beta:                 float
    metrics:              pd.DataFrame
    confusion_matrix:     np.ndarray
    ood_metrics:          pd.DataFrame | None
    ood_confusion_matrix: np.ndarray | None
    inf_time:             float
    model:                object = None

    def _repr_html_(self) -> str:
        return _build_html(
            self.metrics, self.confusion_matrix,
            self.ood_metrics, self.ood_confusion_matrix,
            labels=[r for r in self.metrics.index if r != "macro"],
            f_col=f"f{self.beta:g}",
        )


def _compute_metrics(y_true, y_pred, y_score, labels, beta, f_col):
    p  = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    r  = recall_score(   y_true, y_pred, labels=labels, average=None, zero_division=0)
    fb = fbeta_score(    y_true, y_pred, beta=beta, labels=labels, average=None, zero_division=0)

    rows = {cls: {"precision": p[i], "recall": r[i], f_col: fb[i]} for i, cls in enumerate(labels)}
    rows["macro"] = {"precision": np.mean(p), "recall": np.mean(r), f_col: np.mean(fb)}

    if y_score is not None:
        y_bin = label_binarize(y_true, classes=labels)
        for i, cls in enumerate(labels):
            rows[cls]["pr_auc"] = average_precision_score(y_bin[:, i], y_score[:, i])
        rows["macro"]["pr_auc"] = np.mean([rows[cls]["pr_auc"] for cls in labels])

    return pd.DataFrame(rows).T.round(3), confusion_matrix(y_true, y_pred, labels=labels)


def eval_multiclass(
    model,
    test_df: pd.DataFrame,
    ood_df: pd.DataFrame | None = None,
    text_col: str = "clean_text",
    label_col: str = "label",
    X=None,
    X_ood=None,
    beta: float = 1.0,
    # BERT-specific kwargs for inference timing; if provided, time_inference_bert is used
    tokenizer=None,
    encoder=None,
) -> EvalMulticlassResult:
    f_col = f"f{beta:g}"  # "f1", "f2", "f0.5", etc.

    X      = test_df[text_col] if X is None else X
    y_true = test_df[label_col]
    labels = sorted(model.classes_.tolist())

    y_pred  = model.predict(X)
    y_score = model.predict_proba(X) if hasattr(model, "predict_proba") else None

    metrics, cm = _compute_metrics(y_true, y_pred, y_score, labels, beta, f_col)

    # OOD metrics — real held-out data from different court types
    ood_metrics, ood_cm = None, None
    if ood_df is not None or X_ood is not None:
        X_ood_      = ood_df[text_col] if X_ood is None else X_ood
        y_ood       = ood_df[label_col]
        y_pred_ood  = model.predict(X_ood_)
        y_score_ood = model.predict_proba(X_ood_) if y_score is not None else None
        ood_metrics, ood_cm = _compute_metrics(y_ood, y_pred_ood, y_score_ood, labels, beta, f_col)

    if tokenizer is not None and encoder is not None:
        inf_time = time_inference_bert(tokenizer, encoder, model, test_df[text_col])
    else:
        inf_time = time_inference(model, test_df[text_col])

    return EvalMulticlassResult(
        beta=beta,
        metrics=metrics,
        confusion_matrix=cm,
        ood_metrics=ood_metrics,
        ood_confusion_matrix=ood_cm,
        inf_time=inf_time,
        model=model,
    )


def time_inference(model, texts: pd.Series, n_samples: int = 100) -> float:
    """Median per-document inference time in ms for sklearn Pipeline / Dummy models."""
    sample = texts.sample(n=min(n_samples, len(texts)), random_state=42).tolist()
    times = []
    for text in sample:
        t0 = time.perf_counter()
        model.predict([text])
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


def time_inference_bert(tokenizer, bert, logit, texts: pd.Series, n_samples: int = 25) -> float:
    """Median per-document inference time in ms for CamemBERT encoder + logit on CPU."""
    import torch
    original_device = next(bert.parameters()).device
    cpu = torch.device("cpu")
    bert.to(cpu)
    try:
        sample = texts.sample(n=min(n_samples, len(texts)), random_state=42).tolist()
        times = []
        for text in sample:
            t0 = time.perf_counter()
            enc = tokenizer([text], padding=True, truncation=True, max_length=128, return_tensors="pt")
            with torch.no_grad():
                emb = bert(**enc).last_hidden_state[:, 0, :].numpy()
            logit.predict(emb)
            times.append((time.perf_counter() - t0) * 1000)
    finally:
        bert.to(original_device)
    return float(np.median(times))


_TBL_STYLES = [
    {"selector": "caption",        "props": [("font-size", "13px"), ("font-weight", "bold"), ("text-align", "center"), ("padding-bottom", "6px")]},
    {"selector": "th",             "props": [("font-weight", "bold"), ("text-align", "center"), ("padding", "4px 10px"), ("background-color", "#f8f8f8"), ("border", "1px solid #ddd")]},
    {"selector": "th.row_heading", "props": [("text-align", "left")]},
    {"selector": "td",             "props": [("text-align", "center"), ("padding", "4px 10px"), ("border", "1px solid #ddd")]},
    {"selector": "",               "props": [("border-collapse", "collapse"), ("font-size", "12px")]},
]

_CM_STYLES = [
    {"selector": "caption",        "props": [("font-size", "13px"), ("font-weight", "bold"), ("text-align", "center"), ("padding-bottom", "6px")]},
    {"selector": "th",             "props": [("font-weight", "bold"), ("text-align", "center"), ("padding", "6px 14px"), ("background-color", "#f8f8f8"), ("border", "1px solid #ddd")]},
    {"selector": "th.row_heading", "props": [("text-align", "left")]},
    {"selector": "th.index_name",  "props": [("font-weight", "normal"), ("font-style", "italic"), ("white-space", "nowrap")]},
    {"selector": "td",             "props": [("text-align", "center"), ("padding", "6px 14px"), ("border", "1px solid #ddd")]},
    {"selector": "",               "props": [("border-collapse", "collapse"), ("font-size", "12px")]},
]


def _metrics_html(metrics: pd.DataFrame, title: str, f_col: str) -> str:
    f_exists = f_col in metrics.columns
    _border  = "border-top: 3px double #aaa"

    def _row_style(row):
        if row.name != "macro":
            return [""] * len(row)
        base = f"background-color: #f0f4ff; {_border}"
        out  = [base] * len(row)
        if f_exists:
            out[list(row.index).index(f_col)] = f"background-color: #ffd700; font-weight: bold; color: #333; {_border}"
        return out

    return (
        metrics.style
        .apply(_row_style, axis=1)
        .apply_index(
            lambda idx: [f"font-weight: bold; {_border}" if v == "macro" else "" for v in idx],
            axis=0,
        )
        .format("{:.1%}")
        .set_caption(title)
        .set_table_styles(_TBL_STYLES)
        .to_html()
    )


def _cm_html(cm: np.ndarray, labels: list, title: str) -> str:
    df = pd.DataFrame(cm, index=labels, columns=pd.Index(labels, name="tr/pr"))
    max_val = int(cm.max()) or 1

    def _color(val):
        t = val / max_val
        r = int(222 + t * (8   - 222))
        g = int(235 + t * (48  - 235))
        b = int(247 + t * (107 - 247))
        text = "white" if t > 0.5 else "#333"
        return f"background-color: rgb({r},{g},{b}); color: {text}"

    return (
        df.style
        .map(_color)
        .format("{:,}")
        .set_caption(title)
        .set_table_styles(_CM_STYLES)
        .to_html()
    )


def _build_html(metrics, cm, ood_metrics, ood_cm, labels, f_col: str = "f1") -> str:
    def _pair(m, matrix, title_m, title_cm):
        return (
            "<div style='display:flex;gap:40px;align-items:flex-start;margin-bottom:28px'>"
            f"{_metrics_html(m, title_m, f_col)}"
            f"{_cm_html(matrix, labels, title_cm)}"
            "</div>"
        )

    body = _pair(metrics, cm, "OOS Metrics", "OOS Confusion Matrix")
    if ood_metrics is not None:
        body += _pair(ood_metrics, ood_cm, "OOD Metrics", "OOD Confusion Matrix")
    return body
