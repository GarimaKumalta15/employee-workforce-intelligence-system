"""Explainable AI: model-agnostic 'occlusion' explanations.

For one employee we replace each feature, one at a time, with the workforce's typical
value (median / mode) and measure how the model's output changes.  A large positive
change means that feature is pushing the prediction UP for this employee.
Works with any scikit-learn pipeline, needs no extra libraries.
"""
import numpy as np
import pandas as pd
from .features import LABELS, fmt_value


def _predict(pipe, X):
    return pipe.predict_proba(X)[:, 1] if hasattr(pipe, "predict_proba") else pipe.predict(X)


def explain_row(pipe, row: pd.Series, feats: list, ref: dict, top=5):
    base = {f: row[f] for f in feats}
    variants = [dict(base)]
    for f in feats:
        v = dict(base)
        v[f] = ref["fill"][f]
        variants.append(v)
    X = pd.DataFrame(variants)[feats]
    out = _predict(pipe, X)
    is_prob = hasattr(pipe, "predict_proba")
    impacts = []
    for i, f in enumerate(feats):
        delta = float(out[0] - out[i + 1])
        if abs(delta) * (100 if is_prob else 1) < (0.3 if is_prob else 0.02):
            continue                      # ignore negligible / noise-level effects
        impacts.append({
            "feature": f, "label": LABELS.get(f, f), "value": fmt_value(f, base[f]),
            "typical": fmt_value(f, ref["fill"][f]),
            "impact": round(delta * (100 if is_prob else 1), 2),
        })
    up = sorted([i for i in impacts if i["impact"] > 0], key=lambda x: -x["impact"])[:top]
    down = sorted([i for i in impacts if i["impact"] < 0], key=lambda x: x["impact"])[:top]
    return {"unit": "percentage points" if is_prob else "score points",
            "increases": up, "decreases": down}
