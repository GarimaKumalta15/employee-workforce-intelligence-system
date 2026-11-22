"""Inference engine: loads the trained bundle and scores employees."""
import joblib
import numpy as np
import pandas as pd
from .features import BASE, WITH_PERF, SEG_FEATURES, ANOM_FEATURES, LABELS, clean, impute, fmt_value


class Engine:
    def __init__(self, bundle: dict):
        self.b = bundle
        self.ref = bundle["ref"]

    @classmethod
    def load(cls, path):
        return cls(joblib.load(path))

    # ------------------------------------------------------------------ prepare
    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        d = impute(clean(df), self.ref)
        if "performance_score" not in d or d["performance_score"].isna().any():
            est = self.b["perf_model"].predict(d[BASE])
            if "performance_score" not in d:
                d["performance_score"] = est
            else:
                d["performance_score"] = d["performance_score"].fillna(pd.Series(est, index=d.index))
        return d

    # -------------------------------------------------------------------- score
    def score(self, df: pd.DataFrame, overrides: dict | None = None, prepared=False) -> pd.DataFrame:
        d = df.copy() if prepared else self.prepare(df)
        b, o = self.b, overrides or {}
        d["pred_performance"] = o.get("perf", b["perf_model"].predict(d[BASE])).clip(1, 5)
        d["attrition_prob"] = o.get("attr", b["attr_model"].predict_proba(d[WITH_PERF])[:, 1])
        d["promotion_prob"] = o.get("promo", b["promo_model"].predict_proba(d[WITH_PERF])[:, 1])
        d["training_prob"] = o.get("train", b["train_model"].predict_proba(d[WITH_PERF])[:, 1])

        th = b["thresholds"]
        d["likely_to_leave"] = d["attrition_prob"] >= th["attr"]
        d["promotion_candidate"] = d["promotion_prob"] >= th["promo"]
        d["training_needed"] = d["training_prob"] >= th["train"]

        # --- segmentation (K-Means)
        Xs = b["seg_scaler"].transform(d[SEG_FEATURES])
        sid = b["kmeans"].predict(Xs)
        d["segment_id"] = sid
        d["segment"] = [b["seg_names"][int(i)] for i in sid]

        # --- anomaly detection (Isolation Forest)
        Xa = b["anom_scaler"].transform(d[ANOM_FEATURES])
        raw = -b["iso"].score_samples(Xa)
        lo, hi = b["anom_range"]
        d["anomaly_score"] = np.clip((raw - lo) / (hi - lo), 0, 1)
        d["anomaly_flag"] = b["iso"].predict(Xa) == -1
        d["anomaly_reasons"] = [self._anomaly_reason(z) for z in Xa]

        # --- composite risk score (0-100)
        perf_gap = np.clip((4.0 - d["performance_score"]) / 3.0, 0, 1)
        absent = np.clip(d["absenteeism_days"] / 20, 0, 1)
        d["risk_score"] = 100 * (0.65 * d["attrition_prob"] + 0.15 * perf_gap
                                 + 0.10 * absent + 0.10 * d["anomaly_score"])
        c_med, c_high = b["risk_cutoffs"]
        d["risk_level"] = np.where(d["risk_score"] >= c_high, "High",
                                   np.where(d["risk_score"] >= c_med, "Medium", "Low"))
        return d

    def _anomaly_reason(self, z):
        order = np.argsort(-np.abs(z))[:2]
        bits = [f"{LABELS.get(ANOM_FEATURES[i], ANOM_FEATURES[i]).lower()} unusually "
                f"{'high' if z[i] > 0 else 'low'}" for i in order if abs(z[i]) >= 1.8]
        return "; ".join(bits) if bits else "overall combination of attributes is rare"

    # ------------------------------------------------------------------ explain
    def explain(self, row: pd.Series):
        from .explain import explain_row
        b, ref = self.b, self.ref
        return {
            "attrition": explain_row(b["attr_model"], row, WITH_PERF, ref),
            "performance": explain_row(b["perf_model"], row, BASE, ref),
            "promotion": explain_row(b["promo_model"], row, WITH_PERF, ref),
            "training": explain_row(b["train_model"], row, WITH_PERF, ref),
        }
