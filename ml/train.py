"""Train every model, evaluate, create figures and save artefacts.

Run:  python -m ml.train
"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
                             precision_score, r2_score, recall_score, roc_auc_score, silhouette_score)
from sklearn.model_selection import cross_val_predict, train_test_split, StratifiedKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .engine import Engine
from .features import (ANOM_FEATURES, BASE, CATEGORICAL, LABELS, NUMERIC, SEG_FEATURES, WITH_PERF,
                       clean, fit_reference, impute)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "employee_hr_data.csv"
MODELS = ROOT / "models"
FIGS = ROOT / "static" / "figures"
SEED = 42
sns.set_theme(style="whitegrid", palette="deep")


def make_pipe(model, feats):
    cats = [f for f in feats if f in CATEGORICAL]
    nums = [f for f in feats if f not in CATEGORICAL]
    pre = ColumnTransformer([("num", "passthrough", nums),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), cats)])
    return Pipeline([("prep", pre), ("model", model)])


def rf_clf():
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=4, max_features="sqrt",
                                  n_jobs=-1, random_state=SEED)


def rf_reg():
    return RandomForestRegressor(n_estimators=300, min_samples_leaf=3, max_features=0.5,
                                 n_jobs=-1, random_state=SEED)


def best_threshold(y, p):
    ts = np.linspace(0.05, 0.8, 76)
    return float(ts[np.argmax([f1_score(y, p >= t) for t in ts])])


def savefig(name):
    plt.tight_layout()
    plt.savefig(FIGS / name, dpi=130)
    plt.close()


def name_segments(km, scaler):
    """Give K-Means clusters human names by matching centroids to HR archetypes."""
    z = pd.DataFrame(km.cluster_centers_, columns=SEG_FEATURES)
    archetypes = {
        "Rising Stars": z.performance_score + z.training_hours_last_year + z.job_involvement - z.years_at_company,
        "Established Core": z.years_at_company + z.monthly_income + z.age + z.job_satisfaction,
        "Burnout Risk": z.overtime_flag + z.absenteeism_days - z.work_life_balance - z.job_satisfaction,
        "Disengaged": -z.job_satisfaction - z.job_involvement - z.performance_score - z.training_hours_last_year,
    }
    names = list(archetypes)
    cost = -np.array([archetypes[n].values for n in names]).T      # clusters x names
    rows, cols = linear_sum_assignment(cost)
    seg_names = {int(r): names[c] for r, c in zip(rows, cols)}
    profiles = {}
    for i in range(len(z)):
        s = z.iloc[i].sort_values()
        hi = [LABELS[f].lower() for f in s.index[::-1][:2]]
        lo = [LABELS[f].lower() for f in s.index[:2]]
        profiles[int(i)] = f"Higher {', '.join(hi)}; lower {', '.join(lo)} than the workforce average."
    return seg_names, profiles


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(exist_ok=True)

    # 1) data collection + 2) cleaning
    raw = pd.read_csv(DATA)
    df = clean(raw)
    ref = fit_reference(df)
    df = impute(df, ref)
    print(f"Loaded {len(raw)} rows -> {len(df)} after cleaning (duplicates removed, missing values imputed)")

    metrics, importances = {}, {}
    oof, finals = {}, {}
    specs = {
        "perf": ("Performance (regression)", rf_reg, BASE, "performance_score"),
        "attr": ("Attrition", rf_clf, WITH_PERF, "attrition_flag"),
        "promo": ("Promotion", rf_clf, WITH_PERF, "promotion_flag"),
        "train": ("Training need", rf_clf, WITH_PERF, "training_flag"),
    }

    # 3) feature selection + 4) training + 5) prediction
    for key, (title, factory, feats, target) in specs.items():
        X, y = df[feats], df[target]
        is_clf = key != "perf"
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED,
                                              stratify=y if is_clf else None)
        pipe = make_pipe(factory(), feats).fit(Xtr, ytr)
        if is_clf:
            p = pipe.predict_proba(Xte)[:, 1]
            # decision threshold tuned for F1 on out-of-fold training predictions (no test leakage)
            thr = best_threshold(ytr, cross_val_predict(make_pipe(factory(), feats), Xtr, ytr, cv=3,
                                                        method="predict_proba")[:, 1])
            pred = p >= thr
            metrics[key] = {"title": title, "type": "classification", "threshold": round(thr, 2),
                            "accuracy": accuracy_score(yte, pred), "precision": precision_score(yte, pred, zero_division=0),
                            "recall": recall_score(yte, pred), "f1": f1_score(yte, pred),
                            "roc_auc": roc_auc_score(yte, p), "positive_rate": float(y.mean()),
                            "test_size": int(len(yte))}
            cm = confusion_matrix(yte, pred)
            plt.figure(figsize=(4.2, 3.6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                        xticklabels=["Stay", "Yes"], yticklabels=["Stay", "Yes"])
            plt.title(f"{title}: confusion matrix"); plt.xlabel("Predicted"); plt.ylabel("Actual")
            savefig(f"cm_{key}.png")
            folds = StratifiedKFold(5, shuffle=True, random_state=SEED)
            oof[key] = cross_val_predict(make_pipe(factory(), feats), X, y, cv=folds, method="predict_proba")[:, 1]
        else:
            pred = pipe.predict(Xte)
            metrics[key] = {"title": title, "type": "regression", "r2": r2_score(yte, pred),
                            "mae": mean_absolute_error(yte, pred), "test_size": int(len(yte))}
            oof[key] = cross_val_predict(make_pipe(factory(), feats), X, y,
                                         cv=KFold(5, shuffle=True, random_state=SEED))
        pi = permutation_importance(pipe, Xte, yte, n_repeats=5, random_state=SEED, n_jobs=-1,
                                    scoring="roc_auc" if is_clf else "r2")
        imp = pd.Series(pi.importances_mean, index=feats).sort_values(ascending=False)
        std = pd.Series(pi.importances_std, index=feats)
        # keep only features whose importance is clearly above permutation noise
        importances[key] = [{"feature": f, "label": LABELS.get(f, f), "importance": round(float(v), 4)}
                            for f, v in imp.items() if v > std[f]][:12]
        finals[key] = make_pipe(factory(), feats).fit(X, y)         # production model: all data
        print(f"  {title:28s}", {k: round(v, 3) for k, v in metrics[key].items() if isinstance(v, float)})

        top = imp.head(10)[::-1]
        plt.figure(figsize=(6.2, 4))
        plt.barh([LABELS.get(f, f) for f in top.index], top.values, color=sns.color_palette("deep")[0])
        plt.title(f"{title}: top features (permutation importance)")
        savefig(f"importance_{key}.png")

    # 6) segmentation (K-Means) -------------------------------------------------
    seg_scaler = StandardScaler().fit(df[SEG_FEATURES])
    Xs = seg_scaler.transform(df[SEG_FEATURES])
    sil = {}
    for k in range(2, 8):
        sil[k] = silhouette_score(Xs, KMeans(k, n_init=10, random_state=SEED).fit_predict(Xs))
    kmeans = KMeans(4, n_init=20, random_state=SEED).fit(Xs)
    seg_names, seg_profiles = name_segments(kmeans, seg_scaler)
    metrics["segmentation"] = {"title": "Employee segmentation (K-Means, k=4)", "type": "clustering",
                               "silhouette": float(silhouette_score(Xs, kmeans.labels_)),
                               "silhouette_by_k": {str(k): round(v, 3) for k, v in sil.items()}}
    plt.figure(figsize=(5.2, 3.6)); plt.plot(list(sil), list(sil.values()), "o-")
    plt.axvline(4, ls="--", c="grey"); plt.xlabel("k (clusters)"); plt.ylabel("Silhouette score")
    plt.title("Choosing k for K-Means"); savefig("kmeans_silhouette.png")
    pcs = PCA(2, random_state=SEED).fit_transform(Xs)
    plt.figure(figsize=(6, 4.4))
    sns.scatterplot(x=pcs[:, 0], y=pcs[:, 1], hue=[seg_names[int(i)] for i in kmeans.labels_], s=22, alpha=.8)
    plt.title("Employee segments (PCA view)"); plt.xlabel("PC1"); plt.ylabel("PC2"); savefig("segments_pca.png")

    # 7) anomaly detection (Isolation Forest) ---------------------------------
    anom_scaler = StandardScaler().fit(df[ANOM_FEATURES])
    Xa = anom_scaler.transform(df[ANOM_FEATURES])
    iso = IsolationForest(n_estimators=300, contamination=0.04, random_state=SEED).fit(Xa)
    raw_sc = -iso.score_samples(Xa)
    anom_range = (float(np.percentile(raw_sc, 1)), float(np.percentile(raw_sc, 99)))

    bundle = {"ref": ref, "perf_model": finals["perf"], "attr_model": finals["attr"],
              "promo_model": finals["promo"], "train_model": finals["train"],
              "seg_scaler": seg_scaler, "kmeans": kmeans, "seg_names": seg_names,
              "anom_scaler": anom_scaler, "iso": iso, "anom_range": anom_range,
              "risk_cutoffs": (0, 0),
              "thresholds": {k: metrics[k]["threshold"] for k in ("attr", "promo", "train")}}

    # 8) risk bands: top 10% = High, next 20% = Medium  --------------------------
    engine = Engine(bundle)
    overrides = {"perf": oof["perf"], "attr": oof["attr"], "promo": oof["promo"], "train": oof["train"]}
    first = engine.score(df, overrides, prepared=True)
    bundle["risk_cutoffs"] = (float(first.risk_score.quantile(0.70)), float(first.risk_score.quantile(0.90)))
    scored = engine.score(df, overrides, prepared=True)

    cols = (["employee_id", "name", "age", "gender", "education", "department", "job_role", "job_level",
             "business_travel", "distance_from_home", "total_working_years", "years_at_company",
             "years_since_last_promotion", "num_companies_worked", "monthly_income", "last_salary_hike_pct",
             "overtime", "job_satisfaction", "work_life_balance", "environment_satisfaction",
             "job_involvement", "manager_rating", "training_hours_last_year", "trainings_attended",
             "projects_completed", "absenteeism_days", "performance_score", "attrition", "promoted_next_cycle",
             "needs_training", "pred_performance", "likely_to_leave", "promotion_candidate", "training_needed", "attrition_prob", "promotion_prob", "training_prob",
             "risk_score", "risk_level", "segment", "anomaly_score", "anomaly_flag", "anomaly_reasons"])
    scored[cols].to_csv(MODELS / "scored_employees.csv", index=False)

    # extra figures --------------------------------------------------------
    plt.figure(figsize=(8, 6.4))
    sns.heatmap(df[NUMERIC + ["performance_score", "attrition_flag"]].corr(), cmap="coolwarm", center=0,
                square=True, cbar_kws={"shrink": .7}, xticklabels=True, yticklabels=True)
    plt.xticks(fontsize=6); plt.yticks(fontsize=6); plt.title("Correlation heatmap"); savefig("correlation.png")
    plt.figure(figsize=(6.4, 3.8))
    order = scored.groupby("department").attrition_prob.mean().sort_values().index
    sns.barplot(data=scored, y="department", x="attrition_prob", order=order, errorbar=None)
    plt.title("Average predicted attrition by department"); savefig("attrition_by_dept.png")
    plt.figure(figsize=(6, 3.6))
    sns.histplot(scored.risk_score, bins=30, kde=True)
    for c in bundle["risk_cutoffs"]:
        plt.axvline(c, ls="--", c="red")
    plt.title("Risk score distribution (dashed = Medium/High cut-offs)"); savefig("risk_distribution.png")

    metrics["data"] = {"rows_raw": int(len(raw)), "rows_clean": int(len(df)),
                       "duplicates_removed": int(len(raw) - len(df)),
                       "missing_values_imputed": int(raw[NUMERIC + CATEGORICAL].isna().sum().sum())}
    meta = {"metrics": metrics, "importances": importances,
            "seg_profiles": {seg_names[i]: p for i, p in seg_profiles.items()},
            "ref": {"options": ref["options"], "ranges": ref["ranges"], "fill": ref["fill"],
                    "roles_by_dept": ref["roles_by_dept"]},
            "risk_cutoffs": list(bundle["risk_cutoffs"]), "thresholds": bundle["thresholds"]}
    (MODELS / "metadata.json").write_text(json.dumps(meta, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    joblib.dump(bundle, MODELS / "workforce_bundle.joblib", compress=3)
    print("Saved models, metadata, scored data and figures.")


if __name__ == "__main__":
    main()
