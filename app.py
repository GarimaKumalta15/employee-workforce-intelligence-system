"""Flask backend + REST API for the Employee & Workforce Intelligence System."""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

from ml.engine import Engine
from ml.features import BASE, LABELS, NUMERIC, CATEGORICAL
from ml.insights import employee_summary, recommendations, workforce_report

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"


def ensure_artifacts():
    """Generate data + train models on first run."""
    if not (MODELS / "workforce_bundle.joblib").exists():
        from data.generate_dataset import generate, OUT
        if not OUT.exists():
            generate().to_csv(OUT, index=False)
        from ml.train import main as train
        train()


def native(o):
    """Make numpy / pandas values JSON friendly."""
    if isinstance(o, dict):
        return {str(k): native(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [native(v) for v in o]
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if (math.isnan(o) or math.isinf(o)) else round(float(o), 4)
    return o


def create_app():
    ensure_artifacts()
    app = Flask(__name__)
    engine = Engine.load(MODELS / "workforce_bundle.joblib")
    meta = json.loads((MODELS / "metadata.json").read_text())
    scored = pd.read_csv(MODELS / "scored_employees.csv")
    scored["anomaly_reasons"] = scored["anomaly_reasons"].fillna("")
    raw_by_id = scored.set_index("employee_id", drop=False)

    def ok(payload, status=200):
        return jsonify(native(payload)), status

    def build_report(row: pd.Series, include_profile=True):
        expl = engine.explain(row)
        recs = recommendations(row, expl["attrition"], expl["promotion"], expl["training"])
        profile_cols = ["employee_id", "name", "age", "gender", "education", "department", "job_role",
                        "job_level", "years_at_company", "total_working_years", "years_since_last_promotion",
                        "monthly_income", "overtime", "job_satisfaction", "work_life_balance",
                        "environment_satisfaction", "job_involvement", "manager_rating",
                        "training_hours_last_year", "absenteeism_days", "projects_completed", "business_travel",
                        "distance_from_home", "last_salary_hike_pct"]
        r = row.to_dict()
        return {
            "profile": {c: r.get(c) for c in profile_cols if c in r},
            "predictions": {
                "attrition_prob": r["attrition_prob"], "likely_to_leave": bool(r["likely_to_leave"]),
                "pred_performance": r["pred_performance"], "performance_score": r["performance_score"],
                "promotion_prob": r["promotion_prob"], "promotion_candidate": bool(r["promotion_candidate"]),
                "training_prob": r["training_prob"], "training_needed": bool(r["training_needed"]),
                "risk_score": r["risk_score"], "risk_level": r["risk_level"],
                "segment": r["segment"], "segment_profile": meta["seg_profiles"].get(r["segment"], ""),
                "anomaly_flag": bool(r["anomaly_flag"]), "anomaly_score": r["anomaly_score"],
                "anomaly_reasons": r["anomaly_reasons"],
            },
            "explanation": expl, "recommendations": recs, "summary": employee_summary(row),
        }

    # ------------------------------------------------------------------ pages
    @app.get("/")
    def index():
        return render_template("index.html")

    # -------------------------------------------------------------------- API
    @app.get("/api/meta")
    def api_meta():
        return ok({"options": meta["ref"]["options"], "ranges": meta["ref"]["ranges"],
                   "fill": meta["ref"]["fill"], "roles_by_dept": meta["ref"]["roles_by_dept"],
                   "numeric": NUMERIC, "categorical": CATEGORICAL, "labels": LABELS,
                   "segments": sorted(scored.segment.unique())})

    @app.get("/api/overview")
    def api_overview():
        s = scored
        dept = s.groupby("department").agg(headcount=("employee_id", "count"),
                                           avg_attrition=("attrition_prob", "mean"),
                                           high_risk=("risk_level", lambda x: (x == "High").sum()))
        bins = [1, 2, 2.5, 3, 3.5, 4, 4.5, 5.01]
        perf = pd.cut(s.performance_score, bins=bins, right=False).value_counts().sort_index()
        seg = s.groupby("segment").agg(count=("employee_id", "count"), avg_risk=("risk_score", "mean"),
                                       avg_perf=("performance_score", "mean"))
        return ok({
            "kpis": {"total": len(s), "attrition_rate": (s.attrition == "Yes").mean(),
                     "avg_attrition_prob": s.attrition_prob.mean(),
                     "high_risk": int((s.risk_level == "High").sum()),
                     "avg_performance": s.performance_score.mean(),
                     "promotion_candidates": int(s.promotion_candidate.sum()),
                     "training_needed": int(s.training_needed.sum()),
                     "anomalies": int(s.anomaly_flag.sum())},
            "departments": dept.round(3).reset_index().to_dict("records"),
            "risk_levels": s.risk_level.value_counts().reindex(["Low", "Medium", "High"]).fillna(0).to_dict(),
            "segments": seg.round(3).reset_index().to_dict("records"),
            "performance_hist": {"labels": [f"{iv.left:g}-{min(iv.right, 5):g}" for iv in perf.index], "counts": perf.tolist()},
            "drivers": meta["importances"]["attr"][:8],
        })

    @app.get("/api/employees")
    def api_employees():
        q = request.args.get("q", "").strip().lower()
        s = scored
        if q:
            s = s[s.employee_id.str.lower().str.contains(q) | s.name.str.lower().str.contains(q)
                  | s.job_role.str.lower().str.contains(q)]
        for key, col in (("department", "department"), ("risk", "risk_level"), ("segment", "segment")):
            v = request.args.get(key)
            if v:
                s = s[s[col] == v]
        if request.args.get("anomaly") == "1":
            s = s[s.anomaly_flag]
        sort = request.args.get("sort", "risk_score")
        if sort not in s.columns:
            sort = "risk_score"
        s = s.sort_values(sort, ascending=request.args.get("order") == "asc")
        page = max(int(request.args.get("page", 1)), 1)
        per = min(int(request.args.get("per_page", 15)), 100)
        cols = ["employee_id", "name", "department", "job_role", "risk_score", "risk_level", "attrition_prob",
                "performance_score", "segment", "anomaly_flag"]
        return ok({"total": len(s), "page": page, "pages": max(math.ceil(len(s) / per), 1),
                   "rows": s.iloc[(page - 1) * per: page * per][cols].to_dict("records")})

    @app.get("/api/employee/<emp_id>")
    def api_employee(emp_id):
        key = emp_id.upper()
        if key not in raw_by_id.index:
            return ok({"error": f"Employee {emp_id} not found"}, 404)
        return ok(build_report(raw_by_id.loc[key]))

    @app.post("/api/predict")
    def api_predict():
        """What-if: score a hypothetical / new employee from submitted attributes."""
        data = request.get_json(silent=True) or {}
        row = {c: data.get(c) for c in BASE if data.get(c) not in (None, "")}
        if not row:
            return ok({"error": "Send a JSON body with employee attributes."}, 400)
        row["name"] = data.get("name") or "Candidate"
        row["employee_id"] = "NEW"
        try:
            d = engine.score(pd.DataFrame([row]))
        except Exception as exc:  # bad input types etc.
            return ok({"error": f"Could not score input: {exc}"}, 400)
        out = build_report(d.iloc[0])
        out["profile"]["name"] = row["name"]
        return ok(out)

    @app.get("/api/report")
    def api_report():
        return ok(workforce_report(scored, meta))

    @app.get("/api/anomalies")
    def api_anomalies():
        a = scored[scored.anomaly_flag].sort_values("anomaly_score", ascending=False)
        return ok(a[["employee_id", "name", "department", "job_role", "anomaly_score", "anomaly_reasons",
                     "risk_level"]].to_dict("records"))

    @app.get("/api/model-info")
    def api_model_info():
        figs = sorted(p.name for p in (ROOT / "static" / "figures").glob("*.png"))
        return ok({"metrics": meta["metrics"], "importances": meta["importances"],
                   "thresholds": meta["thresholds"], "figures": figs})

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
