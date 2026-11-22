"""Rule-based HR recommendations + the AI workforce report text."""
import numpy as np
import pandas as pd

ACTIONS = {
    "overtime": "Reduce overtime: rebalance workload, add support or hire backfill.",
    "job_satisfaction": "Hold a 1:1 to uncover dissatisfaction; explore role enrichment or a project change.",
    "work_life_balance": "Offer flexible hours / hybrid working to improve work-life balance.",
    "environment_satisfaction": "Review team and workspace environment; run a short pulse survey.",
    "years_since_last_promotion": "Discuss career path and a clear promotion timeline.",
    "monthly_income": "Benchmark pay against the market band for this role and consider a revision.",
    "last_salary_hike_pct": "Review the last increment: it was low relative to peers.",
    "distance_from_home": "Offer hybrid/remote days or commute support.",
    "business_travel": "Reduce travel load or rotate travel assignments.",
    "manager_rating": "Provide manager coaching / consider a team change.",
    "job_involvement": "Give ownership of a visible project to raise involvement.",
    "age": "Pair with a mentor and a growth plan for early-career retention.",
    "years_at_company": "Run a structured career conversation about long-term growth.",
    "num_companies_worked": "History of frequent moves: set clear growth milestones early.",
    "training_hours_last_year": "Provide a learning plan; low training hours reduce engagement.",
    "performance_score": "Set a performance-improvement plan with measurable goals.",
}


def recommendations(row, attr_expl, promo_expl, train_expl):
    recs = []
    level = row["risk_level"]
    if level == "High":
        recs.append({"priority": "High", "text": "Schedule a retention conversation with the manager within 2 weeks."})
    seen = set()
    for d in attr_expl["increases"][:3]:
        if d["feature"] in ACTIONS and d["feature"] not in seen and d["impact"] >= 1.0:
            seen.add(d["feature"])
            recs.append({"priority": "High" if level == "High" else "Medium",
                         "text": f"{ACTIONS[d['feature']]}  (driver: {d['label']} = {d['value']})"})
    if row["promotion_candidate"]:
        recs.append({"priority": "Medium", "text": "Shortlist for the next promotion review: model rates promotion suitability highly."})
    if row["training_needed"]:
        top = [d["label"] for d in train_expl["increases"][:2]]
        extra = f" Focus areas: {', '.join(top)}." if top else ""
        recs.append({"priority": "Medium", "text": "Enrol in a targeted training programme." + extra})
    if bool(row["anomaly_flag"]):
        recs.append({"priority": "Medium", "text": f"Unusual profile detected ({row['anomaly_reasons']}). Verify the data and review with HR."})
    if row["segment"] == "Burnout Risk":
        recs.append({"priority": "Medium", "text": "Belongs to the Burnout Risk segment: monitor workload and absenteeism monthly."})
    if row["segment"] == "Rising Stars" and level != "Low":
        recs.append({"priority": "High", "text": "High-potential employee with elevated risk: prioritise retention (growth plan, recognition)."})
    if not recs:
        recs.append({"priority": "Low", "text": "No immediate action needed. Continue regular check-ins."})
    return recs


def employee_summary(row):
    name = row.get("name") or "This employee"
    parts = [f"{name} has a {row['risk_level'].lower()} overall risk level (score {row['risk_score']:.0f}/100) "
             f"with an estimated {row['attrition_prob'] * 100:.0f}% chance of leaving.",
             f"Predicted performance is {row['pred_performance']:.2f}/5; promotion suitability is "
             f"{row['promotion_prob'] * 100:.0f}% and training need is {row['training_prob'] * 100:.0f}%.",
             f"Workforce segment: {row['segment']}."]
    if bool(row["anomaly_flag"]):
        parts.append(f"Flagged as unusual: {row['anomaly_reasons']}.")
    return " ".join(parts)


def workforce_report(scored: pd.DataFrame, meta: dict) -> dict:
    n = len(scored)
    high = scored[scored.risk_level == "High"]
    dept = (scored.groupby("department")
            .agg(headcount=("employee_id", "count"), avg_attrition=("attrition_prob", "mean"),
                 high_risk=("risk_level", lambda s: (s == "High").sum()),
                 avg_perf=("performance_score", "mean")).sort_values("avg_attrition", ascending=False))
    worst = dept.index[0]
    ot = scored.groupby("overtime")["attrition_prob"].mean()
    seg = scored.groupby("segment").agg(count=("employee_id", "count"), avg_risk=("risk_score", "mean"),
                                        avg_perf=("performance_score", "mean"))
    top_drivers = meta["importances"]["attr"][:5]
    findings = [
        f"{len(high)} of {n} employees ({len(high) / n * 100:.0f}%) are in the High risk band and need attention.",
        f"{worst} has the highest average attrition probability ({dept.iloc[0].avg_attrition * 100:.0f}%).",
        f"Employees working overtime are {ot.get('Yes', 0) / max(ot.get('No', 1e-9), 1e-9):.1f}x more likely to leave than those who do not.",
        f"{int(scored.promotion_candidate.sum())} employees look promotion-ready; "
        f"{int(scored.training_needed.sum())} would benefit from training.",
        f"{int(scored.anomaly_flag.sum())} employees show unusual patterns worth a data / HR review.",
        "Top attrition drivers company-wide: " + ", ".join(d["label"] for d in top_drivers) + ".",
    ]
    actions = [
        f"Run stay-interviews for the {min(len(high), 15)} highest-risk employees first.",
        f"Audit workload and overtime policy in {worst}.",
        "Fast-track promotion reviews for high-probability candidates to prevent regretted exits.",
        "Launch targeted training for employees with a high training-need score.",
    ]
    return {
        "generated_for": n, "findings": findings, "actions": actions,
        "departments": dept.round(3).reset_index().to_dict("records"),
        "segments": seg.round(3).reset_index().to_dict("records"),
        "top_risk": high.sort_values("risk_score", ascending=False).head(10)[
            ["employee_id", "name", "department", "job_role", "risk_score", "attrition_prob"]].round(3).to_dict("records"),
    }
