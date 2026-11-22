"""Feature definitions + data cleaning shared by training, API and tests."""
import numpy as np
import pandas as pd

CATEGORICAL = ["gender", "department", "job_role", "education", "overtime", "business_travel"]
NUMERIC = ["age", "distance_from_home", "job_level", "total_working_years", "years_at_company",
           "years_since_last_promotion", "num_companies_worked", "monthly_income",
           "last_salary_hike_pct", "job_satisfaction", "work_life_balance",
           "environment_satisfaction", "job_involvement", "manager_rating",
           "training_hours_last_year", "trainings_attended", "projects_completed",
           "absenteeism_days"]
BASE = NUMERIC + CATEGORICAL                 # features that do not use the performance score
WITH_PERF = BASE + ["performance_score"]      # attrition / promotion / training models

SEG_FEATURES = ["age", "monthly_income", "years_at_company", "performance_score", "job_satisfaction",
                "work_life_balance", "job_involvement", "training_hours_last_year",
                "absenteeism_days", "overtime_flag"]
ANOM_FEATURES = SEG_FEATURES + ["last_salary_hike_pct", "projects_completed",
                                "distance_from_home", "manager_rating"]

TARGETS = {"attrition": "attrition_flag", "promoted_next_cycle": "promotion_flag",
           "needs_training": "training_flag"}

RANGES = {"age": (18, 70), "distance_from_home": (0, 100), "job_level": (1, 5),
          "total_working_years": (0, 45), "years_at_company": (0, 40),
          "years_since_last_promotion": (0, 30), "num_companies_worked": (0, 15),
          "monthly_income": (5000, 1_000_000), "last_salary_hike_pct": (0, 50),
          "job_satisfaction": (1, 4), "work_life_balance": (1, 4),
          "environment_satisfaction": (1, 4), "job_involvement": (1, 4),
          "manager_rating": (1, 5), "training_hours_last_year": (0, 200),
          "trainings_attended": (0, 30), "projects_completed": (0, 50),
          "absenteeism_days": (0, 60), "performance_score": (1, 5)}

LABELS = {
    "age": "Age", "distance_from_home": "Distance from home", "job_level": "Job level",
    "total_working_years": "Total experience", "years_at_company": "Tenure at company",
    "years_since_last_promotion": "Years since last promotion", "num_companies_worked": "Companies worked at",
    "monthly_income": "Monthly income", "last_salary_hike_pct": "Last salary hike %",
    "job_satisfaction": "Job satisfaction", "work_life_balance": "Work-life balance",
    "environment_satisfaction": "Environment satisfaction", "job_involvement": "Job involvement",
    "manager_rating": "Manager rating", "training_hours_last_year": "Training hours (last year)",
    "trainings_attended": "Trainings attended", "projects_completed": "Projects completed",
    "absenteeism_days": "Absenteeism days", "gender": "Gender", "department": "Department",
    "job_role": "Job role", "education": "Education", "overtime": "Overtime",
    "business_travel": "Business travel", "performance_score": "Performance score",
    "overtime_flag": "Overtime",
}


def fmt_value(feature, v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    if feature == "monthly_income":
        return f"₹{float(v):,.0f}"
    if isinstance(v, str):
        return v
    v = float(v)
    return f"{v:.0f}" if v == int(v) else f"{v:.2f}".rstrip("0").rstrip(".")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Data cleaning: de-duplicate, tidy text, coerce numbers, clip outliers, derive flags."""
    d = df.copy()
    if "employee_id" in d:
        d = d.drop_duplicates(subset="employee_id")
    for c in CATEGORICAL:
        if c in d:
            d[c] = (d[c].astype(str).str.strip()
                    .replace({"nan": np.nan, "None": np.nan, "": np.nan, "<NA>": np.nan}))
    for c in NUMERIC + ["performance_score"]:
        if c in d:
            d[c] = pd.to_numeric(d[c], errors="coerce")
            lo, hi = RANGES[c]
            d[c] = d[c].clip(lo, hi)
    for raw, flag in TARGETS.items():
        if raw in d:
            d[flag] = d[raw].astype(str).str.strip().str.lower().eq("yes").astype(int)
    if "overtime" in d:
        d["overtime_flag"] = d["overtime"].eq("Yes").astype(int)
    return d.reset_index(drop=True)


def fit_reference(d: pd.DataFrame) -> dict:
    """Medians / modes used to impute missing values and as the 'typical employee' baseline."""
    fill = {c: float(d[c].median()) for c in NUMERIC + ["performance_score"]}
    fill.update({c: d[c].mode().iloc[0] for c in CATEGORICAL})
    options = {c: sorted(d[c].dropna().unique().tolist()) for c in CATEGORICAL}
    ranges = {c: [float(d[c].min()), float(d[c].max())] for c in NUMERIC + ["performance_score"]}
    roles = d.groupby("department")["job_role"].unique().apply(lambda s: sorted(s)).to_dict()
    return {"fill": fill, "options": options, "ranges": ranges, "roles_by_dept": roles}


def impute(d: pd.DataFrame, ref: dict) -> pd.DataFrame:
    d = d.copy()
    for c, v in ref["fill"].items():
        if c == "performance_score":
            continue
        if c not in d:
            d[c] = v
        else:
            d[c] = d[c].fillna(v)
    if "overtime" in d:
        d["overtime_flag"] = d["overtime"].eq("Yes").astype(int)
    return d
