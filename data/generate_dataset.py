"""Generate a realistic synthetic Employee HR dataset (CSV).

Targets (attrition, promotion, training need, performance) are produced from
the other columns + random noise, so the ML models have real signal to learn
while still being imperfect, like real HR data.  A few missing values and
duplicate rows are injected on purpose to exercise the data-cleaning step.
"""
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(__file__).with_name("employee_hr_data.csv")

ROLES = {
    "Engineering": {"Software Engineer": 60000, "Senior Software Engineer": 95000,
                    "QA Engineer": 48000, "DevOps Engineer": 78000},
    "Data & Analytics": {"Data Analyst": 55000, "Data Scientist": 85000, "ML Engineer": 95000},
    "Human Resources": {"HR Executive": 38000, "HR Manager": 70000, "Recruiter": 36000},
    "Sales": {"Sales Executive": 42000, "Account Manager": 62000, "Sales Manager": 88000},
    "Marketing": {"Marketing Executive": 40000, "Content Strategist": 45000, "Brand Manager": 75000},
    "Finance": {"Accountant": 44000, "Financial Analyst": 58000, "Finance Manager": 90000},
    "Operations": {"Operations Executive": 38000, "Operations Manager": 80000, "Support Specialist": 32000},
}
DEPT_P = [0.26, 0.14, 0.08, 0.17, 0.10, 0.10, 0.15]
FIRST = ["Aarav", "Vivaan", "Aditya", "Arjun", "Rohan", "Karan", "Rahul", "Siddharth", "Amit", "Vikram",
         "Neeraj", "Manish", "Deepak", "Harsh", "Nikhil", "Priya", "Ananya", "Isha", "Neha", "Pooja",
         "Kavya", "Riya", "Sneha", "Meera", "Divya", "Simran", "Tanvi", "Aditi", "Shreya", "Nisha",
         "Mohit", "Gaurav", "Varun", "Tarun", "Ritika", "Komal", "Swati", "Jaspreet", "Harleen", "Pankaj"]
LAST = ["Sharma", "Verma", "Singh", "Gupta", "Mehta", "Kaur", "Joshi", "Rawat", "Negi", "Bisht",
        "Patel", "Reddy", "Nair", "Iyer", "Kapoor", "Malhotra", "Chauhan", "Bhatt", "Pant", "Agarwal",
        "Mishra", "Yadav", "Thakur", "Sethi", "Arora", "Bansal", "Saxena", "Tiwari", "Dutta", "Khanna"]


def _sigmoid_pick(z, rate, rng):
    """Draw a binary outcome from latent logit z, shifting intercept so mean == rate."""
    noise = rng.logistic(0, 1, len(z))
    lo, hi = -20.0, 20.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if ((z + mid + noise) > 0).mean() > rate:
            hi = mid
        else:
            lo = mid
    return ((z + mid + noise) > 0).astype(int)


def generate(n=1500, seed=42):
    rng = np.random.default_rng(seed)
    dept = rng.choice(list(ROLES), n, p=DEPT_P)
    role = np.array([rng.choice(list(ROLES[d])) for d in dept])
    base_pay = np.array([ROLES[d][r] for d, r in zip(dept, role)], dtype=float)

    age = np.clip(rng.normal(36, 9, n), 21, 60).round().astype(int)
    work_years = np.clip((age - 21) * rng.uniform(0.6, 1.0, n), 0, 40).round().astype(int)
    years_co = np.clip(work_years * rng.uniform(0.1, 0.8, n) + rng.exponential(0.5, n), 0, 30).round().astype(int)
    years_co = np.minimum(years_co, work_years)
    since_promo = np.minimum(years_co, rng.exponential(2.2, n).round().astype(int))
    job_level = np.clip(1 + work_years // 8 + rng.integers(-1, 2, n), 1, 5)
    income = (base_pay * (1 + 0.035 * years_co) * rng.lognormal(0, 0.12, n)).round(-2)
    income_ratio = income / base_pay

    gender = rng.choice(["Male", "Female"], n, p=[0.56, 0.44])
    education = rng.choice(["Below College", "College", "Bachelor", "Master", "Doctor"], n,
                           p=[0.06, 0.16, 0.45, 0.27, 0.06])
    travel = rng.choice(["Non-Travel", "Travel_Rarely", "Travel_Frequently"], n, p=[0.12, 0.68, 0.20])
    overtime = rng.choice(["Yes", "No"], n, p=[0.28, 0.72])
    ot = (overtime == "Yes").astype(int)
    distance = np.clip(rng.exponential(9, n), 1, 60).round().astype(int)
    companies = np.clip(rng.poisson(2.2, n), 0, 9)

    job_sat = rng.choice([1, 2, 3, 4], n, p=[0.17, 0.2, 0.33, 0.30])
    wlb = np.clip(rng.choice([1, 2, 3, 4], n, p=[0.08, 0.22, 0.5, 0.2]) - (ot * (rng.random(n) < 0.35)), 1, 4)
    env_sat = rng.choice([1, 2, 3, 4], n, p=[0.15, 0.2, 0.34, 0.31])
    involvement = rng.choice([1, 2, 3, 4], n, p=[0.06, 0.24, 0.5, 0.2])
    manager = np.clip(np.round(rng.normal(3.5, 0.8, n)), 1, 5).astype(int)
    train_hours = np.clip(rng.normal(32, 14, n), 0, 90).round().astype(int)
    trainings = np.clip(rng.poisson(2.6, n), 0, 12)
    projects = np.clip(rng.poisson(6, n) + job_level, 0, 25)
    hike = np.clip(rng.normal(12, 4, n), 3, 25).round(1)
    absent = np.clip(rng.poisson(4.5, n) + ot * rng.poisson(2, n) + (job_sat == 1) * rng.poisson(2, n), 0, 40)

    perf = (3.0 + 0.30 * (job_sat - 2.5) + 0.40 * (manager - 3.5) + 0.012 * (train_hours - 32)
            + 0.03 * (projects - 8) - 0.035 * (absent - 5) + 0.15 * (involvement - 2.8)
            + 0.02 * (years_co ** 0.5) + rng.normal(0, 0.3, n))
    perf = np.clip(perf, 1, 5).round(2)

    z_attr = (1.1 * ot - 0.55 * (job_sat - 2.5) - 0.45 * (wlb - 2.5) - 0.35 * (env_sat - 2.5)
              + 0.12 * since_promo - 1.6 * (income_ratio - 1.05) + 0.02 * distance
              + 0.7 * (age < 30) - 0.05 * years_co + 0.6 * (travel == "Travel_Frequently")
              - 0.4 * (perf - 3.2) - 0.05 * (hike - 12) + 0.08 * companies
              - 0.25 * (involvement - 2.8))
    attrition = _sigmoid_pick(z_attr, 0.17, rng)

    z_promo = (1.4 * (perf - 3.4) + 0.22 * np.minimum(since_promo, 6) + 0.25 * (manager - 3.5)
               + 0.012 * (train_hours - 32) + 0.2 * (involvement - 2.8) - 0.5 * (job_level >= 5)
               + 0.15 * (education == "Master") - 0.2 * (years_co < 2))
    promo = _sigmoid_pick(z_promo, 0.14, rng)

    z_train = (-0.06 * (train_hours - 32) - 1.3 * (perf - 3.2) - 0.25 * (trainings - 2.6)
               + 0.5 * (job_level <= 2) + 0.3 * (years_co < 2) - 0.2 * (manager - 3.5))
    train_need = _sigmoid_pick(z_train, 0.30, rng)

    df = pd.DataFrame({
        "employee_id": [f"EMP{i:04d}" for i in range(1, n + 1)],
        "name": [f"{rng.choice(FIRST)} {rng.choice(LAST)}" for _ in range(n)],
        "age": age, "gender": gender, "education": education,
        "department": dept, "job_role": role, "job_level": job_level,
        "business_travel": travel, "distance_from_home": distance,
        "total_working_years": work_years, "years_at_company": years_co,
        "years_since_last_promotion": since_promo, "num_companies_worked": companies,
        "monthly_income": income.astype(int), "last_salary_hike_pct": hike,
        "overtime": overtime, "job_satisfaction": job_sat, "work_life_balance": wlb,
        "environment_satisfaction": env_sat, "job_involvement": involvement,
        "manager_rating": manager, "training_hours_last_year": train_hours,
        "trainings_attended": trainings, "projects_completed": projects,
        "absenteeism_days": absent, "performance_score": perf,
        "attrition": np.where(attrition == 1, "Yes", "No"),
        "promoted_next_cycle": np.where(promo == 1, "Yes", "No"),
        "needs_training": np.where(train_need == 1, "Yes", "No"),
    })

    # --- inject realistic mess: ~1.5% missing values + a few duplicate rows ---
    for col in ["distance_from_home", "training_hours_last_year", "education",
                "last_salary_hike_pct", "job_satisfaction", "manager_rating"]:
        df.loc[rng.random(n) < 0.015, col] = np.nan
    df = pd.concat([df, df.sample(8, random_state=seed)], ignore_index=True)
    return df


if __name__ == "__main__":
    data = generate()
    data.to_csv(OUT, index=False)
    print(f"Saved {len(data)} rows -> {OUT}")
