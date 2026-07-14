"""Canonical configuration. Every number in the paper derives from these settings.

CRITICAL — the two-sample design:
  * Model selection / estimation / hold-out evaluation : FULL panel, year-median imputation
  * SHAP interpretation                                : COMPLETE-CASE subset (no imputed value
                                                         is ever assigned a feature attribution)
Changing the TRAINING sample changes the global ranking materially (sewerage 4 -> 8, rho = 0.82)
and flips model selection to XGBoost. Changing the EVALUATION sample does not (rho = 0.99).
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "panel_2017_2024.csv"
OUT_T = ROOT / "outputs" / "tables"
OUT_F = ROOT / "outputs" / "figures"

TARGET = "net_rate"
YEAR_TRAIN_END = 2022          # estimation window 2017-2022
YEAR_HOLDOUT = (2023, 2024)    # untouched hold-out
SEED = 42

FEATURES = [
    "pagerank_lag1", "closeness", "house_age", "childcare_pk", "pop_density",
    "seoul_dist_km", "ln_pop", "fertility", "doctor_per1000", "aging_ratio",
    "youth_ratio", "fiscal_indep", "employ_rate", "biz_count", "sewer_supply",
    "academy_pk", "culture_facility_count", "senior_fac_pk", "hospital_bed",
    "extinction_risk",
]
LABELS = {
    "pagerank_lag1": "PageRank (t-1)", "closeness": "Closeness centrality",
    "house_age": "Housing age", "childcare_pk": "Childcare facilities",
    "pop_density": "Population density", "seoul_dist_km": "Distance from Seoul",
    "ln_pop": "Population (ln)", "fertility": "Fertility rate",
    "doctor_per1000": "Doctors", "aging_ratio": "Ageing ratio",
    "youth_ratio": "Youth ratio", "fiscal_indep": "Fiscal independence",
    "employ_rate": "Employment rate", "biz_count": "Business establishments",
    "sewer_supply": "Sewerage supply", "academy_pk": "Private academies",
    "culture_facility_count": "Cultural facilities", "senior_fac_pk": "Senior-care facilities",
    "hospital_bed": "Hospital beds", "extinction_risk": "Extinction-risk index",
}
# Optuna-selected (80 TPE trials per algorithm, expanding-window CV objective)
CATBOOST = dict(iterations=677, learning_rate=0.10884860813834339, depth=8,
                l2_leaf_reg=2.1524708091423577, random_state=SEED, verbose=0)

STAGES = ["low", "intermediate", "high"]     # population-density tertiles
N_BOOT_STAGE = 2000     # fixed-model cluster bootstrap (municipality-level)
N_PERM = 2000           # stage-label permutation test
N_BOOT_GLOBAL = 50      # bootstrap WITH model refitting (global rank CIs)
N_BOOT_PERF = 2000      # province-block bootstrap of hold-out performance
