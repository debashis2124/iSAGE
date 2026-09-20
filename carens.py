#!/usr/bin/env python3
"""
iSAGE multi-dataset experiment: S-NN vs iSAGE-Train vs iSAGE-Fusion
================================

Runs the iSAGE experimental pattern on:
1) Breast Cancer
2) Cardiovascular Disease
3) Diabetes

Main experiments:
- full-resource performance
- training-data scarcity: 100, 50, 25, 10, 5%
- Rule-Only complementarity
- observation scarcity: 100, 75, 50, 25%
- symbolic knowledge is injected into iSAGE during neural training
- symbolic knowledge corruption: removal + reversal at 0,10,20,30,40%
- joint resource degradation: S1/S2/S3
- Resource Robustness Score (RRS)
- paired Wilcoxon statistics
- CPU inference latency / serialized model size
- publication-ready combined figures containing all three datasets
- CSV tables for every experiment

IMPORTANT RESEARCH NOTE
-----------------------
The symbolic rules below are dataset-specific candidate knowledge bases.
Before publication, every clinical threshold/rule should be justified by
a clinical reference and reported in the paper/supplement. The code never
learns symbolic rule directions from test labels.

Run from VS Code terminal:
    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install -U pip
    pip install numpy pandas scipy scikit-learn matplotlib psutil joblib xgboost
    python isage_three_dataset_experiment.py

Quick smoke test:
    python isage_three_dataset_experiment.py --quick

Paper run:
    python isage_three_dataset_experiment.py --paper

Optional conventional full-resource baselines:
    python isage_three_dataset_experiment.py --paper --full-baselines
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import wilcoxon
from scipy.interpolate import PchipInterpolator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    brier_score_loss,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except Exception:
    HAVE_XGB = False


# ============================================================
# GLOBAL CONFIGURATION
# ============================================================

PAPER_SEEDS = [11, 22, 33, 44, 55, 66, 77, 88, 99, 111]
QUICK_SEEDS = [11, 22, 33]

DATA_FRACTIONS = [1.00, 0.50, 0.30, 0.20, 0.10, 0.05]
OBSERVATION_FRACTIONS = [1.00, 0.50, 0.30, 0.20, 0.10]
TRAINING_PERCENT_TICKS = [5, 10, 20, 30, 50, 100]
OBSERVATION_PERCENT_TICKS = [10, 20, 30, 50, 100]
CORRUPTION_LEVELS = [0.00, 0.10, 0.20, 0.30, 0.40]
CORRUPTION_MODES = ["removal", "reversal"]

RESOURCE_CONDITIONS = {
    "S1_Full": {"data_fraction": 1.00, "observation_fraction": 1.00},
    "S2_Moderate": {"data_fraction": 0.50, "observation_fraction": 0.75},
    "S3_Severe": {"data_fraction": 0.25, "observation_fraction": 0.50},
}

BETA_GRID = [0.5, 1.0, 2.0, 3.0]
ALPHA_GRID = [0.5, 0.6, 0.7, 0.8, 0.9]

METRICS_HIGHER_BETTER = [
    "AUROC", "Macro_F1", "Balanced_Accuracy", "Sensitivity", "Specificity"
]
METRICS_LOWER_BETTER = ["Brier", "ECE", "RVR"]

# User requested different colors for the three datasets.
DATASET_COLORS = {
    "Breast Cancer": "#3B82F6",
    "Cardiovascular": "#F97316",
    "Diabetes": "#10B981",
}
MODEL_LINESTYLES = {
    "S-NN": "--",
    "iSAGE-Train": "-.",
    "iSAGE-Fusion": "-",
    "Rule-Only": ":",
}
MODEL_MARKERS = {
    "S-NN": "s",
    "iSAGE-Train": "D",
    "iSAGE-Fusion": "o",
    "Rule-Only": "^",
}

PLOT_FONT_SIZE = 30

# Method names used in plots/tables:
# S-NN          : matched neural-only baseline
# iSAGE-Train  : symbolic knowledge is incorporated during neural training
# iSAGE-Fusion : proposed method; S-NN is trained first, then symbolic
#                 probability is fused with the neural output using
#                 validation-selected alpha and beta
# Rule-Only     : auxiliary symbolic-only reference

plt.rcParams.update({
    "font.size": PLOT_FONT_SIZE,
    "axes.titlesize": PLOT_FONT_SIZE,
    "axes.labelsize": PLOT_FONT_SIZE,
    "xtick.labelsize": PLOT_FONT_SIZE,
    "ytick.labelsize": PLOT_FONT_SIZE,
    "legend.fontsize": PLOT_FONT_SIZE,
    "figure.dpi": 130,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.18,
    "grid.linestyle": "--",
    "lines.linewidth": 3.8,
    "lines.markersize": 10,
})


@dataclass
class DatasetSpec:
    name: str
    filename: str
    target: str
    positive_value: object
    drop_columns: List[str]
    symbolic_features: List[str]


DATASETS = {
    "Breast Cancer": DatasetSpec(
        name="Breast Cancer",
        filename="breast_cancer_enhanced_dataset(1).csv",
        target="diagnosis",
        positive_value="M",
        drop_columns=["id"],
        symbolic_features=[
            "radius_mean", "texture_mean", "compactness_mean",
            "concavity_mean", "concave points_mean", "smoothness_mean",
        ],
    ),
    "Cardiovascular": DatasetSpec(
        name="Cardiovascular",
        filename="cardio_train(1).csv",
        target="cardio",
        positive_value=1,
        drop_columns=["id"],
        symbolic_features=[
            "age", "height", "weight", "ap_hi", "ap_lo",
            "cholesterol", "gluc", "smoke", "active",
        ],
    ),
    "Diabetes": DatasetSpec(
        name="Diabetes",
        filename="diabetes_prediction_dataset(1).csv",
        target="diabetes",
        positive_value=1,
        drop_columns=[],
        symbolic_features=[
            "age", "hypertension", "heart_disease", "bmi",
            "HbA1c_level", "blood_glucose_level",
        ],
    ),
}


# ============================================================
# UTILITIES
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_wilcoxon(a, b) -> Tuple[float, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) == 0:
        return np.nan, np.nan
    if np.allclose(a, b):
        return 0.0, 1.0
    try:
        stat, p = wilcoxon(a, b, alternative="two-sided")
        return float(stat), float(p)
    except Exception:
        return np.nan, np.nan


def significance_label(p: float) -> str:
    if not np.isfinite(p):
        return "NA"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def expected_calibration_error(y_true, prob, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true, dtype=int)
    prob = np.asarray(prob, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (prob >= bins[i]) & (prob <= bins[i + 1])
        else:
            mask = (prob >= bins[i]) & (prob < bins[i + 1])
        if not np.any(mask):
            continue
        confidence = float(np.mean(prob[mask]))
        accuracy = float(np.mean(y_true[mask]))
        ece += (np.sum(mask) / len(y_true)) * abs(accuracy - confidence)
    return float(ece)


def calculate_metrics(y_true, prob, threshold: float = 0.5) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    prob = np.clip(np.asarray(prob, dtype=float), 0.0, 1.0)
    pred = (prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn + 1e-12)
    specificity = tn / (tn + fp + 1e-12)

    return {
        "AUROC": float(roc_auc_score(y_true, prob)),
        "Macro_F1": float(f1_score(y_true, pred, average="macro")),
        "Balanced_Accuracy": float(balanced_accuracy_score(y_true, pred)),
        "Sensitivity": float(sensitivity),
        "Specificity": float(specificity),
        "Brier": float(brier_score_loss(y_true, prob)),
        "ECE": float(expected_calibration_error(y_true, prob)),
    }


def paired_effect_size(diff: np.ndarray) -> float:
    """Paired standardized mean difference dz."""
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    if len(diff) < 2:
        return np.nan
    sd = np.std(diff, ddof=1)
    if sd < 1e-12:
        return np.nan
    return float(np.mean(diff) / sd)


def bootstrap_ci(values, seed=12345, n_boot=4000, alpha=0.05):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        means.append(np.mean(sample))
    lo = np.quantile(means, alpha / 2)
    hi = np.quantile(means, 1 - alpha / 2)
    return float(lo), float(hi)


# ============================================================
# DATA LOADING
# ============================================================

def locate_file(data_dir: Path, filename: str) -> Path:
    p = data_dir / filename
    if p.exists():
        return p

    # Also support cleaner names if user renames files.
    alternatives = {
        "breast_cancer_enhanced_dataset(1).csv": [
            "breast_cancer_enhanced_dataset.csv",
        ],
        "cardio_train(1).csv": ["cardio_train.csv"],
        "diabetes_prediction_dataset(1).csv": [
            "diabetes_prediction_dataset.csv",
        ],
    }
    for alt in alternatives.get(filename, []):
        q = data_dir / alt
        if q.exists():
            return q

    raise FileNotFoundError(
        f"\nCould not find dataset:\n  {p}\n"
        f"Put the CSV files in the same folder as this script, "
        f"or pass --data-dir /path/to/folder\n"
    )


def load_dataset(spec: DatasetSpec, data_dir: Path, max_samples: Optional[int], seed=42):
    path = locate_file(data_dir, spec.filename)

    # --------------------------------------------------------
    # Robust CSV loading
    # --------------------------------------------------------
    # The cardiovascular dataset is commonly distributed as a
    # semicolon-separated CSV (e.g., "id;age;gender;...;cardio").
    # Other datasets here are comma-separated.  Auto-detect the
    # delimiter first, then fall back to explicit comma/semicolon
    # parsing if needed.
    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(path)

    # If pandas still read the whole header as one column,
    # retry with the delimiter visible in that header.
    if len(df.columns) == 1:
        only_col = str(df.columns[0])
        if ";" in only_col:
            df = pd.read_csv(path, sep=";")
        elif "," in only_col:
            df = pd.read_csv(path, sep=",")

    # Remove accidental whitespace around column names.
    df.columns = [str(c).strip() for c in df.columns]

    if spec.target not in df.columns:
        raise ValueError(
            f"{spec.name}: target '{spec.target}' not found. "
            f"Columns are: {df.columns.tolist()}"
        )

    # Remove duplicate rows if exact duplicates exist.
    df = df.drop_duplicates().reset_index(drop=True)

    # Optional quick-mode cap. Stratified by target.
    if max_samples is not None and len(df) > max_samples:
        y_temp = (df[spec.target] == spec.positive_value).astype(int)
        keep, _ = train_test_split(
            np.arange(len(df)),
            train_size=max_samples,
            stratify=y_temp,
            random_state=seed,
        )
        df = df.iloc[keep].reset_index(drop=True)

    y = (df[spec.target] == spec.positive_value).astype(int)

    drop_cols = [spec.target] + [c for c in spec.drop_columns if c in df.columns]
    X = df.drop(columns=drop_cols).copy()

    # Keep raw clinical representation. Categorical encoding occurs later.
    return X, y, path


def dataset_summary_row(name, X, y, path):
    return {
        "Dataset": name,
        "File": path.name,
        "Samples": len(X),
        "Features": X.shape[1],
        "Positive_n": int(y.sum()),
        "Negative_n": int((1 - y).sum()),
        "Positive_rate": float(y.mean()),
        "Missing_values": int(X.isna().sum().sum()),
    }


# ============================================================
# TRAIN / VALIDATION / TEST AND PREPROCESSING
# ============================================================

def create_fixed_split(X, y, seed):
    """
    Same 70/10/20 overall split pattern as the Colab code:
    80% train+validation / 20% test,
    then 12.5% of train+validation becomes validation.
    """
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=0.20,
        stratify=y,
        random_state=seed,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=0.125,
        stratify=y_trainval,
        random_state=seed,
    )

    return (
        X_train.copy(), X_val.copy(), X_test.copy(),
        y_train.copy(), y_val.copy(), y_test.copy()
    )


def make_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    num_cols = X_train.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    cat_cols = [c for c in X_train.columns if c not in num_cols]

    transformers = []

    if num_cols:
        numeric_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", numeric_pipe, num_cols))

    if cat_cols:
        categorical_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("cat", categorical_pipe, cat_cols))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.0,
    )


def stratified_subsample(X, y, fraction, seed):
    if fraction >= 0.999999:
        return X.copy(), y.copy()

    n = max(2, int(round(len(X) * fraction)))
    # Must keep both classes.
    n = max(n, 2 * y.nunique())
    n = min(n, len(X))

    X_sub, _, y_sub, _ = train_test_split(
        X, y,
        train_size=n,
        stratify=y,
        random_state=seed + int(fraction * 1000),
    )
    return X_sub.copy(), y_sub.copy()


def build_snn(seed, quick=False):
    return MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size="auto",
        learning_rate_init=1e-3,
        max_iter=220 if quick else 500,
        early_stopping=True,
        validation_fraction=0.12,
        n_iter_no_change=12,
        random_state=seed,
    )


# ============================================================
# DATASET-SPECIFIC SYMBOLIC KNOWLEDGE
# ============================================================

def _series(X, col):
    if col not in X.columns:
        return pd.Series(np.nan, index=X.index)
    return X[col]


def _rule_frame(X, rules: Dict[str, Tuple[pd.Series, int]]):
    """
    rules[name] = (condition, direction)
    direction +1 supports positive class; -1 supports negative class.
    """
    rm = pd.DataFrame(0, index=X.index, columns=list(rules), dtype=int)
    dm = pd.DataFrame(0.0, index=X.index, columns=list(rules), dtype=float)

    for name, (condition, direction) in rules.items():
        condition = condition.fillna(False)
        rm.loc[condition, name] = 1
        dm.loc[condition, name] = float(direction)

    return rm, dm


def breast_cancer_rules(X):
    """
    Refined WDBC-style morphology rules.

    The earlier rules used several broad single-feature triggers.
    These updated rules place more emphasis on clinically plausible
    combinations, so the symbolic signal is sharper and less noisy.
    """
    radius = pd.to_numeric(_series(X, "radius_mean"), errors="coerce")
    texture = pd.to_numeric(_series(X, "texture_mean"), errors="coerce")
    compact = pd.to_numeric(_series(X, "compactness_mean"), errors="coerce")
    concavity = pd.to_numeric(_series(X, "concavity_mean"), errors="coerce")
    points = pd.to_numeric(_series(X, "concave points_mean"), errors="coerce")
    smooth = pd.to_numeric(_series(X, "smoothness_mean"), errors="coerce")

    rules = {
        "R1_very_large_radius": (radius >= 17.5, +1),
        "R2_high_concavity": (concavity >= 0.16, +1),
        "R3_high_concave_points": (points >= 0.085, +1),
        "R4_compact_concave_combo": (
            (compact >= 0.14) & (concavity >= 0.12), +1
        ),
        "R5_large_irregular_combo": (
            (radius >= 15.5) & (compact >= 0.12) & (points >= 0.06), +1
        ),
        "R6_texture_irregular_combo": (
            (texture >= 23.0) & (concavity >= 0.10), +1
        ),
        "R7_benign_low_irregularity": (
            (radius < 13.0) & (concavity < 0.05) & (points < 0.03), -1
        ),
        "R8_benign_low_compactness": (
            (compact < 0.08) & (points < 0.035), -1
        ),
        "R9_benign_small_smooth": (
            (radius < 14.0) & (smooth < 0.10) & (concavity < 0.06), -1
        ),
        "R10_benign_small_texture": (
            (radius < 13.5) & (texture < 20.0) & (points < 0.04), -1
        ),
    }
    return _rule_frame(X, rules)



def cardiovascular_rules(X):
    """
    Refined cardiovascular-risk rules.

    These rules reduce noisy one-factor voting and emphasize stronger
    combinations of blood pressure, metabolic markers, age, and lifestyle.
    """
    age_years = pd.to_numeric(_series(X, "age"), errors="coerce") / 365.25
    height_m = pd.to_numeric(_series(X, "height"), errors="coerce") / 100.0
    weight = pd.to_numeric(_series(X, "weight"), errors="coerce")
    bmi = weight / (height_m.pow(2))
    sbp = pd.to_numeric(_series(X, "ap_hi"), errors="coerce")
    dbp = pd.to_numeric(_series(X, "ap_lo"), errors="coerce")
    chol = pd.to_numeric(_series(X, "cholesterol"), errors="coerce")
    gluc = pd.to_numeric(_series(X, "gluc"), errors="coerce")
    smoke = pd.to_numeric(_series(X, "smoke"), errors="coerce")
    active = pd.to_numeric(_series(X, "active"), errors="coerce")
    pulse_pressure = sbp - dbp

    rules = {
        "R1_stage2_hypertension": ((sbp >= 160) | (dbp >= 100), +1),
        "R2_hypertension_metabolic": (
            ((sbp >= 140) | (dbp >= 90)) & ((chol >= 2) | (gluc >= 2)), +1
        ),
        "R3_obesity_metabolic": (
            (bmi >= 30.0) & ((chol >= 2) | (gluc >= 2)), +1
        ),
        "R4_older_hypertensive": (
            (age_years >= 60) & ((sbp >= 140) | (dbp >= 90)), +1
        ),
        "R5_smoker_hypertensive": (
            (smoke == 1) & ((sbp >= 140) | (dbp >= 90)), +1
        ),
        "R6_wide_pulse_pressure": (
            (pulse_pressure >= 60) & (age_years >= 50), +1
        ),
        "R7_triple_metabolic": (
            (chol >= 3) & (gluc >= 2) & (bmi >= 28.0), +1
        ),
        "R8_lowrisk_young_active": (
            (age_years < 45) & (sbp < 130) & (dbp < 85) & (active == 1), -1
        ),
        "R9_lowrisk_normal_metabolic": (
            (chol == 1) & (gluc == 1) & (bmi < 25.0) & (active == 1), -1
        ),
        "R10_lowrisk_all_normal": (
            (sbp < 120) & (dbp < 80) & (chol == 1) & (gluc == 1), -1
        ),
    }
    return _rule_frame(X, rules)



def diabetes_rules(X):
    """
    Refined diabetes-risk rules.

    The earlier rule set over-fired on weak single risk factors.
    This version gives more weight to glycemic evidence and uses
    cleaner low-risk profiles for negative support.
    """
    age = pd.to_numeric(_series(X, "age"), errors="coerce")
    htn = pd.to_numeric(_series(X, "hypertension"), errors="coerce")
    hd = pd.to_numeric(_series(X, "heart_disease"), errors="coerce")
    bmi = pd.to_numeric(_series(X, "bmi"), errors="coerce")
    a1c = pd.to_numeric(_series(X, "HbA1c_level"), errors="coerce")
    glucose = pd.to_numeric(_series(X, "blood_glucose_level"), errors="coerce")

    rules = {
        "R1_diabetic_a1c": (a1c >= 6.5, +1),
        "R2_diabetic_glucose": (glucose >= 200, +1),
        "R3_glycemic_combo": ((a1c >= 6.0) & (glucose >= 140), +1),
        "R4_prediabetes_plus_obesity": (
            (a1c >= 5.7) & (glucose >= 126) & (bmi >= 30.0), +1
        ),
        "R5_obesity_hypertension_age": (
            (bmi >= 30.0) & (htn == 1) & (age >= 45), +1
        ),
        "R6_cardiometabolic_glucose": (
            ((htn == 1) | (hd == 1)) & (glucose >= 140), +1
        ),
        "R7_multirisk_older": (
            (age >= 50) & (bmi >= 30.0) & ((htn == 1) | (hd == 1)), +1
        ),
        "R8_normal_glycemia": ((a1c < 5.7) & (glucose < 100), -1),
        "R9_lean_normal_profile": (
            (bmi < 25.0) & (a1c < 5.7) & (glucose < 100), -1
        ),
        "R10_young_healthy_profile": (
            (age < 40) & (bmi < 27.0) & (htn == 0) & (hd == 0) & (glucose < 100), -1
        ),
    }
    return _rule_frame(X, rules)



def evaluate_symbolic_rules(dataset_name, X_raw):
    if dataset_name == "Breast Cancer":
        return breast_cancer_rules(X_raw)
    if dataset_name == "Cardiovascular":
        return cardiovascular_rules(X_raw)
    if dataset_name == "Diabetes":
        return diabetes_rules(X_raw)
    raise KeyError(dataset_name)


def rule_names(dataset_name):
    dummy = pd.DataFrame(index=[0])
    # Build column names safely with all required columns.
    if dataset_name == "Breast Cancer":
        dummy = pd.DataFrame({
            c: [np.nan] for c in DATASETS[dataset_name].symbolic_features
        })
    elif dataset_name == "Cardiovascular":
        dummy = pd.DataFrame({
            c: [np.nan] for c in DATASETS[dataset_name].symbolic_features
        })
    else:
        dummy = pd.DataFrame({
            c: [np.nan] for c in DATASETS[dataset_name].symbolic_features
        })
    rm, _ = evaluate_symbolic_rules(dataset_name, dummy)
    return rm.columns.tolist()



def estimate_rule_weights(dataset_name, X_train_raw, y_train):
    """
    Estimate rule reliability using TRAINING DATA ONLY.

    Each rule keeps its symbolic direction (+1 or -1), but its contribution
    is down-weighted when that rule is unreliable on the current training
    subset. A Beta(2,2) prior shrinks low-support rules toward neutral.

    This is deliberately training-only to avoid test-label leakage.
    """
    _, dm = evaluate_symbolic_rules(dataset_name, X_train_raw)
    y = np.asarray(y_train, dtype=int)

    weights = {}
    diagnostics = []

    for rule in dm.columns:
        vals = dm[rule].to_numpy(dtype=float)
        active = np.abs(vals) > 0

        support = int(active.sum())
        if support == 0:
            reliability = 0.5
            weight = 0.20
        else:
            symbolic_pred = (vals[active] > 0).astype(int)
            correct = int(np.sum(symbolic_pred == y[active]))

            # Smoothed reliability. Beta(2,2) prior prevents unstable
            # weights when scarcity leaves only a few active examples.
            reliability = (correct + 2.0) / (support + 4.0)

            # Convert reliability to a conservative positive weight.
            # 0.5 reliability -> 0.20
            # 0.75 reliability -> ~0.85
            # 1.0 reliability -> 1.50
            weight = 0.20 + 1.30 * max(
                0.0, min(1.0, 2.0 * (reliability - 0.5))
            )

        weights[rule] = float(weight)
        diagnostics.append({
            "Rule": rule,
            "Support": support,
            "Reliability": float(reliability),
            "Weight": float(weight),
        })

    return weights, pd.DataFrame(diagnostics)


def apply_rule_weights(direction_matrix, rule_weights):
    """
    Weight symbolic directions without changing their logical direction.
    """
    dm = direction_matrix.copy()
    if rule_weights is None:
        return dm

    for rule in dm.columns:
        dm[rule] = dm[rule].astype(float) * float(
            rule_weights.get(rule, 1.0)
        )
    return dm


def calculate_symbolic_score(direction_matrix):
    signed_sum = direction_matrix.sum(axis=1).to_numpy(dtype=float)
    active_count = direction_matrix.abs().sum(axis=1).to_numpy(dtype=float)
    score = np.zeros(len(direction_matrix), dtype=float)
    active = active_count > 0
    score[active] = signed_sum[active] / active_count[active]
    score = np.clip(score, -1.0, 1.0)
    return score, active


def symbolic_probability(symbolic_score, beta=2.0):
    symbolic_score = np.asarray(symbolic_score, dtype=float)
    return 1.0 / (1.0 + np.exp(-beta * symbolic_score))


def symbolic_reasoning(dataset_name, X_raw, beta=2.0, rule_weights=None):
    rule_matrix, direction_matrix = evaluate_symbolic_rules(dataset_name, X_raw)
    direction_matrix = apply_rule_weights(direction_matrix, rule_weights)
    score, active = calculate_symbolic_score(direction_matrix)
    prob = symbolic_probability(score, beta=beta)
    prob[~active] = 0.5
    return {
        "probability": prob,
        "score": score,
        "active": active,
        "rule_matrix": rule_matrix,
        "direction_matrix": direction_matrix,
    }


def isage_fusion(neural_prob, symbolic_prob, symbolic_active, alpha=0.7):
    neural_prob = np.asarray(neural_prob, dtype=float)
    symbolic_prob = np.asarray(symbolic_prob, dtype=float)
    active = np.asarray(symbolic_active, dtype=bool)

    final_prob = neural_prob.copy()
    final_prob[active] = (
        alpha * neural_prob[active] +
        (1.0 - alpha) * symbolic_prob[active]
    )
    return np.clip(final_prob, 0.0, 1.0)


def calculate_rvr(probabilities, symbolic_score, symbolic_active, threshold=0.5):
    """
    Rule Violation Rate at the patient-level symbolic consensus:
    compare model class with sign of nonzero symbolic score.
    Zero/conflicting symbolic score is excluded.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    score = np.asarray(symbolic_score, dtype=float)
    active = np.asarray(symbolic_active, dtype=bool)

    valid = active & (np.abs(score) > 1e-12)
    if not np.any(valid):
        return np.nan

    predicted_class = (probabilities[valid] >= threshold).astype(int)
    symbolic_class = (score[valid] > 0).astype(int)
    return float(np.mean(predicted_class != symbolic_class))


def get_corruption_order(dataset_name, seed):
    rng = np.random.default_rng(seed + 5000)
    return list(rng.permutation(rule_names(dataset_name)))


def get_corrupted_rules(rule_order, corruption_level):
    n_corrupt = int(round(corruption_level * len(rule_order)))
    return list(rule_order[:n_corrupt])


def symbolic_reasoning_corrupted(
    dataset_name, X_raw, beta, corrupted_rules, corruption_mode,
    rule_weights=None
):
    rm, dm = evaluate_symbolic_rules(dataset_name, X_raw)
    dm = apply_rule_weights(dm, rule_weights)
    rm = rm.copy()
    dm = dm.copy()

    for rule in corrupted_rules:
        if corruption_mode == "removal":
            rm[rule] = 0
            dm[rule] = 0.0
        elif corruption_mode == "reversal":
            dm[rule] = -1.0 * dm[rule]
        else:
            raise ValueError("corruption_mode must be 'removal' or 'reversal'")

    score, active = calculate_symbolic_score(dm)
    prob = symbolic_probability(score, beta=beta)
    prob[~active] = 0.5

    return {
        "probability": prob,
        "score": score,
        "active": active,
        "rule_matrix": rm,
        "direction_matrix": dm,
    }


# ============================================================
# OBSERVATION MASKING
# ============================================================

def apply_observation_mask(X_raw, observation_fraction, seed):
    """
    Random test-time observation masking.
    Same mask realization is used for S-NN and iSAGE.
    ID/target columns have already been removed.
    """
    if observation_fraction >= 0.999999:
        return X_raw.copy(), 1.0

    rng = np.random.default_rng(seed)
    X_masked = X_raw.copy()

    # Independent cell-wise availability.
    keep = rng.random(X_masked.shape) < observation_fraction

    # Guarantee at least one observed feature per row.
    all_missing_rows = np.where(~keep.any(axis=1))[0]
    for i in all_missing_rows:
        keep[i, rng.integers(0, X_masked.shape[1])] = True

    for j, col in enumerate(X_masked.columns):
        X_masked.loc[~keep[:, j], col] = np.nan

    actual = float(np.mean(keep))
    return X_masked, actual


# ============================================================
# MODEL CACHE / TRAINING
# ============================================================

def symbolic_training_features_from_result(symbolic_result):
    """
    Convert the symbolic rule state into trainable neural input features.

    Features include:
      1) signed firing state of each symbolic rule,
      2) aggregate symbolic score,
      3) symbolic-rule coverage / active fraction.

    These features are concatenated with the ordinary preprocessed clinical
    features BEFORE fitting the iSAGE MLP. Therefore symbolic knowledge is
    part of neural training rather than being fused only after training.
    """
    dm = symbolic_result["direction_matrix"].to_numpy(dtype=float)
    score = np.asarray(symbolic_result["score"], dtype=float).reshape(-1, 1)

    # Fraction of rules active for each sample.
    active_fraction = (
        np.abs(dm).sum(axis=1, keepdims=True) /
        max(1, dm.shape[1])
    )

    return np.hstack([dm, score, active_fraction])


def symbolic_training_features(dataset_name, X_raw, rule_weights=None):
    sym = symbolic_reasoning(
        dataset_name, X_raw, beta=2.0, rule_weights=rule_weights
    )
    return symbolic_training_features_from_result(sym), sym


class ExperimentCache:
    def __init__(self, quick=False):
        self.quick = quick
        self.snn_cache = {}
        self.isage_cache = {}

    def fit_snn(self, dataset_name, seed, fraction, X_train, y_train):
        """
        Standard neural baseline. No symbolic features are used.
        """
        key = (dataset_name, seed, round(float(fraction), 4))
        if key in self.snn_cache:
            return self.snn_cache[key]

        X_sub, y_sub = stratified_subsample(
            X_train, y_train, fraction, seed
        )
        preprocessor = make_preprocessor(X_sub)
        Xt = preprocessor.fit_transform(X_sub)
        Xt = np.asarray(Xt, dtype=float)

        model = build_snn(seed, quick=self.quick)
        model.fit(Xt, y_sub)

        obj = {
            "preprocessor": preprocessor,
            "model": model,
            "training_samples": len(X_sub),
            "X_sub": X_sub,
            "y_sub": y_sub,
        }
        self.snn_cache[key] = obj
        return obj

    def fit_isage(self, dataset_name, seed, fraction, X_train, y_train):
        """
        Knowledge-aware neural training.

        iSAGE-Train uses the SAME MLP architecture and optimizer settings as S-NN.
        The difference is that dataset-specific symbolic rule states are
        appended to the neural input during training.
        """
        key = (dataset_name, seed, round(float(fraction), 4))
        if key in self.isage_cache:
            return self.isage_cache[key]

        # Use exactly the same deterministic scarcity subset as S-NN.
        X_sub, y_sub = stratified_subsample(
            X_train, y_train, fraction, seed
        )

        preprocessor = make_preprocessor(X_sub)
        Xt_clinical = preprocessor.fit_transform(X_sub)
        Xt_clinical = np.asarray(Xt_clinical, dtype=float)

        # Estimate symbolic-rule reliability on this TRAINING SUBSET ONLY.
        # This suppresses harmful/noisy rules without using validation/test labels.
        rule_weights, rule_diagnostics = estimate_rule_weights(
            dataset_name, X_sub, y_sub
        )

        Xt_symbolic, train_sym = symbolic_training_features(
            dataset_name, X_sub, rule_weights=rule_weights
        )

        # Keep symbolic inputs on a comparable scale with the standardized
        # clinical inputs. The bounded weighted rule features already lie
        # approximately in [-1.5, 1.5].
        Xt = np.hstack([Xt_clinical, Xt_symbolic])

        # Same neural backbone as S-NN.
        model = build_snn(seed, quick=self.quick)
        model.fit(Xt, y_sub)

        obj = {
            "preprocessor": preprocessor,
            "model": model,
            "training_samples": len(X_sub),
            "X_sub": X_sub,
            "y_sub": y_sub,
            "n_clinical_features": Xt_clinical.shape[1],
            "n_symbolic_features": Xt_symbolic.shape[1],
            "training_symbolic_coverage": float(
                np.mean(train_sym["active"])
            ),
            "rule_weights": rule_weights,
            "rule_diagnostics": rule_diagnostics,
        }
        self.isage_cache[key] = obj
        return obj


def predict_model(model_obj, X_raw):
    """
    Predict with the standard S-NN baseline.
    """
    Xt = model_obj["preprocessor"].transform(X_raw)
    Xt = np.asarray(Xt, dtype=float)
    return model_obj["model"].predict_proba(Xt)[:, 1]


def predict_isage(model_obj, dataset_name, X_raw):
    """
    Predict with iSAGE-Train using the same kind of symbolic features that were
    available during iSAGE neural training.
    """
    Xt_clinical = model_obj["preprocessor"].transform(X_raw)
    Xt_clinical = np.asarray(Xt_clinical, dtype=float)
    Xt_symbolic, sym = symbolic_training_features(
        dataset_name,
        X_raw,
        rule_weights=model_obj.get("rule_weights"),
    )
    Xt = np.hstack([Xt_clinical, Xt_symbolic])
    prob = model_obj["model"].predict_proba(Xt)[:, 1]
    return prob, sym


def predict_isage_from_symbolic_result(
    model_obj, X_raw, symbolic_result
):
    """
    Predict with a supplied symbolic state. This is used for the knowledge
    corruption experiment so the trained iSAGE model is evaluated with
    removed/reversed rules without post-hoc probability fusion.
    """
    Xt_clinical = model_obj["preprocessor"].transform(X_raw)
    Xt_clinical = np.asarray(Xt_clinical, dtype=float)
    Xt_symbolic = symbolic_training_features_from_result(symbolic_result)
    Xt = np.hstack([Xt_clinical, Xt_symbolic])
    return model_obj["model"].predict_proba(Xt)[:, 1]


# NOTE:
# The previous implementation tuned alpha/beta and fused symbolic
# probabilities after S-NN training. That is intentionally no longer the
# primary iSAGE mechanism. iSAGE is now trained directly with symbolic
# features. beta=2.0 is retained only for the Rule-Only reference model.
# ============================================================
# POST-TRAINING SYMBOLIC FUSION
# ============================================================

def tune_post_training_fusion(
    y_val,
    neural_val_prob,
    symbolic_score,
    symbolic_active,
):
    """
    Tune post-training symbolic fusion on the validation set only.

    S-NN is trained first without symbolic inputs. Symbolic knowledge is then
    added to the trained neural probabilities. alpha controls the neural
    contribution and beta controls symbolic probability sharpness.
    """
    best = {
        "alpha": ALPHA_GRID[0],
        "beta": BETA_GRID[0],
        "val_auc": -np.inf,
    }

    for beta in BETA_GRID:
        sym_prob = symbolic_probability(symbolic_score, beta=beta)
        sym_prob = np.asarray(sym_prob, dtype=float)
        sym_prob[~np.asarray(symbolic_active, dtype=bool)] = 0.5

        for alpha in ALPHA_GRID:
            fused = isage_fusion(
                neural_val_prob,
                sym_prob,
                symbolic_active,
                alpha=alpha,
            )
            auc = calculate_metrics(y_val, fused)["AUROC"]

            if np.isfinite(auc) and auc > best["val_auc"]:
                best = {
                    "alpha": float(alpha),
                    "beta": float(beta),
                    "val_auc": float(auc),
                }

    return best


# ============================================================
# ONE-SEED / ONE-FRACTION EVALUATION
# ============================================================

def evaluate_fraction(
    dataset_name, seed, fraction,
    X_train, y_train, X_val, y_val, X_test, y_test,
    cache: ExperimentCache,
    observation_fraction=1.0,
    mask_seed_offset=0,
):
    """
    Three-way comparison using the SAME neural backbone:

      1) S-NN:
         Neural model trained without symbolic knowledge.

      2) iSAGE-Train:
         Symbolic knowledge is converted to reliability-weighted symbolic
         features and appended BEFORE neural training.

      3) iSAGE:
         The S-NN is trained first without knowledge. Symbolic knowledge is
         added AFTER training by validation-tuned probability fusion.

    This makes the timing of knowledge injection explicit and comparable.
    """

    # --------------------------------------------------------
    # Train neural-only and knowledge-before-training models.
    # --------------------------------------------------------
    snn_obj = cache.fit_snn(
        dataset_name, seed, fraction, X_train, y_train
    )
    pre_obj = cache.fit_isage(
        dataset_name, seed, fraction, X_train, y_train
    )

    # --------------------------------------------------------
    # Tune the post-training symbolic fusion on VALIDATION only.
    # The symbolic rule weights were learned from TRAINING data only.
    # --------------------------------------------------------
    snn_val_prob = predict_model(snn_obj, X_val)
    symbolic_val = symbolic_reasoning(
        dataset_name,
        X_val,
        beta=2.0,
        rule_weights=pre_obj.get("rule_weights"),
    )

    post_tuned = tune_post_training_fusion(
        y_val,
        snn_val_prob,
        symbolic_val["score"],
        symbolic_val["active"],
    )

    # --------------------------------------------------------
    # Apply observation scarcity to the SAME test input for all models.
    # --------------------------------------------------------
    X_test_eval, actual_obs = apply_observation_mask(
        X_test,
        observation_fraction,
        seed=seed + 10000 + mask_seed_offset,
    )

    # 1) Neural only
    snn_prob = predict_model(snn_obj, X_test_eval)

    # 2) Knowledge added BEFORE neural training
    pre_prob, symbolic_test = predict_isage(
        pre_obj, dataset_name, X_test_eval
    )

    # 3) Knowledge added AFTER S-NN training
    post_symbolic = symbolic_reasoning(
        dataset_name,
        X_test_eval,
        beta=post_tuned["beta"],
        rule_weights=pre_obj.get("rule_weights"),
    )
    post_sym_prob = np.asarray(
        post_symbolic["probability"], dtype=float
    )
    post_prob = isage_fusion(
        snn_prob,
        post_sym_prob,
        post_symbolic["active"],
        alpha=post_tuned["alpha"],
    )

    # Rule-only reference
    rule_prob = symbolic_probability(
        symbolic_test["score"], beta=2.0
    )
    rule_prob[~symbolic_test["active"]] = 0.5

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------
    snn_metrics = calculate_metrics(y_test, snn_prob)
    pre_metrics = calculate_metrics(y_test, pre_prob)
    post_metrics = calculate_metrics(y_test, post_prob)
    rule_metrics = calculate_metrics(y_test, rule_prob)

    snn_rvr = calculate_rvr(
        snn_prob,
        symbolic_test["score"],
        symbolic_test["active"],
    )
    pre_rvr = calculate_rvr(
        pre_prob,
        symbolic_test["score"],
        symbolic_test["active"],
    )
    post_rvr = calculate_rvr(
        post_prob,
        post_symbolic["score"],
        post_symbolic["active"],
    )
    rule_rvr = calculate_rvr(
        rule_prob,
        symbolic_test["score"],
        symbolic_test["active"],
    )

    common = {
        "Dataset": dataset_name,
        "Seed": seed,
        "DataFraction": fraction,
        "TrainingPercent": fraction * 100,
        "TrainingSamples": snn_obj["training_samples"],
        "ObservationFraction": observation_fraction,
        "ObservationPercent": observation_fraction * 100,
        "ActualObservationPercent": actual_obs * 100,
        "Best_alpha": post_tuned["alpha"],
        "Best_beta": post_tuned["beta"],
        "PostFusionValidationAUROC": post_tuned["val_auc"],
        "SymbolicCoverage": float(
            np.mean(symbolic_test["active"])
        ),
    }

    rows = [
        {
            **common,
            "Model": "S-NN",
            "KnowledgeTiming": "None",
            **snn_metrics,
            "RVR": snn_rvr,
        },
        {
            **common,
            "Model": "Rule-Only",
            "KnowledgeTiming": "Symbolic only",
            **rule_metrics,
            "RVR": rule_rvr,
        },
        {
            **common,
            "Model": "iSAGE-Train",
            "KnowledgeTiming": "Before neural training",
            **pre_metrics,
            "RVR": pre_rvr,
        },
        {
            **common,
            "Model": "iSAGE-Fusion",
            "KnowledgeTiming": "After neural training",
            **post_metrics,
            "RVR": post_rvr,
        },
    ]

    extras = {
        "snn_obj": snn_obj,
        "isage_obj": pre_obj,
        "pre_obj": pre_obj,
        "model_obj": snn_obj,
        "post_tuned": post_tuned,
        "tuned": post_tuned,
        "snn_prob": snn_prob,
        "isage_prob": pre_prob,
        "pre_prob": pre_prob,
        "post_prob": post_prob,
        "rule_prob": rule_prob,
        "symbolic": symbolic_test,
        "post_symbolic": post_symbolic,
        "X_test_eval": X_test_eval,
        "actual_obs": actual_obs,
    }
    return rows, extras


# ============================================================
# FULL EXPERIMENTS
# ============================================================

def run_dataset_experiments(
    dataset_name, X, y, seeds, cache, outdir
):
    ds_dir = outdir / dataset_name.replace(" ", "_")
    ensure_dir(ds_dir)

    scarcity_rows = []
    observation_rows = []
    knowledge_rows = []
    joint_rows = []
    compute_rows = []
    full_resource_rows = []

    for seed in seeds:
        print(f"\n[{dataset_name}] seed {seed}")
        set_seed(seed)

        X_train, X_val, X_test, y_train, y_val, y_test = create_fixed_split(
            X, y, seed
        )

        # ----------------------------------------------------
        # A. DATA SCARCITY + RULE-ONLY COMPLEMENTARITY
        # ----------------------------------------------------
        full_extras = None
        for frac in DATA_FRACTIONS:
            rows, extras = evaluate_fraction(
                dataset_name, seed, frac,
                X_train, y_train, X_val, y_val, X_test, y_test,
                cache=cache,
                observation_fraction=1.0,
            )
            scarcity_rows.extend(rows)

            if frac == 1.0:
                full_resource_rows.extend(rows)
                full_extras = extras

        # ----------------------------------------------------
        # B. OBSERVATION SCARCITY
        # Training remains 100%.
        # ----------------------------------------------------
        for obs in OBSERVATION_FRACTIONS:
            rows, _ = evaluate_fraction(
                dataset_name, seed, 1.0,
                X_train, y_train, X_val, y_val, X_test, y_test,
                cache=cache,
                observation_fraction=obs,
                mask_seed_offset=int(obs * 1000),
            )
            # Main comparison is S-NN and iSAGE. Keep Rule-Only too
            # for supplementary analysis.
            observation_rows.extend(rows)

        # ----------------------------------------------------
        # C. KNOWLEDGE CORRUPTION
        # Use unchanged full-data S-NN and tuned hyperparameters.
        # ----------------------------------------------------
        if full_extras is None:
            raise RuntimeError("Full-resource model was not created.")

        snn_obj = full_extras["snn_obj"]
        isage_obj = full_extras["isage_obj"]

        # Save learned symbolic-rule reliability for auditability.
        if "rule_diagnostics" in isage_obj:
            isage_obj["rule_diagnostics"].to_csv(
                ds_dir / f"rule_reliability_seed{seed}.csv",
                index=False,
            )
        beta = 2.0
        snn_prob = predict_model(snn_obj, X_test)
        snn_metrics = calculate_metrics(y_test, snn_prob)

        rule_order = get_corruption_order(dataset_name, seed)

        for mode in CORRUPTION_MODES:
            for q in CORRUPTION_LEVELS:
                corrupted_rules = get_corrupted_rules(
                    rule_order, q
                )
                sym = symbolic_reasoning_corrupted(
                    dataset_name,
                    X_test,
                    beta=beta,
                    corrupted_rules=corrupted_rules,
                    corruption_mode=mode,
                    rule_weights=isage_obj.get("rule_weights"),
                )
                # iSAGE-Train under corrupted symbolic inputs.
                # This is an auxiliary robustness analysis because the model
                # was trained with clean symbolic features.
                train_prob = predict_isage_from_symbolic_result(
                    isage_obj,
                    X_test,
                    sym,
                )
                train_metrics = calculate_metrics(y_test, train_prob)

                # iSAGE-Fusion: the neural model remains unchanged.
                # Only the symbolic branch is corrupted, matching the
                # knowledge-quality experiment described in the manuscript.
                post_tuned = full_extras["post_tuned"]
                sym_fusion_prob = symbolic_probability(
                    sym["score"], beta=post_tuned["beta"]
                )
                sym_fusion_prob = np.asarray(sym_fusion_prob, dtype=float)
                sym_fusion_prob[~sym["active"]] = 0.5
                fusion_prob = isage_fusion(
                    snn_prob,
                    sym_fusion_prob,
                    sym["active"],
                    alpha=post_tuned["alpha"],
                )
                fusion_metrics = calculate_metrics(y_test, fusion_prob)

                snn_rvr = calculate_rvr(
                    snn_prob, sym["score"], sym["active"]
                )
                train_rvr = calculate_rvr(
                    train_prob, sym["score"], sym["active"]
                )
                fusion_rvr = calculate_rvr(
                    fusion_prob, sym["score"], sym["active"]
                )

                base = {
                    "Dataset": dataset_name,
                    "Seed": seed,
                    "Mode": mode,
                    "CorruptionLevel": q,
                    "CorruptionPercent": q * 100,
                    "N_Corrupted": len(corrupted_rules),
                    "Best_alpha": post_tuned["alpha"],
                    "Best_beta": post_tuned["beta"],
                    "SymbolicCoverage": float(np.mean(sym["active"])),
                }
                knowledge_rows.append({
                    **base, "Model": "S-NN",
                    **snn_metrics, "RVR": snn_rvr,
                })
                knowledge_rows.append({
                    **base, "Model": "iSAGE-Train",
                    **train_metrics, "RVR": train_rvr,
                })
                knowledge_rows.append({
                    **base, "Model": "iSAGE-Fusion",
                    **fusion_metrics, "RVR": fusion_rvr,
                })

        # ----------------------------------------------------
        # D. JOINT RESOURCE DEGRADATION
        # ----------------------------------------------------
        for cond, cfg in RESOURCE_CONDITIONS.items():
            rows, _ = evaluate_fraction(
                dataset_name, seed, cfg["data_fraction"],
                X_train, y_train, X_val, y_val, X_test, y_test,
                cache=cache,
                observation_fraction=cfg["observation_fraction"],
                mask_seed_offset=20000 + int(cfg["observation_fraction"] * 1000),
            )
            for row in rows:
                if row["Model"] not in ["S-NN", "iSAGE-Train", "iSAGE-Fusion"]:
                    continue
                joint_rows.append({
                    **row,
                    "Condition": cond,
                })

        # ----------------------------------------------------
        # E. COMPUTE PROFILE (full-data S-NN vs iSAGE)
        # ----------------------------------------------------
        snn_obj = full_extras["snn_obj"]
        isage_obj = full_extras["isage_obj"]

        repeats = 10

        # S-NN latency: preprocessing + neural inference
        snn_times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            _ = predict_model(snn_obj, X_test)
            snn_times.append(time.perf_counter() - t0)

        # iSAGE latency: preprocessing + symbolic feature construction
        # + knowledge-aware neural inference
        care_times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            _ = predict_isage(isage_obj, dataset_name, X_test)
            care_times.append(time.perf_counter() - t0)

        # Serialized sizes
        snn_tmp = ds_dir / f"_tmp_snn_seed{seed}.joblib"
        care_tmp = ds_dir / f"_tmp_isage_seed{seed}.joblib"

        joblib.dump(
            {
                "preprocessor": snn_obj["preprocessor"],
                "model": snn_obj["model"],
            },
            snn_tmp,
            compress=3,
        )
        joblib.dump(
            {
                "preprocessor": isage_obj["preprocessor"],
                "model": isage_obj["model"],
                "n_symbolic_features": isage_obj["n_symbolic_features"],
            },
            care_tmp,
            compress=3,
        )

        snn_size_mb = snn_tmp.stat().st_size / (1024 ** 2)
        care_size_mb = care_tmp.stat().st_size / (1024 ** 2)

        for p in [snn_tmp, care_tmp]:
            try:
                p.unlink()
            except Exception:
                pass

        snn_mlp = snn_obj["model"]
        care_mlp = isage_obj["model"]

        snn_params = int(sum(
            w.size + b.size
            for w, b in zip(snn_mlp.coefs_, snn_mlp.intercepts_)
        ))
        care_params = int(sum(
            w.size + b.size
            for w, b in zip(care_mlp.coefs_, care_mlp.intercepts_)
        ))

        n_test = len(X_test)
        compute_rows.extend([
            {
                "Dataset": dataset_name,
                "Seed": seed,
                "Model": "S-NN",
                "Parameters": snn_params,
                "SerializedModelMB": snn_size_mb,
                "BatchSamples": n_test,
                "TotalLatencyMs_mean": 1000 * np.mean(snn_times),
                "LatencyPerSampleMs": 1000 * np.mean(snn_times) / n_test,
            },
            {
                "Dataset": dataset_name,
                "Seed": seed,
                "Model": "iSAGE-Train",
                "Parameters": care_params,
                "SerializedModelMB": care_size_mb,
                "BatchSamples": n_test,
                "TotalLatencyMs_mean": 1000 * np.mean(care_times),
                "LatencyPerSampleMs": 1000 * np.mean(care_times) / n_test,
            },
        ])

    return {
        "full": pd.DataFrame(full_resource_rows),
        "scarcity": pd.DataFrame(scarcity_rows),
        "observation": pd.DataFrame(observation_rows),
        "knowledge": pd.DataFrame(knowledge_rows),
        "joint": pd.DataFrame(joint_rows),
        "compute": pd.DataFrame(compute_rows),
    }


# ============================================================
# OPTIONAL CONVENTIONAL FULL-RESOURCE BASELINES
# ============================================================

def run_conventional_baselines(dataset_name, X, y, seeds, quick=False):
    rows = []

    for seed in seeds:
        X_train, X_val, X_test, y_train, y_val, y_test = create_fixed_split(
            X, y, seed
        )
        pre = make_preprocessor(X_train)
        Xt = pre.fit_transform(X_train)
        Xv = pre.transform(X_val)
        Xte = pre.transform(X_test)

        models = {
            "LR": LogisticRegression(max_iter=2000, random_state=seed),
            "RF": RandomForestClassifier(
                n_estimators=180 if quick else 300,
                max_depth=8,
                random_state=seed,
                n_jobs=-1,
            ),
            "NN": MLPClassifier(
                hidden_layer_sizes=(128, 64),
                max_iter=250 if quick else 500,
                early_stopping=True,
                random_state=seed,
            ),
            "S-NN": build_snn(seed, quick=quick),
        }

        if HAVE_XGB:
            models["GBT"] = XGBClassifier(
                n_estimators=180 if quick else 300,
                max_depth=4,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=seed,
                eval_metric="logloss",
                n_jobs=max(1, (os.cpu_count() or 2) - 1),
            )
        else:
            models["GBT"] = HistGradientBoostingClassifier(
                max_iter=180 if quick else 300,
                max_depth=4,
                learning_rate=0.05,
                random_state=seed,
            )

        for name, model in models.items():
            model.fit(Xt, y_train)
            prob = model.predict_proba(Xte)[:, 1]
            rows.append({
                "Dataset": dataset_name,
                "Seed": seed,
                "Model": name,
                **calculate_metrics(y_test, prob),
            })

    return pd.DataFrame(rows)


# ============================================================
# SUMMARIES / STATISTICS
# ============================================================

def mean_std_table(df, group_cols, metrics=None):
    if metrics is None:
        metrics = [
            "AUROC", "Macro_F1", "Balanced_Accuracy",
            "Sensitivity", "Specificity", "Brier", "ECE", "RVR",
        ]
    metrics = [m for m in metrics if m in df.columns]

    agg = {}
    for m in metrics:
        agg[f"{m}_mean"] = (m, "mean")
        agg[f"{m}_std"] = (m, "std")

    return (
        df.groupby(group_cols, dropna=False)
          .agg(**agg)
          .reset_index()
    )


def paired_comparison_stats(df, condition_col, conditions, metric="AUROC"):
    rows = []
    for dataset in df["Dataset"].unique():
        dset = df[df["Dataset"] == dataset]
        for cond in conditions:
            care = dset[
                (dset[condition_col] == cond) &
                (dset["Model"] == "iSAGE-Fusion")
            ][["Seed", metric, "RVR"]].rename(columns={
                metric: "iSAGE_metric", "RVR": "iSAGE_RVR"
            })

            snn = dset[
                (dset[condition_col] == cond) &
                (dset["Model"] == "S-NN")
            ][["Seed", metric, "RVR"]].rename(columns={
                metric: "SNN_metric", "RVR": "SNN_RVR"
            })

            merged = care.merge(snn, on="Seed")
            if merged.empty:
                continue

            diff = (
                merged["iSAGE_metric"].to_numpy() -
                merged["SNN_metric"].to_numpy()
            )
            rvr_impr = (
                merged["SNN_RVR"].to_numpy() -
                merged["iSAGE_RVR"].to_numpy()
            )
            _, p = safe_wilcoxon(
                merged["iSAGE_metric"],
                merged["SNN_metric"],
            )
            _, p_rvr = safe_wilcoxon(
                merged["iSAGE_RVR"],
                merged["SNN_RVR"],
            )
            ci_lo, ci_hi = bootstrap_ci(
                diff,
                seed=777 + int(abs(hash((dataset, str(cond)))) % 10000),
            )

            rows.append({
                "Dataset": dataset,
                condition_col: cond,
                "N_pairs": len(merged),
                f"Delta_{metric}": float(np.mean(diff)),
                f"{metric}_CI95_low": ci_lo,
                f"{metric}_CI95_high": ci_hi,
                f"{metric}_p": p,
                f"{metric}_sig": significance_label(p),
                f"{metric}_effect_dz": paired_effect_size(diff),
                "RVR_Improvement": float(np.nanmean(rvr_impr)),
                "RVR_p": p_rvr,
                "RVR_sig": significance_label(p_rvr),
            })
    return pd.DataFrame(rows)


def compute_rrs(joint_df):
    """
    RRS averages retention under constrained S2 and S3 relative to S1,
    per dataset / seed / model.
    """
    rows = []
    for (dataset, seed, model), temp in joint_df.groupby(
        ["Dataset", "Seed", "Model"]
    ):
        score = dict(zip(temp["Condition"], temp["AUROC"]))
        if not all(c in score for c in RESOURCE_CONDITIONS):
            continue
        full = score["S1_Full"]
        s2 = score["S2_Moderate"] / full
        s3 = score["S3_Severe"] / full
        rows.append({
            "Dataset": dataset,
            "Seed": seed,
            "Model": model,
            "Retention_S2": s2,
            "Retention_S3": s3,
            "RRS": float(np.mean([s2, s3])),
        })
    return pd.DataFrame(rows)


def knowledge_boundaries(knowledge_stats):
    rows = []
    for (dataset, mode), temp in knowledge_stats.groupby(["Dataset", "Mode"]):
        temp = temp.sort_values("CorruptionPercent")
        failed = temp[temp["Delta_AUROC"] <= 0]
        if len(failed):
            q = float(failed["CorruptionPercent"].iloc[0])
            text = f"{q:.0f}%"
        else:
            text = "Not observed"
        rows.append({
            "Dataset": dataset,
            "Corruption_Type": mode.capitalize(),
            "q_star": text,
        })
    return pd.DataFrame(rows)


def build_knowledge_stats(knowledge_df):
    rows = []
    for dataset in knowledge_df["Dataset"].unique():
        for mode in CORRUPTION_MODES:
            for q in sorted(knowledge_df["CorruptionPercent"].unique()):
                care = knowledge_df[
                    (knowledge_df["Dataset"] == dataset) &
                    (knowledge_df["Mode"] == mode) &
                    (knowledge_df["CorruptionPercent"] == q) &
                    (knowledge_df["Model"] == "iSAGE-Fusion")
                ][["Seed", "AUROC"]].rename(columns={"AUROC": "iSAGE-Fusion"})

                snn = knowledge_df[
                    (knowledge_df["Dataset"] == dataset) &
                    (knowledge_df["Mode"] == mode) &
                    (knowledge_df["CorruptionPercent"] == q) &
                    (knowledge_df["Model"] == "S-NN")
                ][["Seed", "AUROC"]].rename(columns={"AUROC": "SNN"})

                merged = care.merge(snn, on="Seed")
                if merged.empty:
                    continue
                diff = merged["iSAGE-Fusion"] - merged["SNN"]
                _, p = safe_wilcoxon(merged["iSAGE-Fusion"], merged["SNN"])
                rows.append({
                    "Dataset": dataset,
                    "Mode": mode,
                    "CorruptionPercent": q,
                    "iSAGE-Fusion_AUROC_mean": merged["iSAGE-Fusion"].mean(),
                    "iSAGE-Fusion_AUROC_std": merged["iSAGE-Fusion"].std(ddof=1),
                    "SNN_AUROC_mean": merged["SNN"].mean(),
                    "SNN_AUROC_std": merged["SNN"].std(ddof=1),
                    "Delta_AUROC": diff.mean(),
                    "p_value": p,
                    "Significance": significance_label(p),
                })
    return pd.DataFrame(rows)


# ============================================================
# PLOTTING HELPERS
# ============================================================

def savefig(fig, outdir, filename):
    ensure_dir(outdir)
    # Do not use tight_layout here. Fixed subplots_adjust margins in
    # _make_fig keep every plot the same width while preserving legend space.
    fig.savefig(
        outdir / f"{filename}.pdf",
        bbox_inches="tight",
        pad_inches=0.12,
    )
    fig.savefig(
        outdir / f"{filename}.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.12,
    )
    plt.close(fig)


def _mean_std(df, group_cols, metric):
    return (
        df.groupby(group_cols)[metric]
          .agg(["mean", "std"])
          .reset_index()
    )


FIG_WIDTH = 24
FIG_HEIGHT = 13.5
LEGEND_TOP_Y = 0.985
AXES_TOP = 0.78

def _make_fig(figsize=None):
    # Total figure width remains identical for every figure.
    # Extra vertical space is reserved above the axes for the 30-pt legend.
    if figsize is None:
        figsize = (FIG_WIDTH, FIG_HEIGHT)
    else:
        # Preserve the requested/common width, but guarantee enough height
        # for the legend band.
        figsize = (FIG_WIDTH, max(float(figsize[1]), FIG_HEIGHT))

    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(
        left=0.12,
        right=0.97,
        bottom=0.14,
        top=AXES_TOP,
    )
    return fig, ax


def _place_legend_top(ax, ncol=3, y=None):
    # Figure-level legend: it uses the reserved top band instead of
    # shrinking/covering the data region.
    fig = ax.figure
    handles, labels = ax.get_legend_handles_labels()

    # Remove duplicate labels while preserving order.
    unique = {}
    for h, lab in zip(handles, labels):
        if lab not in unique:
            unique[lab] = h

    fig.legend(
        list(unique.values()),
        list(unique.keys()),
        loc="upper center",
        bbox_to_anchor=(0.5, LEGEND_TOP_Y),
        ncol=ncol,
        frameon=True,
        fancybox=True,
        shadow=False,
        borderaxespad=0.35,
        columnspacing=1.0,
        handlelength=2.0,
        handletextpad=0.55,
        fontsize=PLOT_FONT_SIZE,
    )


def _annotate_series(ax, x, y, prefix="", fmt="{:.3f}", dy=0.012, index_shift=0):
    """
    Generic value-label helper.
    In the current version, ordinary plot-value labels are disabled
    to avoid overlap. P-value plots use their own annotations.
    """
    for i, (xi, yi) in enumerate(zip(x, y)):
        direction = 1 if ((i + index_shift) % 2 == 0) else -1
        yoff = dy * direction * (1 + 0.35 * (index_shift % 3))
        ax.annotate(
            f"{prefix}{fmt.format(float(yi))}",
            xy=(xi, yi),
            xytext=(0, 12 if yoff >= 0 else -18),
            textcoords="offset points",
            ha="center",
            va="bottom" if yoff >= 0 else "top",
            fontsize=PLOT_FONT_SIZE,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="gray", alpha=0.72),
        )


def _metric_display_name(metric):
    mapping = {
        "AUROC": "AUROC",
        "Macro_F1": "Macro-F1",
        "Balanced_Accuracy": "Bal. Acc.",
        "Sensitivity": "Sensitivity",
        "Specificity": "Specificity",
        "Brier": "Brier",
        "ECE": "ECE",
        "RVR": "RVR",
    }
    return mapping.get(metric, metric)


def _oriented_metric_value(metric, value):
    if metric == "Brier":
        return 1.0 - value
    if metric == "ECE":
        return 1.0 - value
    if metric == "RVR":
        return 1.0 - value
    return value


def _oriented_metric_label(metric):
    if metric == "Brier":
        return "1-Brier"
    if metric == "ECE":
        return "1-ECE"
    if metric == "RVR":
        return "1-RVR"
    return _metric_display_name(metric)


def _style_for(dataset, model):
    return {
        "color": DATASET_COLORS[dataset],
        "linestyle": MODEL_LINESTYLES.get(model, "-"),
        "marker": MODEL_MARKERS.get(model, "o"),
        "label": f"{dataset} — {model}",
    }


def _smooth_line(
    ax, x, y, *,
    color,
    linestyle="-",
    marker="o",
    linewidth=3.8,
    markersize=10,
    label=None,
    points=300,
):
    """
    Draw a rounded, shape-preserving curve through the observed points.

    The smooth line is only a visual interpolation. Original experimental
    observations remain visible as markers at their exact x/y locations.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if len(x) == 0:
        return None, None

    # PCHIP requires strictly increasing x values.
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    # Collapse duplicate x values if any.
    ux, inv = np.unique(x, return_inverse=True)
    if len(ux) != len(x):
        uy = np.array([y[inv == i].mean() for i in range(len(ux))], dtype=float)
        x, y = ux, uy

    if len(x) >= 3:
        xs = np.linspace(x.min(), x.max(), points)
        ys = PchipInterpolator(x, y)(xs)
        ax.plot(
            xs, ys,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            label=label,
        )
    else:
        xs, ys = x, y
        ax.plot(
            x, y,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            label=label,
        )

    # Keep the true measured points clearly visible.
    ax.plot(
        x, y,
        linestyle="None",
        marker=marker,
        color=color,
        markersize=markersize,
        label="_nolegend_",
    )
    return xs, ys


def _plot_mean_curve(ax, temp, x_col, y_col, dataset, model, annotate=False, alpha_fill=0.12, annotation_shift=0):
    if temp.empty:
        return
    s = _mean_std(temp, [x_col], y_col).sort_values(x_col)
    x = s[x_col].to_numpy(dtype=float)
    y = s["mean"].to_numpy(dtype=float)
    sd = s["std"].fillna(0).to_numpy(dtype=float)

    style = _style_for(dataset, model)
    _smooth_line(
        ax, x, y,
        color=style["color"],
        linestyle=style["linestyle"],
        marker=style["marker"],
        linewidth=3.8,
        markersize=10,
        label=style["label"],
    )
    ax.fill_between(
        x, y - sd, y + sd,
        color=style["color"],
        alpha=alpha_fill,
        linewidth=0,
    )
    if annotate:
        pass  # value labels removed for cleaner plots
    return s


def plot_metric_curves(
    df, x_col, metric, outdir, filename, title, xlabel,
    models=("S-NN", "iSAGE-Train", "iSAGE-Fusion"), x_descending=False
):
    fig, ax = _make_fig()

    line_index = 0
    for dataset in DATASETS:
        for model in models:
            temp = df[
                (df["Dataset"] == dataset) &
                (df["Model"] == model)
            ]
            if temp.empty:
                continue
            _plot_mean_curve(
                ax, temp, x_col, metric, dataset, model,
                annotate=False, alpha_fill=0.09 if model == "S-NN" else 0.14,
                annotation_shift=line_index,
            )
            line_index += 1

    if x_descending:
        pass  # keep natural left-to-right increasing order

    if x_col == "TrainingPercent":
        ax.set_xticks(TRAINING_PERCENT_TICKS)
    if x_col == "ObservationPercent":
        ax.set_xticks(OBSERVATION_PERCENT_TICKS)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(_metric_display_name(metric))
    # Title removed as requested; use figure caption instead
    _place_legend_top(ax, ncol=3, y=1.28)
    ax.grid(alpha=0.22, linestyle="--")
    savefig(fig, outdir, filename)


def _full_metric_summary(full_df, dataset, model):
    temp = full_df[
        (full_df["Dataset"] == dataset) &
        (full_df["Model"] == model)
    ]
    rows = []
    for metric in [
        "AUROC", "Macro_F1", "Balanced_Accuracy",
        "Sensitivity", "Specificity", "Brier", "ECE", "RVR"
    ]:
        rows.append({
            "metric": metric,
            "value": _oriented_metric_value(metric, temp[metric].mean()),
        })
    return pd.DataFrame(rows)


def plot_effect_adding_symbolic(full_df, figdir):
    fig, ax = _make_fig()
    metrics = [
        "AUROC", "Macro_F1", "Balanced_Accuracy",
        "Sensitivity", "Specificity", "Brier", "ECE", "RVR"
    ]
    labels = [_oriented_metric_label(m) for m in metrics]
    x = np.arange(len(metrics))

    line_index = 0
    for dataset in DATASETS:
        for model in ["S-NN", "iSAGE-Train", "iSAGE-Fusion"]:
            s = _full_metric_summary(full_df, dataset, model)
            y = s["value"].to_numpy(dtype=float)
            style = _style_for(dataset, model)
            _smooth_line(
                ax, x, y,
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                linewidth=3.8,
                markersize=11,
                label=style["label"],
            )
            pass  # value labels removed for cleaner plots
            line_index += 1

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylim(0.60, 1.02)
    ax.set_ylabel("Performance-Oriented Score")
    ax.set_xlabel("Evaluation Metric")
    # Title removed as requested; use figure caption instead
    _place_legend_top(ax, ncol=3, y=1.30)
    savefig(fig, figdir, "EffectAddingSymbolicKnowledge_MultiDataset")


def plot_ablation_components(full_df, figdir):
    fig, ax = _make_fig()
    metrics = [
        "AUROC", "Macro_F1", "Balanced_Accuracy",
        "Sensitivity", "Specificity", "Brier", "ECE", "RVR"
    ]
    labels = [_oriented_metric_label(m) for m in metrics]
    x = np.arange(len(metrics))

    line_index = 0
    for dataset in DATASETS:
        for model in ["S-NN", "Rule-Only", "iSAGE-Train", "iSAGE-Fusion"]:
            s = _full_metric_summary(full_df, dataset, model)
            y = s["value"].to_numpy(dtype=float)
            style = _style_for(dataset, model)
            _smooth_line(
                ax, x, y,
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                linewidth=3.8,
                markersize=11,
                label=style["label"],
            )
            pass  # value labels removed for cleaner plots
            line_index += 1

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylim(0.60, 1.02)
    ax.set_ylabel("Performance-Oriented Score")
    ax.set_xlabel("Evaluation Metric")
    # Title removed as requested; use figure caption instead
    _place_legend_top(ax, ncol=3, y=1.32)
    savefig(fig, figdir, "Ablation_Neural_Symbolic_Components_MultiDataset")


def plot_three_way_knowledge_timing(scarcity_df, figdir):
    """
    One separate figure per dataset comparing:
      S-NN vs iSAGE-Train vs iSAGE
    under training-data scarcity.
    """
    model_order = ["S-NN", "iSAGE-Train", "iSAGE-Fusion"]

    for dataset in DATASETS:
        fig, ax = _make_fig()

        for model in model_order:
            temp = scarcity_df[
                (scarcity_df["Dataset"] == dataset) &
                (scarcity_df["Model"] == model)
            ]
            if temp.empty:
                continue

            s = _mean_std(
                temp, ["TrainingPercent"], "AUROC"
            ).sort_values("TrainingPercent")

            x = s["TrainingPercent"].to_numpy(dtype=float)
            y = s["mean"].to_numpy(dtype=float)
            sd = s["std"].fillna(0).to_numpy(dtype=float)

            _smooth_line(
                ax, x, y,
                color={
                    "S-NN": "#1f77b4",
                    "iSAGE-Train": "#2ca02c",
                    "iSAGE-Fusion": "#ff7f0e",
                }[model],
                linestyle=MODEL_LINESTYLES[model],
                marker=MODEL_MARKERS[model],
                linewidth=3.8,
                markersize=10,
                label={
                    "S-NN": "S-NN",
                    "iSAGE-Train": "iSAGE-Train",
                    "iSAGE-Fusion": "iSAGE-Fusion",
                }[model],
            )

            ax.fill_between(
                x, y - sd, y + sd,
                color={
                    "S-NN": "#1f77b4",
                    "iSAGE-Train": "#2ca02c",
                    "iSAGE-Fusion": "#ff7f0e",
                }[model],
                alpha=0.08,
                linewidth=0,
            )

        ax.set_xticks(TRAINING_PERCENT_TICKS)
        ax.set_xlabel("Available Training Data (%)")
        ax.set_ylabel("AUROC")
        ax.grid(alpha=0.22, linestyle="--")
        _place_legend_top(ax, ncol=3, y=1.18)

        savefig(
            fig,
            figdir,
            f"Ablation_KnowledgeTiming_{_dataset_slug(dataset)}"
        )



def plot_data_scarcity_all_metrics(scarcity_df, figdir):
    for metric in [
        "AUROC", "Macro_F1", "Balanced_Accuracy",
        "Sensitivity", "Specificity", "Brier", "ECE", "RVR"
    ]:
        plot_metric_curves(
            scarcity_df,
            x_col="TrainingPercent",
            metric=metric,
            outdir=figdir,
            filename=f"DataScarcity_{metric}",
            title=f"{_metric_display_name(metric)} Under Training-Data Scarcity",
            xlabel="Available Training Data (%)",
            models=("S-NN", "iSAGE-Train", "iSAGE-Fusion"),
            x_descending=True,
        )


def plot_observation_all_metrics(observation_df, figdir):
    for metric in [
        "AUROC", "Macro_F1", "Balanced_Accuracy",
        "Sensitivity", "Specificity", "Brier", "ECE", "RVR"
    ]:
        plot_metric_curves(
            observation_df,
            x_col="ObservationPercent",
            metric=metric,
            outdir=figdir,
            filename=f"ObservationScarcity_{metric}",
            title=f"{_metric_display_name(metric)} Under Observation Scarcity",
            xlabel="Available Observations (%)",
            models=("S-NN", "iSAGE-Train", "iSAGE-Fusion"),
            x_descending=True,
        )


def plot_three_model_scarcity(scarcity_df, figdir):
    fig, ax = _make_fig()
    line_index = 0
    for dataset in DATASETS:
        for model in ["S-NN", "Rule-Only", "iSAGE-Train", "iSAGE-Fusion"]:
            t = scarcity_df[
                (scarcity_df["Dataset"] == dataset) &
                (scarcity_df["Model"] == model)
            ]
            _plot_mean_curve(
                ax, t, "TrainingPercent", "AUROC", dataset, model,
                annotate=False, alpha_fill=0.10, annotation_shift=line_index
            )
            line_index += 1
    # left-to-right increasing order requested
    ax.set_xticks(TRAINING_PERCENT_TICKS)
    ax.set_xlabel("Available Training Data (%)")
    ax.set_ylabel("AUROC")
    # Title removed as requested; use figure caption instead
    _place_legend_top(ax, ncol=3, y=1.30)
    savefig(fig, figdir, "NeuralSymbolicPerformance_MultiDataset")


def plot_performance_retention(scarcity_df, figdir):
    fig, ax = _make_fig()

    line_index = 0
    for dataset in DATASETS:
        color = DATASET_COLORS[dataset]
        for model in ["S-NN", "iSAGE-Train", "iSAGE-Fusion"]:
            temp = scarcity_df[
                (scarcity_df["Dataset"] == dataset) &
                (scarcity_df["Model"] == model)
            ].copy()

            pivot = temp.pivot_table(
                index="Seed",
                columns="TrainingPercent",
                values="AUROC"
            )
            if 100.0 not in pivot.columns:
                continue
            x = np.array(sorted(pivot.columns))
            retention = pivot.div(pivot[100.0], axis=0)
            means = retention[x].mean(axis=0).to_numpy(dtype=float)
            stds = retention[x].std(axis=0, ddof=1).fillna(0).to_numpy(dtype=float)

            style = _style_for(dataset, model)
            _smooth_line(
                ax, x, means,
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                linewidth=3.8,
                markersize=10,
                label=style["label"],
            )
            ax.fill_between(
                x, means - stds, means + stds,
                color=style["color"], alpha=0.10
            )
            pass  # value labels removed for cleaner plots
            line_index += 1

    # left-to-right increasing order requested
    ax.axhline(1.0, color="tab:blue", linestyle="--", linewidth=2.5)
    ax.set_xticks(TRAINING_PERCENT_TICKS)
    ax.set_xlabel("Available Training Data (%)")
    ax.set_ylabel("AUROC Performance Retention")
    # Title removed as requested; use figure caption instead
    _place_legend_top(ax, ncol=3, y=1.28)
    savefig(fig, figdir, "PerformanceRetention_UnderDataScarcity_MultiDataset")


def plot_isage_advantage(scarcity_df, figdir):
    fig, ax = _make_fig()

    line_index = 0
    for dataset in DATASETS:
        d = scarcity_df[scarcity_df["Dataset"] == dataset]
        care = d[d["Model"] == "iSAGE-Fusion"][
            ["Seed", "TrainingPercent", "AUROC"]
        ].rename(columns={"AUROC": "iSAGE-Fusion"})
        snn = d[d["Model"] == "S-NN"][
            ["Seed", "TrainingPercent", "AUROC"]
        ].rename(columns={"AUROC": "SNN"})
        m = care.merge(snn, on=["Seed", "TrainingPercent"])
        m["Delta"] = m["iSAGE-Fusion"] - m["SNN"]
        s = m.groupby("TrainingPercent")["Delta"].agg(["mean", "std"]).reset_index()
        s = s.sort_values("TrainingPercent")

        ax.errorbar(
            s["TrainingPercent"],
            s["mean"],
            yerr=s["std"].fillna(0),
            color=DATASET_COLORS[dataset],
            marker="o",
            linewidth=3.8,
            markersize=10,
            capsize=6,
            label=dataset,
        )
        pass  # value labels removed for cleaner plots
        line_index += 1

    ax.axhline(0, color="black", linestyle="--", linewidth=2.5)
    # left-to-right increasing order requested
    ax.set_xticks(TRAINING_PERCENT_TICKS)
    ax.set_xlabel("Available Training Data (%)")
    ax.set_ylabel(r"$\Delta$AUROC (iSAGE-Fusion - S-NN)")
    # Title removed as requested; use figure caption instead
    _place_legend_top(ax, ncol=3, y=1.22)
    savefig(fig, figdir, "Value_SymbolicKnowledge_UnderDataScarcity_MultiDataset")



def plot_symbolic_knowledge_gain(scarcity_df, figdir):
    """
    Paper-style visualization of the value of symbolic knowledge under
    training-data scarcity.

    For each dataset:
      - dashed line = S-NN
      - solid line = iSAGE
      - translucent band = AUROC gain from adding symbolic knowledge

    No ordinary point-value labels are drawn, keeping the figure clean.
    """
    fig, ax = _make_fig()

    for dataset in DATASETS:
        d = scarcity_df[scarcity_df["Dataset"] == dataset].copy()

        snn = (
            d[d["Model"] == "S-NN"]
            .groupby("TrainingPercent")["AUROC"]
            .agg(["mean", "std"])
            .reset_index()
            .sort_values("TrainingPercent")
        )
        care = (
            d[d["Model"] == "iSAGE-Fusion"]
            .groupby("TrainingPercent")["AUROC"]
            .agg(["mean", "std"])
            .reset_index()
            .sort_values("TrainingPercent")
        )

        merged = snn.merge(
            care,
            on="TrainingPercent",
            suffixes=("_SNN", "_iSAGE")
        )
        if merged.empty:
            continue

        x = merged["TrainingPercent"].to_numpy(dtype=float)
        y_snn = merged["mean_SNN"].to_numpy(dtype=float)
        y_care = merged["mean_iSAGE"].to_numpy(dtype=float)

        color = DATASET_COLORS[dataset]

        # Shaded region = gain due to symbolic knowledge.
        ax.fill_between(
            x,
            y_snn,
            y_care,
            color=color,
            alpha=0.10,
            linewidth=0,
            label=f"{dataset} — Symbolic gain",
        )

        # S-NN baseline.
        _smooth_line(
            ax,
            x,
            y_snn,
            color=color,
            linestyle="--",
            marker="s",
            linewidth=3.8,
            markersize=10,
            label=f"{dataset} — S-NN",
        )

        # iSAGE after adding symbolic knowledge.
        _smooth_line(
            ax,
            x,
            y_care,
            color=color,
            linestyle="-",
            marker="o",
            linewidth=3.8,
            markersize=10,
            label=f"{dataset} — iSAGE-Fusion",
        )

    ax.set_xticks(TRAINING_PERCENT_TICKS)
    ax.set_xlabel("Available Training Data (%)")
    ax.set_ylabel("AUROC")
    _place_legend_top(ax, ncol=3, y=1.30)
    ax.grid(alpha=0.22, linestyle="--")

    savefig(
        fig,
        figdir,
        "Value_SymbolicKnowledge_UnderDataScarcity_AllDatasets"
    )

def _dataset_slug(name: str) -> str:
    return (
        name.replace(" ", "_")
            .replace("-", "_")
            .replace("/", "_")
    )


def plot_symbolic_knowledge_gain_per_dataset(scarcity_df, figdir):
    """
    Create one separate paper-style figure per dataset with non-overlapping
    labels. Point values are offset away from markers/error bars, and delta
    boxes are placed above or below the curve pair depending on local spacing.
    """
    for dataset in DATASETS:
        d = scarcity_df[scarcity_df["Dataset"] == dataset].copy()
        if d.empty:
            continue

        snn = (
            d[d["Model"] == "S-NN"]
            .groupby("TrainingPercent")["AUROC"]
            .agg(["mean", "std"])
            .reset_index()
            .sort_values("TrainingPercent")
        )
        train = (
            d[d["Model"] == "iSAGE-Train"]
            .groupby("TrainingPercent")["AUROC"]
            .agg(["mean", "std"])
            .reset_index()
            .sort_values("TrainingPercent")
        )
        fusion = (
            d[d["Model"] == "iSAGE-Fusion"]
            .groupby("TrainingPercent")["AUROC"]
            .agg(["mean", "std"])
            .reset_index()
            .sort_values("TrainingPercent")
        )

        merged = (
            snn.merge(
                train,
                on="TrainingPercent",
                suffixes=("_SNN", "_Train")
            )
            .merge(
                fusion,
                on="TrainingPercent"
            )
            .rename(columns={
                "mean": "mean_Fusion",
                "std": "std_Fusion",
            })
        )
        if merged.empty:
            continue

        x = merged["TrainingPercent"].to_numpy(dtype=float)
        y_snn = merged["mean_SNN"].to_numpy(dtype=float)
        y_train = merged["mean_Train"].to_numpy(dtype=float)
        y_fusion = merged["mean_Fusion"].to_numpy(dtype=float)

        e_snn = merged["std_SNN"].fillna(0).to_numpy(dtype=float)
        e_train = merged["std_Train"].fillna(0).to_numpy(dtype=float)
        e_fusion = merged["std_Fusion"].fillna(0).to_numpy(dtype=float)

        # Proposed-method gain remains defined as iSAGE-Fusion - S-NN.
        delta = y_fusion - y_snn

        fig, ax = _make_fig()

        ax.fill_between(
            x,
            y_snn,
            y_fusion,
            color="lightsteelblue",
            alpha=0.42,
            linewidth=0,
            label="Fusion Knowledge Gain",
            zorder=1,
        )

        ax.errorbar(
            x,
            y_snn,
            yerr=e_snn,
            fmt="o-",
            color="#1f77b4",
            linewidth=3.8,
            markersize=10,
            capsize=8,
            label="S-NN",
            zorder=3,
        )
        ax.errorbar(
            x,
            y_train,
            yerr=e_train,
            fmt="D-.",
            color="#2ca02c",
            linewidth=3.4,
            markersize=9,
            capsize=8,
            label="iSAGE-Train",
            zorder=4,
        )
        ax.errorbar(
            x,
            y_fusion,
            yerr=e_fusion,
            fmt="s-",
            color="#ff7f0e",
            linewidth=3.8,
            markersize=10,
            capsize=8,
            label="iSAGE-Fusion",
            zorder=5,
        )

        # Expand y-range so labels have dedicated room above/below curves.
        all_low = np.minimum.reduce([
            y_snn - e_snn,
            y_train - e_train,
            y_fusion - e_fusion,
        ])
        all_high = np.maximum.reduce([
            y_snn + e_snn,
            y_train + e_train,
            y_fusion + e_fusion,
        ])
        y_min = float(np.nanmin(all_low))
        y_max = float(np.nanmax(all_high))
        yrange = max(y_max - y_min, 0.01)
        ax.set_ylim(y_min - 0.12 * yrange, y_max + 0.16 * yrange)

        # Manually separated positions for the six scarcity points.
        # These offsets keep labels away from axis ticks and one another.
        snn_offsets = [
            (-10, 18), (0, -26), (0, 18), (0, 18), (0, 18), (0, 18)
        ]
        care_offsets = [
            (10, -28), (0, 18), (0, -28), (0, -28), (0, -28), (0, -28)
        ]
        delta_offsets = [
            (0, 34), (12, 34), (-10, 34), (8, -36), (0, -36), (0, -36)
        ]

        # Dataset-specific tweaks when curves are very close.
        if dataset == "Breast Cancer":
            snn_offsets = [(-10, 20), (0, -30), (0, 20), (0, 20), (0, 20), (0, 20)]
            care_offsets = [(10, -30), (0, 20), (0, -30), (0, -30), (0, -30), (0, -30)]
            delta_offsets = [(0, 38), (12, 38), (-10, 38), (8, -40), (0, -40), (0, -40)]
        elif dataset == "Cardiovascular":
            delta_offsets = [(0, 36), (12, 36), (-10, 36), (8, -38), (0, -38), (0, -38)]
        elif dataset == "Diabetes":
            # Diabetes curves are close, so spread labels more aggressively.
            snn_offsets = [(-12, 20), (0, 20), (0, 20), (0, 20), (0, 20), (0, 20)]
            care_offsets = [(12, -30), (0, -30), (0, -30), (0, -30), (0, -30), (0, -30)]
            delta_offsets = [(0, 42), (14, 42), (-12, 42), (10, -42), (0, -42), (0, -42)]

        for i, (xi, ys, yc) in enumerate(zip(x, y_snn, y_fusion)):
            sx, sy = snn_offsets[min(i, len(snn_offsets)-1)]
            cx, cy = care_offsets[min(i, len(care_offsets)-1)]

            ax.annotate(
                f"{ys:.3f}",
                xy=(xi, ys),
                xytext=(sx, sy),
                textcoords="offset points",
                ha="center",
                va="bottom" if sy >= 0 else "top",
                fontsize=16,
                color="black",
                clip_on=False,
            )
            ax.annotate(
                f"{yc:.3f}",
                xy=(xi, yc),
                xytext=(cx, cy),
                textcoords="offset points",
                ha="center",
                va="bottom" if cy >= 0 else "top",
                fontsize=16,
                color="black",
                clip_on=False,
            )

        for i, (xi, ys, yc, dval) in enumerate(zip(x, y_snn, y_fusion, delta)):
            mid_y = (ys + yc) / 2.0
            dx, dy = delta_offsets[min(i, len(delta_offsets)-1)]

            ax.annotate(
                f"Δ = {dval:+.3f}",
                xy=(xi, mid_y),
                xytext=(dx, dy),
                textcoords="offset points",
                ha="center",
                va="bottom" if dy >= 0 else "top",
                fontsize=16,
                bbox=dict(
                    boxstyle="round,pad=0.18",
                    fc="white",
                    ec="gray",
                    alpha=0.84,
                ),
                clip_on=False,
                zorder=8,
            )

        ax.set_xticks(TRAINING_PERCENT_TICKS)
        ax.set_xlabel("Available Training Data (%)", fontsize=30)
        ax.set_ylabel("AUROC", fontsize=30)
        ax.tick_params(axis="both", labelsize=30, pad=8)
        ax.margins(x=0.035)
        ax.grid(alpha=0.22, linestyle="--")

        ax.legend(
            loc="lower right",
            frameon=True,
            fancybox=True,
            fontsize=21,
        )

        savefig(
            fig,
            figdir,
            f"Value_SymbolicKnowledge_UnderDataScarcity_{_dataset_slug(dataset)}"
        )



def plot_symbolic_knowledge_gain_three_panels(scarcity_df, figdir):
    """
    Also create a 1x3 panel figure so the same paper-style generation pattern
    is available side-by-side for all three datasets.
    """
    fig, axes = plt.subplots(1, 3, figsize=(28, 9.5))
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.18, top=0.90, wspace=0.22)

    for ax, dataset in zip(axes, DATASETS):
        d = scarcity_df[scarcity_df["Dataset"] == dataset].copy()
        if d.empty:
            continue

        snn = (
            d[d["Model"] == "S-NN"]
            .groupby("TrainingPercent")["AUROC"]
            .agg(["mean", "std"])
            .reset_index()
            .sort_values("TrainingPercent")
        )
        care = (
            d[d["Model"] == "iSAGE-Fusion"]
            .groupby("TrainingPercent")["AUROC"]
            .agg(["mean", "std"])
            .reset_index()
            .sort_values("TrainingPercent")
        )

        merged = snn.merge(
            care,
            on="TrainingPercent",
            suffixes=("_SNN", "_iSAGE")
        )
        if merged.empty:
            continue

        x = merged["TrainingPercent"].to_numpy(dtype=float)
        y_snn = merged["mean_SNN"].to_numpy(dtype=float)
        y_care = merged["mean_iSAGE"].to_numpy(dtype=float)
        e_snn = merged["std_SNN"].fillna(0).to_numpy(dtype=float)
        e_care = merged["std_iSAGE"].fillna(0).to_numpy(dtype=float)
        delta = y_care - y_snn

        ax.fill_between(
            x, y_snn, y_care,
            color="lightsteelblue", alpha=0.45, linewidth=0
        )
        ax.errorbar(
            x, y_snn, yerr=e_snn,
            fmt="o-", color="#1f77b4",
            linewidth=3.0, markersize=8, capsize=6,
            label="S-NN (Neural Only)"
        )
        ax.errorbar(
            x, y_care, yerr=e_care,
            fmt="s-", color="#ff7f0e",
            linewidth=3.0, markersize=8, capsize=6,
            label="iSAGE-Fusion (Post-Training Knowledge)"
        )

        for i, (xi, ys, yc, dval) in enumerate(zip(x, y_snn, y_care, delta)):
            ax.annotate(
                f"Δ={dval:+.3f}",
                xy=(xi, (ys + yc) / 2.0),
                xytext=(0, 7 if i % 2 == 0 else -10),
                textcoords="offset points",
                ha="center",
                va="center",
                fontsize=12,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="gray", alpha=0.72),
            )

        ax.set_xticks(TRAINING_PERCENT_TICKS)
        ax.set_xlabel("Available Training Data (%)", fontsize=20)
        ax.set_ylabel("AUROC", fontsize=20)
        ax.tick_params(axis="both", labelsize=18)
        ax.set_title(dataset, fontsize=20)
        ax.grid(alpha=0.22, linestyle="--")
        ax.legend(loc="lower right", fontsize=12, frameon=True)

    savefig(fig, figdir, "Value_SymbolicKnowledge_UnderDataScarcity_ThreePanels")


def plot_symbolic_coverage(observation_df, figdir):
    fig, ax = _make_fig()
    base = observation_df.drop_duplicates(
        ["Dataset", "Seed", "ObservationPercent"]
    )
    line_index = 0
    for dataset in DATASETS:
        t = base[base["Dataset"] == dataset]
        s = t.groupby("ObservationPercent")["SymbolicCoverage"].agg(
            ["mean", "std"]
        ).reset_index().sort_values("ObservationPercent")
        ax.errorbar(
            s["ObservationPercent"], s["mean"],
            yerr=s["std"].fillna(0),
            color=DATASET_COLORS[dataset],
            marker="o",
            linewidth=3.8,
            markersize=10,
            capsize=6,
            label=dataset,
        )
        pass  # value labels removed for cleaner plots
        line_index += 1
    # left-to-right increasing order requested
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Available Observations (%)")
    ax.set_ylabel("Symbolic Coverage")
    # Title removed as requested; use figure caption instead
    _place_legend_top(ax, ncol=3, y=1.22)
    savefig(fig, figdir, "SymbolicCoverage_ObservationScarcity_MultiDataset")


def plot_symbolic_consistency_observation(observation_df, figdir):
    fig, ax = _make_fig()
    line_index = 0
    for dataset in DATASETS:
        for model in ["S-NN", "iSAGE-Train", "iSAGE-Fusion"]:
            temp = observation_df[
                (observation_df["Dataset"] == dataset) &
                (observation_df["Model"] == model)
            ]
            _plot_mean_curve(
                ax, temp, "ObservationPercent", "RVR", dataset, model,
                annotate=False, alpha_fill=0.12, annotation_shift=line_index
            )
            line_index += 1
    # left-to-right increasing order requested
    ax.set_xlabel("Available Observations (%)")
    ax.set_ylabel("Rule Violation Rate (RVR)")
    # Title removed as requested; use figure caption instead
    _place_legend_top(ax, ncol=3, y=1.28)
    savefig(fig, figdir, "SymbolicConsistency_ObservationScarcity_MultiDataset")


def plot_knowledge_reversal(knowledge_df, figdir):
    fig, ax = _make_fig()
    line_index = 0

    for dataset in DATASETS:
        # iSAGE under removal and reversal
        for mode, style_name in [("removal", "iSAGE-Fusion"), ("reversal", "iSAGE-Fusion")]:
            t = knowledge_df[
                (knowledge_df["Dataset"] == dataset) &
                (knowledge_df["Mode"] == mode) &
                (knowledge_df["Model"] == "iSAGE-Fusion")
            ]
            if t.empty:
                continue
            s = _mean_std(t, ["CorruptionPercent"], "AUROC").sort_values("CorruptionPercent")
            color = DATASET_COLORS[dataset]
            linestyle = "--" if mode == "removal" else "-"
            marker = "s" if mode == "removal" else "o"
            label = f"{dataset} — {mode.capitalize()}"
            x = s["CorruptionPercent"].to_numpy(dtype=float)
            y = s["mean"].to_numpy(dtype=float)
            sd = s["std"].fillna(0).to_numpy(dtype=float)
            _smooth_line(
                ax, x, y,
                color=color,
                linestyle=linestyle,
                marker=marker,
                linewidth=3.8,
                markersize=10,
                label=label,
            )
            ax.fill_between(x, y - sd, y + sd, color=color, alpha=0.10)
            pass  # value labels removed for cleaner plots
            line_index += 1

        # S-NN baseline for the same dataset
        snn_base = knowledge_df[
            (knowledge_df["Dataset"] == dataset) &
            (knowledge_df["Model"] == "S-NN")
        ]["AUROC"].mean()
        ax.axhline(
            snn_base,
            color=DATASET_COLORS[dataset],
            linestyle=":",
            linewidth=2.5,
            alpha=0.85,
            label=f"{dataset} — S-NN baseline",
        )

    ax.set_xlabel("Corrupted Symbolic Rules (%)")
    ax.set_ylabel("AUROC")
    # Title removed as requested; use figure caption instead
    _place_legend_top(ax, ncol=3, y=1.34)
    savefig(fig, figdir, "Effect_MissingIncorrectSymbolicKnowledge_MultiDataset")


def plot_removal_vs_reversal(knowledge_df, figdir):
    # Keep a second version with only iSAGE lines, if user wants a cleaner figure too.
    fig, ax = _make_fig()
    line_index = 0
    for dataset in DATASETS:
        color = DATASET_COLORS[dataset]
        for mode in CORRUPTION_MODES:
            t = knowledge_df[
                (knowledge_df["Dataset"] == dataset) &
                (knowledge_df["Mode"] == mode) &
                (knowledge_df["Model"] == "iSAGE-Fusion")
            ]
            s = _mean_std(t, ["CorruptionPercent"], "AUROC").sort_values(
                "CorruptionPercent"
            )
            x = s["CorruptionPercent"].to_numpy(dtype=float)
            y = s["mean"].to_numpy(dtype=float)
            _smooth_line(
                ax, x, y,
                color=color,
                linestyle="--" if mode == "removal" else "-",
                marker="s" if mode == "removal" else "o",
                linewidth=3.8,
                markersize=10,
                label=f"{dataset} — {mode.capitalize()}",
            )
            pass  # value labels removed for cleaner plots
            line_index += 1
    ax.set_xlabel("Corrupted Symbolic Rules (%)")
    ax.set_ylabel("iSAGE-Fusion AUROC")
    # Title removed as requested; use figure caption instead
    _place_legend_top(ax, ncol=3, y=1.30)
    savefig(fig, figdir, "iSAGEFusion_RemovalVsReversal_MultiDataset")


def plot_symbolic_consistency_data(scarcity_df, figdir):
    fig, ax = _make_fig()
    line_index = 0
    for dataset in DATASETS:
        for model in ["S-NN", "iSAGE-Train", "iSAGE-Fusion"]:
            temp = scarcity_df[
                (scarcity_df["Dataset"] == dataset) &
                (scarcity_df["Model"] == model)
            ]
            _plot_mean_curve(
                ax, temp, "TrainingPercent", "RVR", dataset, model,
                annotate=False, alpha_fill=0.12, annotation_shift=line_index
            )
            line_index += 1
    # left-to-right increasing order requested
    ax.set_xticks(TRAINING_PERCENT_TICKS)
    ax.set_xlabel("Available Training Data (%)")
    ax.set_ylabel("Rule Violation Rate (RVR)")
    # Title removed as requested; use figure caption instead
    _place_legend_top(ax, ncol=3, y=1.28)
    savefig(fig, figdir, "SymbolicConsistency_DataScarcity_MultiDataset")


def plot_performance_observation_scarcity(observation_df, figdir):
    fig, ax = _make_fig()
    line_index = 0
    for dataset in DATASETS:
        for model in ["S-NN", "iSAGE-Train", "iSAGE-Fusion"]:
            temp = observation_df[
                (observation_df["Dataset"] == dataset) &
                (observation_df["Model"] == model)
            ]
            _plot_mean_curve(
                ax, temp, "ObservationPercent", "AUROC", dataset, model,
                annotate=False, alpha_fill=0.12, annotation_shift=line_index
            )
            line_index += 1
    # left-to-right increasing order requested
    ax.set_xlabel("Available Observations (%)")
    ax.set_ylabel("AUROC")
    # Title removed as requested; use figure caption instead
    _place_legend_top(ax, ncol=3, y=1.28)
    savefig(fig, figdir, "Performance_ObservationScarcity_MultiDataset")


def plot_joint_resource(joint_df, figdir):
    order = ["S1_Full", "S2_Moderate", "S3_Severe"]
    x = np.arange(len(order))

    fig, ax = _make_fig()
    line_index = 0

    for dataset in DATASETS:
        color = DATASET_COLORS[dataset]
        for model in ["S-NN", "iSAGE-Train", "iSAGE-Fusion"]:
            temp = joint_df[
                (joint_df["Dataset"] == dataset) &
                (joint_df["Model"] == model)
            ]
            means = []
            stds = []
            for c in order:
                vals = temp[temp["Condition"] == c]["AUROC"]
                means.append(vals.mean())
                stds.append(vals.std(ddof=1))
            means = np.asarray(means, dtype=float)
            stds = np.nan_to_num(np.asarray(stds, dtype=float), nan=0.0)

            style = _style_for(dataset, model)
            _smooth_line(
                ax, x, means,
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                linewidth=3.8,
                markersize=10,
                label=style["label"],
            )
            ax.fill_between(
                x, means - stds, means + stds,
                color=style["color"], alpha=0.10,
            )
            pass  # value labels removed for cleaner plots
            line_index += 1

    ax.set_xticks(x)
    ax.set_xticklabels(["S1 Full", "S2 Moderate", "S3 Severe"])
    ax.set_ylabel("AUROC")
    # Title removed as requested; use figure caption instead
    _place_legend_top(ax, ncol=3, y=1.28)
    savefig(fig, figdir, "JointResourceDegradation_MultiDataset")


def plot_rrs(rrs_df, figdir):
    summary = (
        rrs_df.groupby(["Dataset", "Model"])["RRS"]
              .agg(["mean", "std"])
              .reset_index()
    )

    datasets = list(DATASETS)
    x = np.arange(len(datasets))
    width = 0.24

    fig, ax = _make_fig()
    compare_models = ["S-NN", "iSAGE-Train", "iSAGE-Fusion"]

    for i, model in enumerate(compare_models):
        means, stds = [], []
        for dataset in datasets:
            row = summary[
                (summary["Dataset"] == dataset) &
                (summary["Model"] == model)
            ]
            means.append(float(row["mean"].iloc[0]))
            stds.append(float(row["std"].fillna(0).iloc[0]))

        positions = x + (i - 1.0) * width
        ax.bar(
            positions,
            means,
            width=width,
            yerr=stds,
            capsize=6,
            color=[DATASET_COLORS[d] for d in datasets],
            edgecolor="black",
            linewidth=1.0,
            hatch={"S-NN": "//", "iSAGE-Train": "..", "iSAGE-Fusion": ""}[model],
            alpha=0.95 if model != "S-NN" else 0.72,
            label=model,
        )
        # Value labels removed to avoid overlap inside bar plots.

    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Resource Robustness Score")
    ax.set_ylim(0, 1.15)
    # Title removed as requested; use figure caption instead
    _place_legend_top(ax, ncol=3, y=1.18)
    savefig(fig, figdir, "ResourceRobustnessScore_MultiDataset")


def plot_full_resource_bars(full_df, figdir):
    summary = (
        full_df[full_df["Model"].isin(["S-NN", "iSAGE-Train", "iSAGE-Fusion"])]
        .groupby(["Dataset", "Model"])["AUROC"]
        .agg(["mean", "std"])
        .reset_index()
    )

    datasets = list(DATASETS)
    x = np.arange(len(datasets))
    width = 0.24

    fig, ax = _make_fig()
    compare_models = ["S-NN", "iSAGE-Train", "iSAGE-Fusion"]
    for i, model in enumerate(compare_models):
        means = []
        stds = []
        for dataset in datasets:
            row = summary[
                (summary["Dataset"] == dataset) &
                (summary["Model"] == model)
            ]
            means.append(float(row["mean"].iloc[0]))
            stds.append(float(row["std"].fillna(0).iloc[0]))

        bars = ax.bar(
            x + (i - 1.0) * width,
            means,
            width,
            yerr=stds,
            capsize=6,
            color=[DATASET_COLORS[d] for d in datasets],
            edgecolor="black",
            hatch={"S-NN": "//", "iSAGE-Train": "..", "iSAGE-Fusion": ""}[model],
            alpha=0.72 if model == "S-NN" else 0.95,
            label=model,
        )
        # Value labels removed to avoid overlap inside bar plots.

    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("AUROC")
    # Title removed as requested; use figure caption instead
    _place_legend_top(ax, ncol=2, y=1.18)
    savefig(fig, figdir, "FullResource_AUROC_MultiDataset")


def plot_p_values(stats_df, x_col, p_col, figdir, filename, xlabel, title):
    fig, ax = _make_fig()
    dataset_order = list(DATASETS)

    # Horizontal nudges by dataset help separate labels at the same x value.
    x_offsets = {
        dataset_order[0]: -18,
        dataset_order[1]: 0,
        dataset_order[2]: 18,
    }

    for dataset_idx, dataset in enumerate(DATASETS):
        t = stats_df[stats_df["Dataset"] == dataset].copy()
        if t.empty:
            continue
        t = t.sort_values(x_col)
        x = t[x_col].to_numpy(dtype=float)
        y = t[p_col].to_numpy(dtype=float)

        _smooth_line(
            ax, x, y,
            color=DATASET_COLORS[dataset],
            linestyle="-",
            marker="o",
            linewidth=3.8,
            markersize=10,
            label=dataset,
        )

        for j, (xi, yi) in enumerate(zip(x, y)):
            label_text = f"p={yi:.3g}"

            # Keep labels away from the x-axis and from each other.
            # Very small p-values are always placed above the point.
            if yi <= 0.01:
                vertical_offset = 28 + 7 * dataset_idx + 4 * (j % 2)
            elif yi <= 0.05:
                vertical_offset = 22 if dataset_idx != 1 else -26
            else:
                vertical_offset = -20 if dataset_idx == 1 else 20

            horizontal_offset = x_offsets.get(dataset, 0)

            ax.annotate(
                label_text,
                xy=(xi, yi),
                xytext=(horizontal_offset, vertical_offset),
                textcoords="offset points",
                ha="center",
                va="bottom" if vertical_offset >= 0 else "top",
                fontsize=18,
                bbox=dict(
                    boxstyle="round,pad=0.14",
                    fc="white",
                    ec="black",
                    alpha=0.72,
                ),
                clip_on=False,
            )

    ax.axhline(0.05, color="tab:blue", linestyle="--", linewidth=2.5, label=r"$p = 0.05$")
    ax.set_yscale("log")
    if x_col == "TrainingPercent":
        ax.set_xticks(TRAINING_PERCENT_TICKS)
    if x_col == "ObservationPercent":
        ax.set_xticks(OBSERVATION_PERCENT_TICKS)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Paired Wilcoxon $p$-value")
    # Title intentionally omitted; use figure caption instead.
    ax.margins(x=0.05, y=0.22)
    _place_legend_top(ax, ncol=4, y=1.18)
    savefig(fig, figdir, filename)
# ============================================================
# TABLE EXPORT
# ============================================================

def export_all_tables(all_results, outdir):
    tabledir = outdir / "tables"
    ensure_dir(tabledir)

    full_df = all_results["full"]
    scarcity_df = all_results["scarcity"]
    observation_df = all_results["observation"]
    knowledge_df = all_results["knowledge"]
    joint_df = all_results["joint"]
    compute_df = all_results["compute"]

    # Model definitions used consistently across all results.
    pd.DataFrame([
        {
            "Model": "S-NN",
            "KnowledgeUse": "None",
            "Definition": "Neural model trained using clinical features only."
        },
        {
            "Model": "iSAGE-Train",
            "KnowledgeUse": "During training",
            "Definition": "Reliability-weighted symbolic features are added to the neural input during training."
        },
        {
            "Model": "iSAGE-Fusion",
            "KnowledgeUse": "After neural training",
            "Definition": "S-NN is trained first; symbolic probability is conditionally fused with neural probability using validation-selected alpha and beta."
        },
        {
            "Model": "Rule-Only",
            "KnowledgeUse": "Symbolic only",
            "Definition": "Symbolic probability without neural prediction; auxiliary reference only."
        },
    ]).to_csv(tabledir / "Model_Definitions.csv", index=False)

    # Raw
    full_df.to_csv(tabledir / "full_resource_raw.csv", index=False)
    scarcity_df.to_csv(tabledir / "data_scarcity_raw.csv", index=False)
    observation_df.to_csv(tabledir / "observation_scarcity_raw.csv", index=False)
    knowledge_df.to_csv(tabledir / "knowledge_corruption_raw.csv", index=False)
    joint_df.to_csv(tabledir / "joint_resource_raw.csv", index=False)
    compute_df.to_csv(tabledir / "compute_profile_raw.csv", index=False)

    # Summaries
    full_summary = mean_std_table(
        full_df,
        ["Dataset", "Model"]
    )
    full_summary.to_csv(tabledir / "Table_FullResource_Summary.csv", index=False)

    scarcity_summary = mean_std_table(
        scarcity_df,
        ["Dataset", "TrainingPercent", "Model"]
    )
    scarcity_summary.to_csv(tabledir / "Table_DataScarcity_Summary.csv", index=False)

    obs_summary = mean_std_table(
        observation_df,
        ["Dataset", "ObservationPercent", "Model"]
    )
    obs_summary.to_csv(tabledir / "Table_ObservationScarcity_Summary.csv", index=False)

    knowledge_summary = mean_std_table(
        knowledge_df,
        ["Dataset", "Mode", "CorruptionPercent", "Model"]
    )
    knowledge_summary.to_csv(
        tabledir / "Table_KnowledgeCorruption_Summary.csv", index=False
    )

    joint_summary = mean_std_table(
        joint_df,
        ["Dataset", "Condition", "Model"]
    )
    joint_summary.to_csv(tabledir / "Table_JointResource_Summary.csv", index=False)

    compute_summary = (
        compute_df.groupby(["Dataset", "Model"])
        .agg(
            Parameters_mean=("Parameters", "mean"),
            SerializedModelMB_mean=("SerializedModelMB", "mean"),
            LatencyPerSampleMs_mean=("LatencyPerSampleMs", "mean"),
            LatencyPerSampleMs_std=("LatencyPerSampleMs", "std"),
        )
        .reset_index()
    )
    compute_summary.to_csv(tabledir / "Table_Compute_Summary.csv", index=False)

    # Paired stats
    scarcity_stats = paired_comparison_stats(
        scarcity_df,
        condition_col="TrainingPercent",
        conditions=sorted(scarcity_df["TrainingPercent"].unique(), reverse=True),
    )
    scarcity_stats.to_csv(tabledir / "Stats_DataScarcity.csv", index=False)

    obs_stats = paired_comparison_stats(
        observation_df,
        condition_col="ObservationPercent",
        conditions=sorted(observation_df["ObservationPercent"].unique(), reverse=True),
    )
    obs_stats.to_csv(tabledir / "Stats_ObservationScarcity.csv", index=False)

    joint_stats = paired_comparison_stats(
        joint_df,
        condition_col="Condition",
        conditions=list(RESOURCE_CONDITIONS),
    )
    joint_stats.to_csv(tabledir / "Stats_JointResource.csv", index=False)

    knowledge_stats = build_knowledge_stats(knowledge_df)
    knowledge_stats.to_csv(tabledir / "Stats_KnowledgeCorruption.csv", index=False)

    boundary_df = knowledge_boundaries(knowledge_stats)
    boundary_df.to_csv(tabledir / "KnowledgeReliabilityBoundary.csv", index=False)

    rrs_df = compute_rrs(joint_df)
    rrs_df.to_csv(tabledir / "RRS_Raw.csv", index=False)
    rrs_summary = (
        rrs_df.groupby(["Dataset", "Model"])["RRS"]
        .agg(["mean", "std"])
        .reset_index()
    )
    rrs_summary.to_csv(tabledir / "Table_RRS_Summary.csv", index=False)

    # Paper-style compact tables
    compact_scarcity = []
    for dataset in DATASETS:
        for pct in sorted(scarcity_df["TrainingPercent"].unique(), reverse=True):
            temp = scarcity_df[
                (scarcity_df["Dataset"] == dataset) &
                (scarcity_df["TrainingPercent"] == pct)
            ]
            snn = temp[temp["Model"] == "S-NN"]["AUROC"]
            train_knowledge = temp[temp["Model"] == "iSAGE-Train"]["AUROC"]
            care = temp[temp["Model"] == "iSAGE-Fusion"]["AUROC"]
            rule = temp[temp["Model"] == "Rule-Only"]["AUROC"]

            merged = (
                temp[temp["Model"] == "iSAGE-Fusion"][["Seed", "AUROC", "RVR"]]
                .rename(columns={"AUROC": "Care_AUC", "RVR": "Care_RVR"})
                .merge(
                    temp[temp["Model"] == "S-NN"][["Seed", "AUROC", "RVR"]]
                    .rename(columns={"AUROC": "SNN_AUC", "RVR": "SNN_RVR"}),
                    on="Seed"
                )
            )
            _, p = safe_wilcoxon(merged["Care_AUC"], merged["SNN_AUC"])
            compact_scarcity.append({
                "Dataset": dataset,
                "Training": f"{pct:.0f}%",
                "S-NN_AUROC": f"{snn.mean():.3f} ± {snn.std(ddof=1):.3f}",
                "iSAGE-Train_AUROC": f"{train_knowledge.mean():.3f} ± {train_knowledge.std(ddof=1):.3f}",
                "RuleOnly_AUROC": f"{rule.mean():.3f} ± {rule.std(ddof=1):.3f}",
                "iSAGE-Fusion_AUROC": f"{care.mean():.3f} ± {care.std(ddof=1):.3f}",
                "Delta_AUROC": (care.mean() - snn.mean()),
                "RVR_Improvement": (
                    merged["SNN_RVR"].mean() - merged["Care_RVR"].mean()
                ),
                "p_value": p,
                "Sig": significance_label(p),
            })
    pd.DataFrame(compact_scarcity).to_csv(
        tabledir / "PaperTable_DataScarcity_ThreeDatasets.csv", index=False
    )


    # Three-way timing comparison table
    timing_rows = []
    for dataset in DATASETS:
        for pct in sorted(
            scarcity_df["TrainingPercent"].unique()
        ):
            temp = scarcity_df[
                (scarcity_df["Dataset"] == dataset) &
                (scarcity_df["TrainingPercent"] == pct)
            ]
            row = {
                "Dataset": dataset,
                "TrainingPercent": pct,
            }
            for model in ["S-NN", "iSAGE-Train", "iSAGE-Fusion"]:
                vals = temp[temp["Model"] == model]["AUROC"]
                row[f"{model}_AUROC_mean"] = vals.mean()
                row[f"{model}_AUROC_std"] = vals.std(ddof=1)
            row["TrainKnowledge_minus_SNN"] = (
                row["iSAGE-Train_AUROC_mean"] -
                row["S-NN_AUROC_mean"]
            )
            row["iSAGE_Fusion_minus_SNN"] = (
                row["iSAGE-Fusion_AUROC_mean"] -
                row["S-NN_AUROC_mean"]
            )
            timing_rows.append(row)

    pd.DataFrame(timing_rows).to_csv(
        tabledir / "PaperTable_KnowledgeTiming_Ablation.csv",
        index=False,
    )

    return {
        "scarcity_stats": scarcity_stats,
        "obs_stats": obs_stats,
        "joint_stats": joint_stats,
        "knowledge_stats": knowledge_stats,
        "rrs": rrs_df,
        "boundaries": boundary_df,
    }


# ============================================================
# FIGURE EXPORT
# ============================================================

def export_all_figures(all_results, stats, outdir):
    figdir = outdir / "figures"
    ensure_dir(figdir)

    full_df = all_results["full"]
    scarcity_df = all_results["scarcity"]
    observation_df = all_results["observation"]
    knowledge_df = all_results["knowledge"]
    joint_df = all_results["joint"]

    # Original generalized outputs
    plot_full_resource_bars(full_df, figdir)
    plot_data_scarcity_all_metrics(scarcity_df, figdir)
    plot_three_way_knowledge_timing(scarcity_df, figdir)
    plot_observation_all_metrics(observation_df, figdir)
    plot_three_model_scarcity(scarcity_df, figdir)
    plot_performance_retention(scarcity_df, figdir)
    plot_isage_advantage(scarcity_df, figdir)
    plot_symbolic_knowledge_gain_per_dataset(scarcity_df, figdir)
    plot_symbolic_coverage(observation_df, figdir)
    plot_symbolic_consistency_data(scarcity_df, figdir)
    plot_symbolic_consistency_observation(observation_df, figdir)
    plot_knowledge_reversal(knowledge_df, figdir)
    plot_removal_vs_reversal(knowledge_df, figdir)
    plot_performance_observation_scarcity(observation_df, figdir)
    plot_joint_resource(joint_df, figdir)
    plot_rrs(stats["rrs"], figdir)

    # New versions matching the user-shared style more closely
    plot_effect_adding_symbolic(full_df, figdir)
    plot_ablation_components(full_df, figdir)

    plot_p_values(
        stats["scarcity_stats"],
        x_col="TrainingPercent",
        p_col="AUROC_p",
        figdir=figdir,
        filename="PValues_DataScarcity_MultiDataset",
        xlabel="Available Training Data (%)",
        title="Statistical Evidence Under Data Scarcity",
    )
    plot_p_values(
        stats["obs_stats"],
        x_col="ObservationPercent",
        p_col="AUROC_p",
        figdir=figdir,
        filename="PValues_ObservationScarcity_MultiDataset",
        xlabel="Available Observations (%)",
        title="Statistical Evidence Under Observation Scarcity",
    )


# ============================================================
# MAIN
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="iSAGE three-dataset experiment"
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path("."),
        help="Folder containing the three CSV files.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results_three_datasets"),
        help="Where CSV tables and figures are written.",
    )

    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--quick",
        action="store_true",
        help="Smoke test: 3 seeds and max 20,000 samples/dataset.",
    )
    mode.add_argument(
        "--paper",
        action="store_true",
        help="Paper mode: 10 seeds and full datasets.",
    )

    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional stratified sample cap per dataset.",
    )
    p.add_argument(
        "--full-baselines",
        action="store_true",
        help="Also run LR/RF/GBT/NN/S-NN full-resource baselines.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.quick:
        seeds = QUICK_SEEDS
        max_samples = args.max_samples or 20000
        quick = True
    else:
        # Default is paper behavior because user requested complete experiments.
        seeds = PAPER_SEEDS
        max_samples = args.max_samples
        quick = False

    ensure_dir(args.output_dir)
    ensure_dir(args.output_dir / "tables")
    ensure_dir(args.output_dir / "figures")

    print("=" * 80)
    print("iSAGE THREE-DATASET EXPERIMENT")
    print("=" * 80)
    print("Seeds:", seeds)
    print("Mode:", "QUICK" if quick else "PAPER/FULL")
    print("Data directory:", args.data_dir.resolve())
    print("Output directory:", args.output_dir.resolve())
    print("=" * 80)

    # Save run configuration.
    config = {
        "seeds": seeds,
        "data_fractions": DATA_FRACTIONS,
        "observation_fractions": OBSERVATION_FRACTIONS,
        "corruption_levels": CORRUPTION_LEVELS,
        "corruption_modes": CORRUPTION_MODES,
        "resource_conditions": RESOURCE_CONDITIONS,
        "alpha_grid": ALPHA_GRID,
        "beta_grid": BETA_GRID,
        "max_samples": max_samples,
        "quick": quick,
        "full_baselines": bool(args.full_baselines),
    }
    with open(args.output_dir / "run_config.json", "w") as f:
        json.dump(config, f, indent=2)

    cache = ExperimentCache(quick=quick)

    all_full = []
    all_scarcity = []
    all_observation = []
    all_knowledge = []
    all_joint = []
    all_compute = []
    summaries = []
    baseline_frames = []

    for dataset_name, spec in DATASETS.items():
        X, y, path = load_dataset(
            spec,
            args.data_dir,
            max_samples=max_samples,
            seed=42,
        )

        print(
            f"\nLoaded {dataset_name}: "
            f"{X.shape[0]:,} samples × {X.shape[1]} features | "
            f"positive rate={y.mean():.4f}"
        )
        summaries.append(
            dataset_summary_row(dataset_name, X, y, path)
        )

        results = run_dataset_experiments(
            dataset_name,
            X, y,
            seeds=seeds,
            cache=cache,
            outdir=args.output_dir,
        )

        all_full.append(results["full"])
        all_scarcity.append(results["scarcity"])
        all_observation.append(results["observation"])
        all_knowledge.append(results["knowledge"])
        all_joint.append(results["joint"])
        all_compute.append(results["compute"])

        if args.full_baselines:
            print(f"\n[{dataset_name}] Running conventional baselines...")
            baseline_frames.append(
                run_conventional_baselines(
                    dataset_name, X, y, seeds, quick=quick
                )
            )

    pd.DataFrame(summaries).to_csv(
        args.output_dir / "tables" / "Dataset_Summary.csv",
        index=False,
    )

    all_results = {
        "full": pd.concat(all_full, ignore_index=True),
        "scarcity": pd.concat(all_scarcity, ignore_index=True),
        "observation": pd.concat(all_observation, ignore_index=True),
        "knowledge": pd.concat(all_knowledge, ignore_index=True),
        "joint": pd.concat(all_joint, ignore_index=True),
        "compute": pd.concat(all_compute, ignore_index=True),
    }

    if baseline_frames:
        baselines = pd.concat(baseline_frames, ignore_index=True)
        baselines.to_csv(
            args.output_dir / "tables" / "Conventional_Baselines_Raw.csv",
            index=False,
        )
        mean_std_table(
            baselines,
            ["Dataset", "Model"],
            metrics=[
                "AUROC", "Macro_F1", "Balanced_Accuracy",
                "Sensitivity", "Specificity", "Brier", "ECE",
            ],
        ).to_csv(
            args.output_dir / "tables" / "Conventional_Baselines_Summary.csv",
            index=False,
        )

    print("\nExporting tables...")
    stats = export_all_tables(all_results, args.output_dir)

    print("Exporting combined three-dataset figures...")
    export_all_figures(all_results, stats, args.output_dir)

    print("\n" + "=" * 80)
    print("EXPERIMENT FINISHED")
    print("=" * 80)
    print("Tables :", (args.output_dir / "tables").resolve())
    print("Figures:", (args.output_dir / "figures").resolve())
    print("\nKey files:")
    print(" - PaperTable_DataScarcity_ThreeDatasets.csv")
    print(" - Table_ObservationScarcity_Summary.csv")
    print(" - Stats_KnowledgeCorruption.csv")
    print(" - KnowledgeReliabilityBoundary.csv")
    print(" - Table_JointResource_Summary.csv")
    print(" - Table_RRS_Summary.csv")
    print(" - DataScarcity_AUROC.pdf/png")
    print(" - ObservationScarcity_AUROC.pdf/png")
    print(" - KnowledgeCorruption_Reversal_AUROC.pdf/png")
    print(" - JointResourceDegradation_AUROC.pdf/png")
    print(" - ResourceRobustnessScore.pdf/png")
    print("=" * 80)


if __name__ == "__main__":
    main()
