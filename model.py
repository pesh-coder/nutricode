"""
NutriCode Uganda — Model Module
Trains a Random Forest + XGBoost ensemble to predict phytate reduction percentage
based on crop, microbial strain, soil conditions, and cooking parameters.

The training data combines published phytase research with Uganda-specific
regional data. This is the decision layer of NutriCode — laboratory validation
at Makerere University is the next step.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import shap
import joblib
from pathlib import Path


FEATURES = [
    "grain_type_code",
    "strain_code",
    "initial_phytate_mg",
    "microbe_dose_log",
    "ph_level",
    "temperature_c",
    "contact_time_min",
    "soil_zinc_ppm",
]

FEATURE_LABELS = {
    "grain_type_code": "Grain type",
    "strain_code": "Microbial strain",
    "initial_phytate_mg": "Initial phytate (mg/100g)",
    "microbe_dose_log": "Microbe dose (log CFU/g)",
    "ph_level": "pH level",
    "temperature_c": "Temperature (°C)",
    "contact_time_min": "Contact time (min)",
    "soil_zinc_ppm": "Soil zinc (ppm)",
}

TARGET = "phytate_reduction_pct"

STRAIN_NAMES = {
    0: "L. plantarum TISTR543 (lactic acid bacterium)",
    1: "B. subtilis natto (spore-forming bacterium)",
    2: "A. niger NRRL3135 (fungal phytase, high efficiency)",
    3: "P. kudriavzevii TY13 (non-conventional yeast)",
    4: "Mixed A. niger + L. plantarum consortium",
}

STRAIN_DESCRIPTIONS = {
    0: "Traditional, widely available lactic acid bacterium. Moderate phytate reduction. Well-suited to fermented porridges but slower action than fungal phytases.",
    1: "Bacillus strain with good heat tolerance. Mid-range performance. Spore-forming, so stable in dry storage conditions typical of rural Uganda.",
    2: "Fungal phytase with highest documented phytate reduction rates (often >85%). The industry benchmark. Requires safety profile assessment for household use.",
    3: "Non-conventional yeast showing very high phytate degradation in short contact times. Emerging strain with strong laboratory performance.",
    4: "Synergistic consortium combining fungal phytase efficiency with lactic acid bacterium fermentation benefits. Designed for staple flour applications.",
}

CROP_NAMES = {
    0: "Pearl Millet / Finger Millet",
    1: "White Maize",
    2: "Cassava Flour",
    3: "Sorghum",
    4: "Composite Flour",
}


def load_training_data(data_path: str = "data/phytase_performance.csv") -> pd.DataFrame:
    """Load the literature-derived phytase performance dataset."""
    df = pd.read_csv(data_path)

    # Encode strain as numeric
    strain_map = {
        "Lactobacillus plantarum": 0,
        "Bacillus subtilis": 1,
        "Aspergillus niger": 2,
        "Pichia kudriavzevii": 3,
        "Aspergillus niger + L.plantarum": 4,
    }
    df["strain_code"] = df["source_organism"].map(strain_map)
    return df


def train_models(df: pd.DataFrame):
    """Train Random Forest and XGBoost on the phytase dataset."""
    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # Random Forest
    rf = RandomForestRegressor(
        n_estimators=200, max_depth=10, min_samples_leaf=2, random_state=42
    )
    rf.fit(X_train, y_train)

    # XGBoost
    xgb_model = xgb.XGBRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.08,
        random_state=42, verbosity=0
    )
    xgb_model.fit(X_train, y_train)

    # Metrics
    rf_pred = rf.predict(X_test)
    xgb_pred = xgb_model.predict(X_test)
    ensemble_pred = (rf_pred + xgb_pred) / 2.0

    metrics = {
        "rf_mae": mean_absolute_error(y_test, rf_pred),
        "rf_r2": r2_score(y_test, rf_pred),
        "xgb_mae": mean_absolute_error(y_test, xgb_pred),
        "xgb_r2": r2_score(y_test, xgb_pred),
        "ensemble_mae": mean_absolute_error(y_test, ensemble_pred),
        "ensemble_r2": r2_score(y_test, ensemble_pred),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    return rf, xgb_model, metrics, X_train


def predict_for_inputs(rf, xgb_model, inputs: dict) -> float:
    """Ensemble prediction of phytate reduction % for a single input dict."""
    x = pd.DataFrame([[inputs[f] for f in FEATURES]], columns=FEATURES)
    rf_p = rf.predict(x)[0]
    xgb_p = xgb_model.predict(x)[0]
    return (rf_p + xgb_p) / 2.0


def rank_candidate_strains(rf, xgb_model, base_inputs: dict) -> pd.DataFrame:
    """For a given region/crop context, rank all candidate strains
    by predicted phytate reduction %."""
    rows = []
    for strain_code, strain_name in STRAIN_NAMES.items():
        inputs = dict(base_inputs)
        inputs["strain_code"] = strain_code

        # Use strain-appropriate default conditions (pH, temp, dose, time)
        if strain_code == 0:  # L. plantarum
            inputs.setdefault("ph_level", 5.4)
            inputs["temperature_c"] = 35
            inputs["microbe_dose_log"] = 6
            inputs["contact_time_min"] = 1440
        elif strain_code == 1:  # B. subtilis
            inputs.setdefault("ph_level", 6.0)
            inputs["temperature_c"] = 40
            inputs["microbe_dose_log"] = 7
            inputs["contact_time_min"] = 1440
        elif strain_code == 2:  # A. niger
            inputs.setdefault("ph_level", 5.0)
            inputs["temperature_c"] = 37
            inputs["microbe_dose_log"] = 5
            inputs["contact_time_min"] = 1440
        elif strain_code == 3:  # P. kudriavzevii
            inputs.setdefault("ph_level", 5.4)
            inputs["temperature_c"] = 30
            inputs["microbe_dose_log"] = 6
            inputs["contact_time_min"] = 120
        else:  # Mixed consortium
            inputs.setdefault("ph_level", 5.2)
            inputs["temperature_c"] = 37
            inputs["microbe_dose_log"] = 6
            inputs["contact_time_min"] = 1440

        pred = predict_for_inputs(rf, xgb_model, inputs)
        rows.append({
            "strain_code": strain_code,
            "strain_name": strain_name,
            "predicted_reduction_pct": float(np.clip(pred, 0, 100)),
            "strain_description": STRAIN_DESCRIPTIONS[strain_code],
        })

    out = pd.DataFrame(rows).sort_values(
        "predicted_reduction_pct", ascending=False
    ).reset_index(drop=True)
    return out


def get_shap_explanation(rf, X_background, inputs: dict):
    """Return SHAP values for a single prediction using RF."""
    x = pd.DataFrame([[inputs[f] for f in FEATURES]], columns=FEATURES)
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(x)
    # Handle both old (1D) and new (2D) SHAP output shapes
    shap_values = np.array(shap_values)
    if shap_values.ndim == 2:
        shap_values = shap_values[0]
    base_value = explainer.expected_value
    # expected_value may also be array-like in newer versions
    if hasattr(base_value, "__len__"):
        base_value = float(base_value[0]) if len(base_value) > 0 else float(base_value)
    else:
        base_value = float(base_value)
    return shap_values, base_value, x.iloc[0]


def compute_mineral_ratios(initial_phytate_mg: float,
                            reduction_pct: float,
                            iron_mg: float = 4.5,
                            zinc_mg: float = 2.0):
    """Compute final phytate:iron and phytate:zinc molar ratios.

    Molar masses: phytic acid (IP6) = 660 g/mol; iron = 55.85 g/mol; zinc = 65.38 g/mol.
    Clinical thresholds: phytate:iron < 1 for good absorption; phytate:zinc < 15.
    """
    final_phytate_mg = initial_phytate_mg * (1 - reduction_pct / 100.0)
    phytate_mmol = final_phytate_mg / 660.0
    iron_mmol = iron_mg / 55.85
    zinc_mmol = zinc_mg / 65.38

    pi_ratio = phytate_mmol / iron_mmol if iron_mmol > 0 else float("inf")
    pz_ratio = phytate_mmol / zinc_mmol if zinc_mmol > 0 else float("inf")

    return pi_ratio, pz_ratio, final_phytate_mg
