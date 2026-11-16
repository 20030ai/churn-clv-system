import pandas as pd
from src.features import clean, build_model_df, encode, FEATURES, TARGET

def raw_rows():
    return pd.DataFrame({
        "gender": ["Male", "Female"], "senior_citizen": ["No", "Yes"],
        "partner": ["Yes", "No"], "dependents": ["No", "No"], "tenure_months": [0, 12],
        "phone_service": ["No", "Yes"], "multiple_lines": ["No phone service", "Yes"],
        "internet_service": ["No", "Fiber optic"],
        "online_security": ["No internet service", "Yes"],
        "online_backup": ["No internet service", "No"],
        "device_protection": ["No internet service", "No"],
        "tech_support": ["No internet service", "Yes"],
        "streaming_tv": ["No internet service", "No"],
        "streaming_movies": ["No internet service", "No"],
        "contract": ["Month-to-month", "Two year"], "paperless_billing": ["Yes", "No"],
        "payment_method": ["Mailed check", "Credit card (automatic)"],
        "monthly_charges": [20.0, 90.5], "total_charges": [" ", "1086.0"],
        "churn_value": [0, 1]})

def test_blank_total_charges_become_zero():
    out = clean(raw_rows())
    assert out["total_charges"].iloc[0] == 0 and out["total_charges"].dtype.kind == "f"

def test_no_service_values_removed():
    out = clean(raw_rows())
    assert "No phone service" not in out["multiple_lines"].values
    assert "No internet service" not in out["tech_support"].values

def test_only_chosen_columns_and_no_leaks():
    out = build_model_df(raw_rows())
    assert list(out.columns) == FEATURES + [TARGET]

def test_encode_numeric_and_aligned():
    X = build_model_df(raw_rows()).drop(columns=TARGET)
    enc = encode(X)
    assert all(t.kind in "iufb" for t in enc.dtypes)
    cols = list(enc.columns) + ["payment_method_Electronic check"]
    assert list(encode(X, cols).columns) == cols