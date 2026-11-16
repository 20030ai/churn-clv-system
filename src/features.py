import pandas as pd

ADDONS = ["online_security", "online_backup", "device_protection",
          "tech_support", "streaming_tv", "streaming_movies"]
YES_NO = ["senior_citizen", "partner", "dependents", "phone_service",
          "multiple_lines", "paperless_billing"] + ADDONS
FEATURES = ["gender", "senior_citizen", "partner", "dependents", "tenure_months",
            "phone_service", "multiple_lines", "internet_service", *ADDONS,
            "contract", "paperless_billing", "payment_method",
            "monthly_charges", "total_charges", "n_services", "has_autopay"]
TARGET = "churn_value"

def clean(df):
    df = df.copy()
    df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce").fillna(0)
    df["multiple_lines"] = df["multiple_lines"].replace({"No phone service": "No"})
    df[ADDONS] = df[ADDONS].replace({"No internet service": "No"})
    return df

def add_features(df):
    df = df.copy()
    df["n_services"] = (df[ADDONS] == "Yes").sum(axis=1)
    df["has_autopay"] = df["payment_method"].str.contains("automatic").astype(int)
    return df

def build_model_df(raw):
    return add_features(clean(raw))[FEATURES + [TARGET]]

def encode(X, columns=None):
    X = X.copy()
    X[YES_NO] = (X[YES_NO] == "Yes").astype(int)
    X["gender"] = (X["gender"] == "Male").astype(int)
    X["contract"] = X["contract"].map(
        {"Month-to-month": 0, "One year": 1, "Two year": 2})
    X = pd.get_dummies(X, columns=["internet_service", "payment_method"], dtype=int)
    return X if columns is None else X.reindex(columns=columns, fill_value=0)
