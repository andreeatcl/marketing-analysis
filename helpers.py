import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler


# ======= CAPITOLUL 1 - FUNCTII DE INCARCARE SI CURATARE =======
@st.cache_data
def load_data():
    return pd.read_csv("marketing_campaign.csv", sep="\t")


def remove_iqr_outliers(dataframe, columns):
    if dataframe.empty or not columns:
        return dataframe, 0, pd.DataFrame()

    mask = pd.Series(True, index=dataframe.index)
    bounds = []
    for col in columns:
        q1 = dataframe[col].quantile(0.25)
        q3 = dataframe[col].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            bounds.append({"Feature": col, "Q1": q1, "Q3": q3, "IQR": iqr, "Lower": q1, "Upper": q3})
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask &= dataframe[col].between(lower, upper)
        bounds.append({"Feature": col, "Q1": q1, "Q3": q3, "IQR": iqr, "Lower": lower, "Upper": upper})

    removed = int((~mask).sum())
    bounds_df = pd.DataFrame(bounds)
    return dataframe.loc[mask].copy(), removed, bounds_df


def preprocess_data(df_raw):
    df_clean = df_raw.copy()
    initial_rows, initial_cols = df_clean.shape
    initial_missing = int(df_clean.isna().sum().sum())

    numeric_candidates = [
        "Year_Birth",
        "Income",
        "Kidhome",
        "Teenhome",
        "Recency",
        "MntWines",
        "MntFruits",
        "MntMeatProducts",
        "MntFishProducts",
        "MntSweetProducts",
        "MntGoldProds",
        "NumDealsPurchases",
        "NumWebPurchases",
        "NumCatalogPurchases",
        "NumStorePurchases",
        "NumWebVisitsMonth",
        "AcceptedCmp1",
        "AcceptedCmp2",
        "AcceptedCmp3",
        "AcceptedCmp4",
        "AcceptedCmp5",
        "Response",
        "Complain",
    ]
    for col in [c for c in numeric_candidates if c in df_clean.columns]:
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    if "Dt_Customer" in df_clean.columns:
        df_clean["Dt_Customer"] = pd.to_datetime(df_clean["Dt_Customer"], errors="coerce", dayfirst=True)

    duplicate_count = int(df_clean.duplicated().sum())
    df_clean = df_clean.drop_duplicates().copy()

    if "Income" in df_clean.columns:
        df_clean["Income"] = df_clean["Income"].fillna(df_clean["Income"].median())

    for cat_col in [c for c in ["Education", "Marital_Status"] if c in df_clean.columns]:
        mode_val = df_clean[cat_col].mode(dropna=True)
        fallback = mode_val.iloc[0] if not mode_val.empty else "Unknown"
        df_clean[cat_col] = df_clean[cat_col].fillna(fallback)

    if "Marital_Status" in df_clean.columns:
        atypical_count = int(df_clean["Marital_Status"].isin(["YOLO", "Absurd", "Alone"]).sum())
        df_clean = df_clean[~df_clean["Marital_Status"].isin(["YOLO", "Absurd", "Alone"])].copy()
    else:
        atypical_count = 0

    current_year = pd.Timestamp.today().year
    if "Year_Birth" in df_clean.columns:
        df_clean["Age"] = current_year - df_clean["Year_Birth"]

    spend_cols = [c for c in df_clean.columns if c.startswith("Mnt")]
    if spend_cols:
        df_clean["Amount_Total_Spent"] = df_clean[spend_cols].sum(axis=1)

    purchase_cols = [c for c in ["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases", "NumDealsPurchases"] if c in df_clean.columns]
    if purchase_cols:
        df_clean["Total_Purchases"] = df_clean[purchase_cols].sum(axis=1)

    if {"Kidhome", "Teenhome"}.issubset(df_clean.columns):
        df_clean["Total_Children"] = df_clean["Kidhome"] + df_clean["Teenhome"]

    accepted_cols = [c for c in ["AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5"] if c in df_clean.columns]
    if accepted_cols:
        df_clean["Accepted_Campaigns_Total"] = df_clean[accepted_cols].sum(axis=1)

    if {"Accepted_Campaigns_Total", "Response"}.issubset(df_clean.columns):
        df_clean["Campaign_Engagement_Total"] = df_clean["Accepted_Campaigns_Total"] + df_clean["Response"]

    if "Dt_Customer" in df_clean.columns and df_clean["Dt_Customer"].notna().any():
        max_dt = df_clean["Dt_Customer"].max()
        df_clean["Customer_Since_Days"] = (max_dt - df_clean["Dt_Customer"]).dt.days

    numeric_cols_after_fe = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols_after_fe:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    if "Age" in df_clean.columns:
        df_clean = df_clean[df_clean["Age"].between(18, 100)].copy()

    outlier_cols = [c for c in ["Income", "Amount_Total_Spent", "Age"] if c in df_clean.columns]
    df_clean, removed_outliers, outlier_bounds = remove_iqr_outliers(df_clean, outlier_cols)

    cols_to_drop = [
        "ID",
        "Z_CostContact",
        "Z_Revenue",
        "Year_Birth",
        "Dt_Customer",
        "Kidhome",
        "Teenhome",
        "NumWebPurchases",
        "NumCatalogPurchases",
        "NumStorePurchases",
        "NumDealsPurchases",
        "AcceptedCmp1",
        "AcceptedCmp2",
        "AcceptedCmp3",
        "AcceptedCmp4",
        "AcceptedCmp5",
    ] + spend_cols
    df_clean.drop(columns=cols_to_drop, inplace=True, errors="ignore")

    final_rows, final_cols = df_clean.shape
    final_missing = int(df_clean.isna().sum().sum())

    cleaning_report = {
        "initial_rows": initial_rows,
        "final_rows": final_rows,
        "initial_cols": initial_cols,
        "final_cols": final_cols,
        "initial_missing": initial_missing,
        "final_missing": final_missing,
        "duplicates_removed": duplicate_count,
        "atypical_status_removed": atypical_count,
        "outliers_removed": removed_outliers,
        "outlier_bounds": outlier_bounds,
    }

    return df_clean, cleaning_report


# ======= CAPITOLUL 2 - FUNCTII DE TRANSFORMARE SI VIZUALIZARE =======
def transform_data(df_clean):
    df_transformed = df_clean.copy()

    for col in [c for c in ["Income", "Amount_Total_Spent"] if c in df_transformed.columns]:
        df_transformed[f"{col}_Log"] = np.log1p(df_transformed[col].clip(lower=0))

    numeric_cols = df_transformed.select_dtypes(include=[np.number]).columns.tolist()
    target_col = "Response" if "Response" in df_transformed.columns else None
    scaling_cols = [c for c in numeric_cols if c != target_col]

    scaler = StandardScaler()
    if scaling_cols:
        df_transformed[scaling_cols] = scaler.fit_transform(df_transformed[scaling_cols])

    categorical_cols = df_transformed.select_dtypes(include=["object", "category"]).columns.tolist()
    df_model = pd.get_dummies(df_transformed, columns=categorical_cols, drop_first=True)
    return df_transformed, df_model


def plot_conf_matrix(y_true, y_pred, title):
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    return fig
