import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split

from helpers import load_data, preprocess_data, transform_data, plot_conf_matrix

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

st.set_page_config(page_title="Analiza Campaniei de Marketing", layout="wide")

st.markdown(
    """
<style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        margin-bottom: 0.4rem;
    }
    .subtitle {
        color: #6b7280;
        margin-bottom: 1rem;
    }
    .block {
        border: 1px solid rgba(120, 120, 120, 0.18);
        border-radius: 14px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ======= CAPITOLUL 0 - INTRODUCERE SI INCARCARE DATE =======


st.markdown('<div class="main-title">Analiza Campaniei de Marketing</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Modelare predictiva si analiza comportamentului clientilor</div>',
    unsafe_allow_html=True,
)

st.info(
    "Setul de date utilizat descrie profilul clientilor, istoricul de achizitii si reactia la campaniile de marketing.\n\n"
    "Analiza urmareste doua obiective. Primul obiectiv este intelegerea tiparelor de consum si a segmentelor de clienti cu comportament similar. "
    "Al doilea obiectiv este estimarea probabilitatii de raspuns la campanie prin metode de clasificare, impreuna cu o analiza de regresie pentru cheltuiala totala.\n\n"
    "Procesul include curatare de date, transformari statistice, analiza exploratorie si comparatie intre modele, cu accent pe interpretare obiectiva a rezultatelor."
)

try:
    df = load_data()
except FileNotFoundError:
    st.error("Fisierul marketing_campaign.csv nu a fost gasit in folderul proiectului.")
    st.stop()

# ======= CAPITOLUL 1 SI 2 - EXPLORAREA INITIALA A DATELOR =======
st.header("1 & 2. Explorarea datelor initiale")
col_a, col_b, col_c = st.columns(3)
col_a.metric("Numar randuri", f"{df.shape[0]:,}")
col_b.metric("Numar coloane", f"{df.shape[1]}")
col_c.metric("Valori lipsa totale", f"{int(df.isna().sum().sum()):,}")

with st.expander("Preview date brute", expanded=True):
    st.dataframe(df.head(10), use_container_width=True)

# ======= CAPITOLUL 3 - PREPROCESAREA SI CURATAREA DATELOR =======
st.header("3. Preprocesarea si curatarea datelor")
df_clean, cleaning_report = preprocess_data(df)

pre_col1, pre_col2, pre_col3 = st.columns(3)
pre_col1.metric("Randuri dupa curatare", f"{df_clean.shape[0]:,}")
pre_col2.metric("Coloane ramase", f"{df_clean.shape[1]:,}")
pre_col3.metric("Valori lipsa ramase", f"{int(df_clean.isna().sum().sum()):,}")

st.write(
    "Curatarea include eliminarea duplicatelor, excluderea categoriilor atipice din statusul marital, "
    "imputare pentru valori lipsa si eliminare de outliers prin metoda intercuartilica pe Income, Amount_Total_Spent si Age."
)

with st.expander("Preview date curate", expanded=False):
    st.dataframe(df_clean.head(10), use_container_width=True)

# ======= CAPITOLUL 4 - TRANSFORMAREA DATELOR =======
st.header("4. Transformarea datelor")
df_transformed, df_model = transform_data(df_clean)

st.markdown(
    """
Transformari aplicate in termeni generali:
- transformare logaritmica pentru variabilele cu distributie asimetrica
- scalare standard pentru variabilele numerice
- codificare one hot pentru variabilele categorice
"""
)

st.write("Esantion din setul transformat prin scalare, transformari si codificare one hot:")
st.dataframe(df_model.head(8), use_container_width=True)

# ======= CAPITOLUL 5 - ANALIZA EXPLORATORIE =======
st.header("5. Analiza exploratorie extinsa")
preview_cols = [c for c in ["Age", "Income", "Amount_Total_Spent", "Response"] if c in df_clean.columns]
if preview_cols:
    eda_preview = df_clean[preview_cols].describe(include="all").T
    st.write("Mini preview rezultate pentru variabilele principale:")
    st.dataframe(eda_preview, use_container_width=True)

tab1, tab2, tab3, tab4 = st.tabs(
    ["Distributii", "Corelatii", "Comportament clienti", "Campanii & Canale"]
)

with tab1:
    dist_candidates = [c for c in ["Age", "Income", "Amount_Total_Spent", "Recency"] if c in df_clean.columns]
    if dist_candidates:
        fig, axes = plt.subplots(1, len(dist_candidates), figsize=(5 * len(dist_candidates), 4))
        if len(dist_candidates) == 1:
            axes = [axes]
        for ax, col in zip(axes, dist_candidates):
            sns.histplot(df_clean[col], kde=True, ax=ax, color="#3b82f6")
            ax.set_title(f"Distributie {col}")
        st.pyplot(fig)

    if {"Education", "Amount_Total_Spent"}.issubset(df_clean.columns):
        fig, ax = plt.subplots(figsize=(9, 4.5))
        edu_order = (
            df_clean.groupby("Education")["Amount_Total_Spent"]
            .median()
            .sort_values(ascending=False)
            .index
        )
        sns.boxplot(data=df_clean, x="Education", y="Amount_Total_Spent", order=edu_order, ax=ax)
        ax.tick_params(axis="x", rotation=35)
        ax.set_title("Distributia cheltuielilor pe nivel de educatie")
        st.pyplot(fig)

with tab2:
    numeric_df = df_clean.select_dtypes(include=[np.number])
    if numeric_df.shape[1] >= 2:
        corr = numeric_df.corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Matrice de corelatie")
        st.pyplot(fig)

        top_corr = (
            corr["Amount_Total_Spent"].drop("Amount_Total_Spent").abs().sort_values(ascending=False).head(10)
            if "Amount_Total_Spent" in corr.columns
            else pd.Series(dtype=float)
        )
        if not top_corr.empty:
            st.write("Top 10 corelatii absolute cu Amount_Total_Spent:")
            st.dataframe(top_corr.rename("|corr|").to_frame())

with tab3:
    if {"Income", "Amount_Total_Spent"}.issubset(df_clean.columns):
        fig, ax = plt.subplots(figsize=(8, 5))
        hue_col = "Response" if "Response" in df_clean.columns else None
        sns.scatterplot(data=df_clean, x="Income", y="Amount_Total_Spent", hue=hue_col, alpha=0.7, ax=ax)
        ax.set_title("Income vs Amount_Total_Spent")
        st.pyplot(fig)

    group_cols = [c for c in ["Marital_Status", "Education"] if c in df_clean.columns]
    for grp_col in group_cols:
        if "Amount_Total_Spent" in df_clean.columns:
            spend_by_group = (
                df_clean.groupby(grp_col)["Amount_Total_Spent"].mean().sort_values(ascending=False).reset_index()
            )
            st.write(f"Cheltuiala medie dupa {grp_col}:")
            st.dataframe(spend_by_group, use_container_width=True)

with tab4:
    campaign_aggr_cols = [
        c for c in ["Accepted_Campaigns_Total", "Response", "Campaign_Engagement_Total"] if c in df_clean.columns
    ]
    if campaign_aggr_cols:
        campaign_totals = df_clean[campaign_aggr_cols].sum().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x=campaign_totals.index, y=campaign_totals.values, ax=ax, palette="viridis")
        ax.set_title("Indicatori agregati de campanie")
        ax.tick_params(axis="x", rotation=20)
        st.pyplot(fig)

    if "Total_Purchases" in df_clean.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(df_clean["Total_Purchases"], kde=True, ax=ax, color="#a21caf")
        ax.set_title("Distributie Total_Purchases")
        st.pyplot(fig)

    # ======= CAPITOLUL 6 - MODELARE PREDICTIVA =======
st.header("6. Modelare Predictiva")

st.subheader("6.1 Segmentarea clientilor prin KMeans")
st.write(
    "Preview segmentare: KMeans grupeaza clientii cu profiluri similare pe baza variabilelor de venit, cheltuiala si varsta."
)
cluster_features = [c for c in ["Income", "Amount_Total_Spent", "Age"] if c in df_transformed.columns]
if len(cluster_features) >= 2:
    X_cluster = df_transformed[cluster_features]

    silhouette_results = []
    k_range = range(2, 7)
    for k in k_range:
        km_tmp = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels_tmp = km_tmp.fit_predict(X_cluster)
        silhouette_results.append((k, silhouette_score(X_cluster, labels_tmp)))

    best_k = max(silhouette_results, key=lambda x: x[1])[0]
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df_clean["Cluster"] = kmeans.fit_predict(X_cluster)

    col1, col2 = st.columns(2)
    col1.metric("Numar clustere ales", best_k)
    col2.metric("Silhouette score", f"{max(silhouette_results, key=lambda x: x[1])[1]:.3f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        data=df_clean,
        x="Income",
        y="Amount_Total_Spent",
        hue="Cluster",
        palette="viridis",
        alpha=0.75,
        ax=ax,
    )
    ax.set_title("Clustere clienti: Income vs Amount_Total_Spent")
    st.pyplot(fig)

    cluster_profile = df_clean.groupby("Cluster")[cluster_features].mean().round(2)
    st.write("Profil mediu pe cluster:")
    st.dataframe(cluster_profile, use_container_width=True)
else:
    st.warning("Nu exista suficiente variabile numerice pentru clustering.")

st.subheader("6.2 Clasificare - raspuns la campanie")
st.write(
    "Preview clasificare: se estimeaza probabilitatea de raspuns la campanie prin comparatie intre mai multe modele de clasificare."
)
if "Response" in df_model.columns:
    X = df_model.drop(columns=["Response", "Campaign_Engagement_Total"], errors="ignore")
    y = df_model["Response"].astype(int)

    class_counts = y.value_counts(dropna=False)
    if y.nunique() < 2:
        st.warning("Nu exista suficiente clase in 'Response' pentru clasificare (toate valorile sunt identice).")
    else:
        stratify_target = y if class_counts.min() >= 2 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=stratify_target
        )

        models = {
            "LogisticRegression": LogisticRegression(max_iter=3000, class_weight="balanced"),
            "RandomForest": RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                class_weight="balanced",
                min_samples_leaf=2,
            ),
        }

        if XGBClassifier is not None:
            pos_count = int((y_train == 1).sum())
            neg_count = int((y_train == 0).sum())
            scale_pos_weight = (neg_count / pos_count) if pos_count > 0 else 1.0
            models["XGBoost"] = XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",
                random_state=42,
                scale_pos_weight=scale_pos_weight,
            )
        else:
            st.info("XGBoost nu este instalat in mediu; rulez comparatia cu LogisticRegression si RandomForest.")

        eval_rows = []
        predictions = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
            predictions[name] = (model, y_pred, y_prob)

            try:
                roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan
            except ValueError:
                roc_auc = np.nan

            eval_rows.append(
                {
                    "Model": name,
                    "Accuracy": accuracy_score(y_test, y_pred),
                    "Precision": precision_score(y_test, y_pred, zero_division=0),
                    "Recall": recall_score(y_test, y_pred, zero_division=0),
                    "F1": f1_score(y_test, y_pred, zero_division=0),
                    "ROC_AUC": roc_auc,
                }
            )

        eval_df = pd.DataFrame(eval_rows).sort_values("ROC_AUC", ascending=False)
        st.dataframe(eval_df.style.format({
            "Accuracy": "{:.3f}",
            "Precision": "{:.3f}",
            "Recall": "{:.3f}",
            "F1": "{:.3f}",
            "ROC_AUC": "{:.3f}",
        }), use_container_width=True)

        best_model_name = eval_df.iloc[0]["Model"]
        best_model, best_pred, _ = predictions[best_model_name]

        best_row = eval_df.iloc[0]
        st.caption(
            f"Rezultat: modelul cu performanta cea mai buna este {best_model_name} "
            f"(F1={best_row['F1']:.3f}, ROC_AUC={best_row['ROC_AUC']:.3f})."
        )

        st.write(f"Clasificare detaliata pentru modelul cel mai bun: **{best_model_name}**")
        st.text(classification_report(y_test, best_pred, zero_division=0))
        st.pyplot(plot_conf_matrix(y_test, best_pred, f"Confusion Matrix - {best_model_name}"))

        cm = confusion_matrix(y_test, best_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            st.write("Interpretare succinta pentru matricea de confuzie:")
            st.markdown(f"- Clasificari corecte pentru clasa 0: **{tn}**")
            st.markdown(f"- Clasificari corecte pentru clasa 1: **{tp}**")
            st.markdown(f"- Fals pozitive: **{fp}**")
            st.markdown(f"- Fals negative: **{fn}**")
            st.markdown(
                "- Modelul este adecvat cand reduce simultan fals pozitive si fals negative, "
                "in functie de obiectivul operational al campaniei."
            )

        if best_model_name in ["RandomForest", "XGBoost"]:
            importances = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False).head(15)
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(x=importances.values, y=importances.index, ax=ax, palette="crest")
            ax.set_title(f"Top 15 feature-uri importante ({best_model_name})")
            st.pyplot(fig)
else:
    st.warning("Coloana tinta 'Response' nu exista; clasificarea nu poate fi rulata.")

st.subheader("6.3 Regresie Multipla OLS")
ols_features = [c for c in ["Income", "Age", "Total_Children", "Total_Purchases", "Recency"] if c in df_transformed.columns]
if "Amount_Total_Spent" in df_transformed.columns and len(ols_features) >= 2:
    X_ols = sm.add_constant(df_transformed[ols_features])
    y_ols = df_transformed["Amount_Total_Spent"]
    ols_model = sm.OLS(y_ols, X_ols).fit()
    st.text(ols_model.summary().as_text())

    params = ols_model.params.drop("const", errors="ignore")
    pvals = ols_model.pvalues.drop("const", errors="ignore")
    significant = pvals[pvals < 0.05].sort_values()
    top_effects = params.abs().sort_values(ascending=False).head(3).index.tolist()

    pos_sig = [idx for idx in significant.index if params.get(idx, 0) > 0]
    neg_sig = [idx for idx in significant.index if params.get(idx, 0) < 0]

    st.write("Interpretare OLS in contextul setului analizat:")
    st.markdown(f"- Variabila tinta este **Amount_Total_Spent**, explicata prin **{', '.join(ols_features)}**.")
    st.markdown(f"- R-squared este **{ols_model.rsquared:.3f}**, ceea ce indica proportia variatiei cheltuielii totale explicata de model.")
    st.markdown(f"- Predictori semnificativi statistic la prag 0.05: **{len(significant)}**.")
    if top_effects:
        st.markdown(f"- Variabile cu efect estimat puternic in model: **{', '.join(top_effects)}**.")
    if pos_sig:
        st.markdown(f"- Predictori semnificativi cu efect pozitiv: **{', '.join(pos_sig)}**.")
    if neg_sig:
        st.markdown(f"- Predictori semnificativi cu efect negativ: **{', '.join(neg_sig)}**.")
else:
    st.warning("Nu exista suficiente variabile pentru regresia OLS.")