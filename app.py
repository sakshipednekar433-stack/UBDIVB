
import io
import base64
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

st.set_page_config(page_title="Universal Bank: Personal Loan Prediction", layout="wide")

# -------------------- Utilities --------------------
@st.cache_data
def load_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df.columns = [c.strip().replace(" ", "").replace(".", "").replace("-", "") for c in df.columns]
    return df

def standardize_columns(df: pd.DataFrame):
    cols = [c.strip().replace(" ", "").replace(".", "").replace("-", "") for c in df.columns]
    df = df.copy()
    df.columns = cols
    return df

def prepare_xy(df: pd.DataFrame, target="PersonalLoan", id_col="ID"):
    df = df.dropna(subset=[target]).copy()
    df[target] = df[target].astype(int)
    if id_col in df.columns:
        df = df.drop_duplicates(subset=[id_col], keep="first")
    features = [c for c in df.columns if c not in [target, id_col]]
    X = df[features].copy()
    y = df[target].copy()
    for c in X.columns:
        if not np.issubdtype(X[c].dtype, np.number):
            X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.fillna(X.median(numeric_only=True))
    return X, y, features

def metrics_table(y_true, y_pred, y_prob):
    return dict(
        Accuracy = accuracy_score(y_true, y_pred),
        Precision = precision_score(y_true, y_pred, zero_division=0),
        Recall = recall_score(y_true, y_pred, zero_division=0),
        F1 = f1_score(y_true, y_pred, zero_division=0),
        AUC = roc_auc_score(y_true, y_prob) if y_prob is not None else np.nan
    )

def download_link(df: pd.DataFrame, filename: str, label: str):
    csv = df.to_csv(index=False).encode()
    b64 = base64.b64encode(csv).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{label}</a>'
    st.markdown(href, unsafe_allow_html=True)

def plot_roc(all_curves, title="ROC Curve (All Models)"):
    fig, ax = plt.subplots(figsize=(6,5))
    for name, (fpr, tpr, auc) in all_curves.items():
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0,1],[0,1], linestyle="--")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(title); ax.legend(loc="lower right"); ax.grid(True, linestyle="--", alpha=0.4)
    st.pyplot(fig)

def plot_confusion(cm, title, cmap="Blues"):
    fig, ax = plt.subplots(figsize=(4.5,3.8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(values_format='d', cmap=cmap, colorbar=False, ax=ax)
    ax.set_title(title); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    st.pyplot(fig)

def feature_importance_bar(model, feature_names, title):
    if not hasattr(model, "feature_importances_"):
        st.info(f"{title}: model has no feature_importances_")
        return
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]
    topk = 15 if len(order) > 15 else len(order)
    order = order[:topk]
    fig, ax = plt.subplots(figsize=(7,5))
    ax.barh(np.array(feature_names)[order][::-1], importances[order][::-1])
    ax.set_xlabel("Importance"); ax.set_title(title)
    st.pyplot(fig)

# -------------------- Sidebar: Data Input --------------------
st.sidebar.header("1) Load Training Data")
uploaded = st.sidebar.file_uploader("Upload Universal Bank CSV (with PersonalLoan & ID columns)", type=["csv"])

if uploaded is None:
    st.sidebar.info("Please upload the training CSV to start.")
    st.title("Universal Bank — Personal Loan Dashboard")
    st.write("Upload your dataset in the sidebar to unlock Insights, Modeling, and Prediction tabs.")
    st.stop()

raw_df = load_csv(uploaded)
df = standardize_columns(raw_df)
target_col = "PersonalLoan"
id_col = "ID"
if target_col not in df.columns:
    st.error("Target column 'PersonalLoan' not found after normalization. Please check your file.")
    st.stop()

X, y, feature_cols = prepare_xy(df, target=target_col, id_col=id_col)
st.sidebar.success(f"Loaded {len(df)} rows, {len(feature_cols)} features.")

# -------------------- Main Tabs --------------------
tab1, tab2, tab3 = st.tabs(["📊 Customer Insights (5 charts)", "🤖 Modeling (3 algorithms + metrics)", "🧮 Predict on New Data"])

# -------------------- Tab 1: Insights --------------------
with tab1:
    st.subheader("Customer Insights for Actionable Targeting")
    st.caption("Tip: Use these segments and patterns to refine campaigns, eligibility, and offer messaging.")

    # 1) Acceptance rate by Income decile
    st.markdown("**1) Acceptance rate by Income decile**")
    try:
        df_copy = df[[target_col, "Income"]].dropna().copy()
        df_copy["income_decile"] = pd.qcut(df_copy["Income"], 10, duplicates='drop')
        rate_by_decile = df_copy.groupby("income_decile")[target_col].mean().reset_index()
        fig, ax = plt.subplots(figsize=(8,4))
        ax.bar(rate_by_decile["income_decile"].astype(str), rate_by_decile[target_col])
        ax.set_xticklabels(rate_by_decile["income_decile"].astype(str), rotation=45, ha="right")
        ax.set_ylabel("Acceptance Rate"); ax.set_title("Income Deciles vs Personal Loan Acceptance")
        st.pyplot(fig)
        st.caption("Higher-income deciles usually show significantly higher uptake; fit premium offers and cross-sell CD/securities.")
    except Exception as e:
        st.warning(f"Could not render chart: {e}")

    # 2) CCAvg vs acceptance — binned trend
    st.markdown("**2) CCAvg (credit card spend) vs acceptance — binned trend**")
    if "CCAvg" in df.columns:
        tmp = df[[target_col, "CCAvg"]].dropna().copy()
        tmp["bin"] = pd.qcut(tmp["CCAvg"], q=20, duplicates="drop")
        tr = tmp.groupby("bin")[target_col].mean().reset_index()
        centers = tmp.groupby("bin")["CCAvg"].mean().values
        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(centers, tr[target_col], marker="o")
        ax.set_xlabel("Avg Monthly CC Spend ($000)"); ax.set_ylabel("Acceptance Rate")
        ax.set_title("CCAvg vs Acceptance (20-quantile bins)")
        st.pyplot(fig)
        st.caption("Rising trend suggests high card spenders are more receptive to loans; test interest-rate personalization for top spenders.")

    # 3) Heatmap — Income decile × Education vs acceptance rate
    st.markdown("**3) Heatmap — Income decile × Education vs acceptance rate**")
    if "Education" in df.columns:
        work = df[[target_col, "Income", "Education"]].dropna().copy()
        work["inc_dec"] = pd.qcut(work["Income"], 8, duplicates="drop")
        pivot = work.pivot_table(index="inc_dec", columns="Education", values=target_col, aggfunc="mean")
        fig, ax = plt.subplots(figsize=(7,5))
        im = ax.imshow(pivot.values, aspect="auto")
        ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index.astype(str))
        ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns.astype(str))
        ax.set_title("Acceptance Heatmap"); ax.set_xlabel("Education"); ax.set_ylabel("Income Decile")
        fig.colorbar(im, ax=ax)
        st.pyplot(fig)
        st.caption("High-income + higher education segments respond best — prime targets for pre-approved offers and digital journeys.")

    # 4) Age vs acceptance (smoothed)
    st.markdown("**4) Age vs acceptance (smoothed)**")
    if "Age" in df.columns:
        a = df[["Age", target_col]].dropna().copy().sort_values("Age")
        grp = a.groupby("Age")[target_col].mean().reset_index()
        grp["smooth"] = grp[target_col].rolling(window=5, min_periods=1, center=True).mean()
        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(grp["Age"], grp["smooth"])
        ax.set_xlabel("Age"); ax.set_ylabel("Acceptance Rate"); ax.set_title("Smoothed Acceptance vs Age")
        st.pyplot(fig)
        st.caption("Middle-aged brackets often show stronger intent; tailor messaging by life-stage (education, renovation, consolidation).")

    # 5) Family × CDAccount — adoption composition
    st.markdown("**5) Family size × CDAccount — adoption composition**")
    if "Family" in df.columns and "CDAccount" in df.columns:
        g = df.groupby(["Family", "CDAccount"])[target_col].mean().reset_index()
        fig, ax = plt.subplots(figsize=(8,4))
        fams = sorted(df["Family"].dropna().unique())
        vals0 = [g[(g["Family"]==f)&(g["CDAccount"]==0)][target_col].mean() for f in fams]
        vals1 = [g[(g["Family"]==f)&(g["CDAccount"]==1)][target_col].mean() for f in fams]
        idx = np.arange(len(fams))
        ax.bar(idx, vals0, label="CDAccount=0")
        ax.bar(idx, vals1, bottom=vals0, label="CDAccount=1")
        ax.set_xticks(idx); ax.set_xticklabels([str(int(x)) for x in fams])
        ax.set_xlabel("Family Size"); ax.set_ylabel("Acceptance Rate (stacked)")
        ax.set_title("Family × CDAccount — Acceptance Composition")
        ax.legend()
        st.pyplot(fig)
        st.caption("Households with CDs show higher acceptance; bundle offers (CD + personal loan) for bigger households.")
    st.success("Insights ready. Move to 'Modeling' to train and compare algorithms.")

# -------------------- Tab 2: Modeling --------------------
with tab2:
    st.subheader("Train & Compare Algorithms")
    colA, colB = st.columns(2)
    with colA:
        test_size = st.slider("Test size", 0.1, 0.5, 0.3, 0.05)
    with colB:
        cv_folds = st.slider("Cross-Validation folds", 3, 10, 5, 1)

    if st.button("Run Decision Tree, Random Forest, Gradient Boosting"):
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)

        models = {
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        }

        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        rows = []
        rocs = {}
        confs = {}

        for name, model in models.items():
            acc_scores = cross_val_score(model, X_tr, y_tr, cv=cv, scoring="accuracy", n_jobs=1)
            auc_scores = cross_val_score(model, X_tr, y_tr, cv=cv, scoring="roc_auc", n_jobs=1)

            model.fit(X_tr, y_tr)
            ytr_pred = model.predict(X_tr)
            yte_pred = model.predict(X_te)
            ytr_prob = model.predict_proba(X_tr)[:,1] if hasattr(model,"predict_proba") else None
            yte_prob = model.predict_proba(X_te)[:,1] if hasattr(model,"predict_proba") else None

            m_train = metrics_table(y_tr, ytr_pred, ytr_prob)
            m_test  = metrics_table(y_te, yte_pred, yte_prob)

            rows.append({
                "Algorithm": name,
                "Training Accuracy": m_train["Accuracy"],
                "Testing Accuracy": m_test["Accuracy"],
                "Precision": m_test["Precision"],
                "Recall": m_test["Recall"],
                "F1 Score": m_test["F1"],
                "AUC (Test)": m_test["AUC"],
                "CV Acc (mean)": np.mean(acc_scores),
                "CV AUC (mean)": np.mean(auc_scores)
            })

            if yte_prob is not None:
                fpr, tpr, _ = roc_curve(y_te, yte_prob)
                rocs[name] = (fpr, tpr, m_test["AUC"])
            confs[name] = {
                "train": confusion_matrix(y_tr, ytr_pred),
                "test": confusion_matrix(y_te, yte_pred),
                "model": model
            }

        metrics_df = pd.DataFrame(rows).set_index("Algorithm").round(4)
        st.dataframe(metrics_df, use_container_width=True)
        download_link(metrics_df.reset_index(), "model_metrics.csv", "⬇️ Download metrics CSV")

        st.markdown("#### ROC Curve (All Models)")
        plot_roc(rocs)

        st.markdown("#### Confusion Matrices")
        for name in confs:
            c1, c2 = st.columns(2)
            with c1: plot_confusion(confs[name]["train"], f"{name} — TRAIN", cmap="Blues")
            with c2: plot_confusion(confs[name]["test"], f"{name} — TEST", cmap="Oranges")

        st.markdown("#### Feature Importances")
        for name in confs:
            feature_importance_bar(confs[name]["model"], feature_cols, f"{name} — Feature Importances")

# -------------------- Tab 3: Predict New Data --------------------
with tab3:
    st.subheader("Upload New Customer File for Predictions")
    st.caption("The app will train a Gradient Boosting model on the training data in the sidebar, then score the new file.")
    new_file = st.file_uploader("Upload new CSV (must contain same feature columns; ID optional)", type=["csv"], key="newfile")

    if new_file is not None:
        new_df = load_csv(new_file)
        new_df = standardize_columns(new_df)

        model = GradientBoostingClassifier(random_state=42)
        model.fit(X, y)

        feat_cols_present = [c for c in feature_cols if c in new_df.columns]
        missing = [c for c in feature_cols if c not in new_df.columns]
        if len(missing) > 0:
            st.warning(f"The following expected feature columns are missing and will be filled with training medians: {missing}")
            tr_medians = pd.DataFrame(X).median(numeric_only=True)
            for m in missing:
                new_df[m] = tr_medians.get(m, 0)

        X_new = new_df[feature_cols].copy()
        for c in X_new.columns:
            if not np.issubdtype(X_new[c].dtype, np.number):
                X_new[c] = pd.to_numeric(X_new[c], errors="coerce")
        X_new = X_new.fillna(pd.DataFrame(X).median(numeric_only=True))

        probs = model.predict_proba(X_new)[:,1]
        preds = (probs >= 0.5).astype(int)
        out = new_df.copy()
        out["PersonalLoan_Pred"] = preds
        out["PersonalLoan_Prob"] = probs

        st.success("Predictions generated.")
        st.dataframe(out.head(50), use_container_width=True)
        download_link(out, "predictions_with_labels.csv", "⬇️ Download predictions CSV")
