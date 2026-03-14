# app/app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix
from xgboost import XGBClassifier


# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🚀",
    layout="wide"
)

# ================= CUSTOM CSS =================
st.markdown("""
<style>
.hero {
    background: linear-gradient(135deg, #141E30, #243B55);
    padding: 70px;
    border-radius: 20px;
    text-align: center;
    color: white;
}
.hero h1 { font-size: 50px; }
.hero p { font-size: 22px; opacity: 0.9; }

.card {
    background-color: #ffffff;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0px 6px 18px rgba(0,0,0,0.08);
    text-align: center;
}

.footer {
    text-align:center;
    font-size:14px;
    color:gray;
    padding:30px;
}
</style>
""", unsafe_allow_html=True)

# ================= SIDEBAR =================
st.sidebar.title("📌 Navigation")
menu = st.sidebar.radio("", ["🏠 Home", "📊 Train & Predict"])

# ================= HOME PAGE =================
if menu == "🏠 Home":

    st.markdown("""
        <div class="hero">
            <h1> Customer Churn Predictor</h1>
            <p>Generalized Machine Learning System for Predicting Customer Behavior</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("## 🔍 Why This System?")

    col1, col2, col3 = st.columns(3)

    col1.markdown("""
        <div class="card">
            <h3>⚡ Dynamic Model Training</h3>
            <p>Train ML models on any dataset without hardcoding features.</p>
        </div>
    """, unsafe_allow_html=True)

    col2.markdown("""
        <div class="card">
            <h3>📊 Automated Preprocessing</h3>
            <p>Handles numeric & categorical data automatically using pipelines.</p>
        </div>
    """, unsafe_allow_html=True)

    col3.markdown("""
        <div class="card">
            <h3>📈 Business Insights</h3>
            <p>Visualize churn trends & risk segments clearly for stakeholders.</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.success("👉 Navigate to Train & Predict to start analysis.")


# ================= TRAIN & PREDICT =================
elif menu == "📊 Train & Predict":

    st.header("📁 Upload Dataset")
    uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

    if uploaded_file:

        df = pd.read_csv(uploaded_file)
        st.subheader("📋 Dataset Preview")
        st.dataframe(df.head())

        valid_targets = [col for col in df.columns if df[col].nunique() < 50]
        target_column = st.selectbox("🎯 Select Target Column", valid_targets)

        if st.button("🚀 Train Model"):

            X = df.drop(target_column, axis=1)
            y = df[target_column]

            if y.dtype == "object":
                y = y.astype("category").cat.codes

            numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
            categorical_features = X.select_dtypes(include=["object"]).columns

            numeric_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            categorical_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ])

            preprocessor = ColumnTransformer([
                ("num", numeric_pipeline, numeric_features),
                ("cat", categorical_pipeline, categorical_features)
            ])

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Random Forest": RandomForestClassifier(),
                "XGBoost": XGBClassifier(eval_metric="logloss")
            }

            best_model = None
            best_score = 0
            best_name = ""

            for name, model in models.items():

                pipeline = Pipeline([
                    ("preprocessor", preprocessor),
                    ("model", model)
                ])

                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)

                f1 = f1_score(y_test, y_pred, average="weighted")

                if f1 > best_score:
                    best_score = f1
                    best_model = pipeline
                    best_name = name

            st.success(f"🏆 Best Model Selected: {best_name}")

            # ================= SMALL CONFUSION MATRIX =================
            st.subheader("📌 Confusion Matrix")

            y_pred_best = best_model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred_best)

            fig, ax = plt.subplots(figsize=(2.5, 2.5))  # smaller size

            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                cbar=False,
                annot_kws={"size": 10},  # smaller numbers
                ax=ax
            )

            ax.set_xlabel("Predicted", fontsize=9)
            ax.set_ylabel("Actual", fontsize=9)
            ax.tick_params(labelsize=8)

            plt.tight_layout()
            st.pyplot(fig)

            # ================= FULL DATA PREDICTION =================
            best_model.fit(X, y)
            full_preds = best_model.predict(X)

            result_df = df.copy()
            result_df["Prediction"] = full_preds

            if y.nunique() == 2:
                probs = best_model.predict_proba(X)[:, 1]
                result_df["Probability"] = probs
                result_df["Label"] = result_df["Prediction"].map(
                    {0: "Stay", 1: "Leave"}
                )

            # ================= KPI METRICS =================
            total_customers = len(result_df)
            leaving_customers = int(result_df["Prediction"].sum())
            staying_customers = total_customers - leaving_customers
            churn_rate = round((leaving_customers / total_customers) * 100, 2)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("👥 Total Customers", total_customers)
            col2.metric("🚨 Customers Leaving", leaving_customers)
            col3.metric("✅ Customers Staying", staying_customers)
            col4.metric("📊 Churn Rate (%)", churn_rate)

            # ================= PIE CHART =================
            st.subheader("📊 Customer Distribution")

            pie = px.pie(
                result_df,
                names="Label",
                title="Stay vs Leave Percentage",
                color="Label",
                color_discrete_map={"Stay": "#2ecc71", "Leave": "#e74c3c"}
            )
            st.plotly_chart(pie, use_container_width=True)

            # ================= FEATURE IMPORTANCE =================
            if hasattr(best_model.named_steps["model"], "feature_importances_"):

                st.subheader("📈 Feature Importance")

                feature_names = best_model.named_steps["preprocessor"].get_feature_names_out()
                importances = best_model.named_steps["model"].feature_importances_

                fi_df = pd.DataFrame({
                    "Feature": feature_names,
                    "Importance": importances
                }).sort_values(by="Importance", ascending=False).head(10)

                fig2 = px.bar(
                    fi_df,
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    title="Top 10 Important Features"
                )
                st.plotly_chart(fig2, use_container_width=True)

            # ================= TABLE =================
            st.subheader("📋 Prediction Results")
            st.dataframe(result_df.head(20))

            st.session_state["result_df"] = result_df

        # ================= DOWNLOAD =================
        if "result_df" in st.session_state:

            st.markdown("### 📥 Download Predictions")
            csv = st.session_state["result_df"].to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download Prediction Results as CSV",
                data=csv,
                file_name="prediction_results.csv",
                mime="text/csv"
            )

    else:
        st.info("Upload a dataset to begin.")

# ================= FOOTER =================
st.markdown("""
<div class="footer">
Built with ❤️ using Machine Learning & Streamlit | AI Customer Analytics Dashboard
</div>
""", unsafe_allow_html=True)