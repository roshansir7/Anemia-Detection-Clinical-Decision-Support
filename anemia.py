import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import ExtraTreesClassifier
import joblib, warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Classification and Detection of Anemia", page_icon="🩸", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_excel("Anemia Dataset.xlsx")
    return df.replace({'Gender': {'Male': 0, 'Female': 1, 'M': 0, 'F': 1, 'm': 0, 'f': 1},
                       'Decision_Class': {'Non-Anemic': 0, 'Anemic': 1}})

@st.cache_resource
def load_model():
    return joblib.load("best_decision_tree_model.pkl")

# Sidebar
logo = "blood-drop.png"
size = (100, 100)
st.sidebar.image(logo, width=size[0])
st.sidebar.markdown("<h2>Data Science Project</h2>", unsafe_allow_html=True)
page = st.sidebar.radio("", ["🏠 Welcome", "📂 Data Overview", "📊 Visualizations", "🔮 Model Evaluation and Prediction"], index=0)
color_map = {"🏠 Welcome": "blue", "📂 Data Overview": "green", "📊 Visualizations": "orange", "🔮 Model Evaluation and Prediction": "purple"}
# Upload dataset
st.sidebar.markdown("<h4>Upload Dataset</h4>", unsafe_allow_html=True)
uploaded = st.sidebar.file_uploader("Upload your anemia dataset (Excel or CSV)", type=["xlsx", "csv"])
df = pd.read_csv(uploaded) if uploaded and uploaded.name.endswith(".csv") else (pd.read_excel(uploaded) if uploaded else load_data())
df = df.replace({'Gender': {'Male': 0, 'Female': 1, 'm': 0, 'f': 1}, 'Decision_Class': {'Non-Anemic': 0, 'Anemic': 1}})
model = load_model()
st.sidebar.markdown("---")
st.sidebar.markdown("""<div style=" bottom:10px; left:15px;,color:gray;">
<small>Made by <strong>Roshan Paudel</strong><br>Student ID: <strong>B01810761</strong></small></div>""", unsafe_allow_html=True)

# Welcome
if page == "🏠 Welcome":
    st.title("🩸Predictive Modeling and Clinical Decision Support for Anemia Detection Using Machine Learning")
    st.markdown("""
    - This project focuses on the **classification and detection of anemia** using machine learning.  
    - People with anemia are labeled as **1**, while healthy people are labeled as **0**.  
    - The dataset contains approximately **1000 patient records**.

    ---
    ### Project Highlights
    - Load and preprocess the anemia dataset  
    - Explore data through visualizations (distributions, correlations, gender impact)  
    - Evaluate multiple ML models (Decision Tree, Random Forest, XGBoost, etc.)  
    - Compare model performance using **Accuracy**, **Precision**, **Recall**, **F1 Score**  
    - Tune hyperparameters with **Grid Search CV**  
    ---
    ### Dataset
    The dataset includes a numbers of features related to anemia diagnosis, such as blood parameters and demographic information.
     Further details can be explored in the Data Overview and EDA sections.

     ### Objectives
     To develop a accurate machine learning model that can help in the early detection of anemia, supporting clinical decision-making and improving patient care.
    
    ### Instructions
    Use the sidebar to navigate between different sections of the app.  
    You can also upload your own anemia dataset (Excel or CSV) in the sidebar to analyze and predict using the trained model.  
    Enjoy exploring the data and the model!
    
     """)

# Data Overview
elif page == "📂 Data Overview":
    st.title("Data Overview")
    has_dc = "Decision_Class" in df.columns
    rows, cols = df.shape
    total_missing = df.isnull().sum().sum()
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{rows:,}"); c2.metric("Columns", f"{cols}"); c3.metric("Missing Values", f"{int(total_missing):,}")
    st.markdown("---")
    if has_dc:
        counts = df["Decision_Class"].value_counts().to_dict()
        anemic, non_anemic = counts.get(1, 0), counts.get(0, 0)
        total = max(anemic + non_anemic, 1)
        c4, c5 = st.columns(2)
        c4.metric("Anemic (1)", f"{anemic:,}", f"{(anemic/total)*100:.1f}%")
        c5.metric("Non-Anemic (0)", f"{non_anemic:,}", f"{(non_anemic/total)*100:.1f}%")
    st.markdown("---")
    st.subheader("🗂️ Dataset Preview")
    n = st.number_input("Rows to display", 5, len(df), 10, step=5)
    st.dataframe(df.head(n), use_container_width=True)
    st.subheader("📊 Statistical Summary")
    st.dataframe(df.describe().T, use_container_width=True)
    st.subheader("ℹ️ Data Info")
    info = pd.DataFrame({
        "Column": df.columns, "Non-Null Count": len(df) - df.isnull().sum(),
        "Null Count": df.isnull().sum(), "Dtype": df.dtypes.astype(str)
    })
    info["% Missing"] = (info["Null Count"] / len(df) * 100).round(2)
    st.dataframe(info, use_container_width=True)

# EDA
elif page == "📊 Visualizations":
    st.title("Exploratory Data Analysis (EDA)")
    has_dc = "Decision_Class" in df.columns

    # Summary metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", len(df))
    c2.metric("Columns", len(df.columns))
    if has_dc:
        c3.metric("Anemic Cases", int((df["Decision_Class"] == 1).sum()))
        c4.metric("Non-Anemic Cases", int((df["Decision_Class"] == 0).sum()))
    st.markdown("---")

    # Selection options
    c1, c2 = st.columns([1.2, 1])
    with c1:
        view = st.radio("Select View:", ["Anemia vs Non-Anemia", "Gender vs Anemia", "Feature Distribution"], horizontal=True)
    with c2:
        chart_type = st.radio("Select Chart Type:", ["Bar", "Pie", "Box"] if view != "Feature Distribution" else ["Bar", "Box"], horizontal=True)
    st.markdown("---")

    #Anemia vs Non-Anemia
    if view == "Anemia vs Non-Anemia" and has_dc:
        counts = df["Decision_Class"].value_counts().reset_index()
        counts.columns = ["Class", "Count"]
        counts["Class"] = counts["Class"].replace({0: "Non-Anemic", 1: "Anemic"})
        if chart_type == "Bar":
            fig = px.bar(counts, x="Class", y="Count", color="Class", text="Count",
                         color_discrete_map={"Anemic": "#ff6666", "Non-Anemic": "#66b3ff"})
        elif chart_type == "Pie":
            fig = px.pie(counts, values="Count", names="Class", color="Class",
                         color_discrete_map={"Anemic": "#ff6666", "Non-Anemic": "#66b3ff"}, hole=0.35)
        else:
            fig = px.box(df, x="Decision_Class", y="Hb", color="Decision_Class",
                         color_discrete_map={0: "#66b3ff", 1: "#ff6666"})
        st.plotly_chart(fig, use_container_width=True)

    #Gender vs Anemia
    elif view == "Gender vs Anemia" and has_dc:
        tmp = df.copy()
        tmp["Gender"] = tmp["Gender"].replace({0: "Male", 1: "Female", "M": "Male", "F": "Female", "m": "Male", "f": "Female"})
        tmp["Decision_Class"] = tmp["Decision_Class"].replace({0: "Non-Anemic", 1: "Anemic"})
        grouped = tmp.groupby(["Gender", "Decision_Class"]).size().reset_index(name="Count")

        if chart_type == "Bar":
            fig = px.bar(grouped, x="Gender", y="Count", color="Decision_Class", text="Count",
                         barmode="group", color_discrete_map={"Non-Anemic": "#66b3ff", "Anemic": "#ff6666"})
        elif chart_type == "Pie":
            grouped["Label"] = grouped["Gender"] + " - " + grouped["Decision_Class"]
            color_map = {"Male - Non-Anemic": "#66b3ff", "Male - Anemic": "#004c99",
                         "Female - Non-Anemic": "#ffb3c1", "Female - Anemic": "#ff6666"}
            fig = px.pie(grouped, values="Count", names="Label", color="Label", color_discrete_map=color_map)
        else:
            fig = px.box(tmp, x="Gender", y="Hb", color="Decision_Class",
                         color_discrete_map={"Non-Anemic": "#66b3ff", "Anemic": "#ff6666"})
        st.plotly_chart(fig, use_container_width=True)

    #Feature Distribution
    elif view == "Feature Distribution":
        tmp = df.copy()
        if "Decision_Class" in tmp.columns:
            tmp["Label"] = tmp["Decision_Class"].map({0: "Non-Anemic", 1: "Anemic"})
        else:
            tmp["Label"] = "Unknown"

        num_cols = tmp.select_dtypes(include=[np.number]).columns.tolist()
        if "Decision_Class" in num_cols:
            num_cols.remove("Decision_Class")

        selected = st.selectbox("Select Feature:", num_cols, index=0)
        if chart_type == "Bar":
            fig = px.histogram(tmp, x=selected, color="Label", nbins=30, opacity=0.7,
                               color_discrete_map={"Non-Anemic": "#66b3ff", "Anemic": "#ff6666"})
        else:
            fig = px.box(tmp, x="Label", y=selected, color="Label",
                         color_discrete_map={"Non-Anemic": "#66b3ff", "Anemic": "#ff6666"})
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    #Pair Plot
    if has_dc:
        st.markdown("---")
        st.subheader("🔍 Pair Plot")
        tmp = df.copy()
        tmp["Decision_Class"] = tmp["Decision_Class"].replace({0: "Non-Anemic", 1: "Anemic"})
        num_cols = [c for c in tmp.select_dtypes(include=[np.number]).columns if c != "Decision_Class"]
        if len(num_cols) >= 2:
            fig = px.scatter_matrix(tmp, dimensions=num_cols, color="Decision_Class",
                                    color_discrete_map={"Non-Anemic": "#4A90E2", "Anemic": "#E74C3C"},
                                    title="Pair Plot by Anemia Status")
            fig.update_traces(marker=dict(size=4, opacity=0.7), diagonal_visible=False)
            st.plotly_chart(fig, use_container_width=True)

        # Correlation and Feature Importance
        st.markdown("---")
        st.subheader("📊 Correlation & Feature Importance")
        c1, c2 = st.columns(2)
        df_corr = df.replace({"Non-Anemic": 0, "Anemic": 1, "Male": 0, "Female": 1})
        with c1:
            corr = df_corr.select_dtypes(include=[np.number]).corr().round(2)
            fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r")
            st.plotly_chart(fig_corr, use_container_width=True)
        with c2:
            X = df_corr.select_dtypes(include=[np.number]).drop(columns=["Decision_Class"], errors="ignore")
            y = df_corr["Decision_Class"]
            fi = ExtraTreesClassifier(random_state=42).fit(X.fillna(0), y)
            feat = pd.Series(fi.feature_importances_, index=X.columns).sort_values(ascending=False)
            fig_fi = px.bar(feat, x=feat.values, y=feat.index, orientation="h",
                            text_auto=True, color_discrete_sequence=["#ff6666"])
            st.plotly_chart(fig_fi, use_container_width=True)


elif page == "🔮 Model Evaluation and Prediction":
    st.title("Model Evaluation and Prediction")
    st.markdown("---")

    # View dropdown
    view = st.selectbox(
        "Select view:",
        ["Original Performance","Tune Performance", "Original vs Tuned Comparison"]
    )

    # Metric radio options depend on view
    if view == "Original Performance":
        metric = st.radio("Select metric:", ["Accuracy", "Precision", "Recall", "F1 Score"], horizontal=True)
        df_perf = pd.read_csv("performance_measures.csv")
    elif view == "Tune Performance":
        metric = st.radio("Select metric:", ["Accuracy", "F1 Score"], horizontal=True)
        df_perf = pd.read_csv("grid_search_results.csv")
    else:
        metric = st.radio("Select metric:", ["Accuracy", "F1 Score"], horizontal=True)
        df_perf = pd.read_csv("model_performance_comparison.csv")

    # Plot performance
    if view == "Original vs Tuned Comparison":
        fig = px.bar(df_perf, x="Model", y=metric, color="Tuning Stage", barmode="group", text_auto=True)
    else:
        fig = px.bar(df_perf, x="Model", y=metric, color="Model", text_auto=True)

    st.plotly_chart(fig, use_container_width=True)

    # Best model
    best_accuracy_comp = df_perf.loc[df_perf["Accuracy"].idxmax()]
    st.success(f"🏆 Overall the best model is: **{best_accuracy_comp['Model']}**")

    # Prediction section
    st.markdown("---")
    st.subheader("Anemia Prediction")
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        g = 1 if gender == "Female" else 0
        age = st.number_input("Age", 0, 120, 30)
        hb = st.number_input("Hemoglobin (g/dL)", 5.0, 20.0, 12.0)
        rbc = st.number_input("RBC (million/uL)", 2.0, 8.0, 4.5)
    with col2:
        pcv = st.number_input("Packed Cell Volume (PCV %)", 20.0, 60.0, 40.0)
        mcv = st.number_input("Mean Corpuscular Volume (MCV fL)", 50.0, 120.0, 85.0)
        mch = st.number_input("Mean Corpuscular Hemoglobin (MCH pg)", 10.0, 40.0, 27.0)
        mchc = st.number_input("Mean Corpuscular Hemoglobin Concentration (MCHC g/dL)", 20.0, 40.0, 33.0)

    if st.button("🔍 Predict"):
        X = pd.DataFrame([[g, age, hb, rbc, pcv, mcv, mch, mchc]],
                         columns=["Gender", "Age", "Hb", "RBC", "PCV", "MCV", "MCH", "MCHC"])
        X = X[model.feature_names_in_]
        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0][1]

        if pred == 1:
            st.error(f"🩸 Sorry, you may be **Anemic** ({prob*100:.1f}% probability)")
        else:
            st.success(f"💪 Congratulations! You are **Healthy** ({(1-prob)*100:.1f}% probability)")
