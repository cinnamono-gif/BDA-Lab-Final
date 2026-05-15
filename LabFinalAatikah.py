import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix

# --- PAGE CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Mental Health ML Dashboard", page_icon="✨")

# --- CUSTOM CSS FOR BEAUTIFUL UI ---
st.markdown("""
    <style>
    /* Main background and font adjustments */
    .main { background-color: #f8f9fa; }
    /* Style the metric numbers to pop */
    div[data-testid="stMetricValue"] {
        font-size: 2.2rem;
        color: #6c63ff; /* Modern Purple */
        font-weight: 800;
    }
    /* Headers */
    h1, h2, h3 { color: #2b2d42; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    /* Sidebar styling */
    [data-testid="stSidebar"] { background-color: #ffffff; box-shadow: 2px 0 5px rgba(0,0,0,0.05); }
    </style>
""", unsafe_allow_html=True)

# Set global modern theme for all charts
sns.set_theme(style="whitegrid", palette="muted")

# --- HEADER ---
st.title("✨ Mental Health Data & Machine Learning Dashboard")
st.markdown("**Complete pipeline including EDA, Statistical Methods, Visualizations, and Predictive Modeling.**")
st.markdown("---")

# --- DATA PREPARATION ---
@st.cache_data
def load_and_prep_data():
    df = pd.read_csv("Mental_Health_Dataset.csv")
    
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col].fillna(df[col].mean(), inplace=True)
    for col in df.select_dtypes(include=['object']).columns:
        df[col].fillna(df[col].mode()[0], inplace=True)
        
    Q1 = df['daily_social_media_hours'].quantile(0.25)
    Q3 = df['daily_social_media_hours'].quantile(0.75)
    IQR = Q3 - Q1
    df['daily_social_media_hours'] = np.clip(df['daily_social_media_hours'], Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
    
    df_sorted = df.sort_values(by='age').reset_index(drop=True)
    df['rolling_mean_sleep'] = df_sorted['sleep_hours'].rolling(window=3).mean()
    
    return df

df = load_and_prep_data()

# --- CREATE TABS ---
# Adding emojis to make tabs look friendly and clickable
tab1, tab2, tab3 = st.tabs(["📊 Exploratory Data Analysis", "⚙️ Machine Learning Models", "🗂️ View Dataset"])

# --- TAB 1: EDA ---
with tab1:
    st.header("Statistical Methods & Visualizations")
    
    with st.expander("📌 View Statistical Summary (Mean, Std, Var)", expanded=False):
        stats_df = pd.DataFrame({
            'Mean': df.select_dtypes(include=[np.number]).mean(),
            'Std Dev': df.select_dtypes(include=[np.number]).std(),
            'Variance': df.select_dtypes(include=[np.number]).var()
        })
        st.dataframe(stats_df.round(2).T, use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.subheader("1. Distribution & Categorical Charts")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Platform Usage**")
        fig1, ax1 = plt.subplots(figsize=(5,4))
        df['platform_usage'].value_counts().plot.pie(autopct='%1.1f%%', cmap='Set3', ax=ax1, shadow=True, wedgeprops={'edgecolor': 'white'})
        ax1.set_ylabel('')
        st.pyplot(fig1)

    with col2:
        st.markdown("**Sleep Hours Distribution**")
        fig2, ax2 = plt.subplots(figsize=(5,4))
        sns.histplot(df['sleep_hours'], bins=15, kde=True, color='#6c63ff', ax=ax2)
        ax2.set(xlabel="Sleep Hours", ylabel="Count")
        st.pyplot(fig2)
        
    with col3:
        st.markdown("**Depression by Gender**")
        fig3, ax3 = plt.subplots(figsize=(5,4))
        sns.countplot(data=df, x='gender', hue='depression_label', palette=['#4CAF50', '#FF5252'], ax=ax3)
        st.pyplot(fig3)

    st.markdown("<hr style='border:1px dashed #d3d3d3'>", unsafe_allow_html=True)
    
    st.subheader("2. Statistical Dispersion & Correlation")
    col4, col5 = st.columns([1, 1.5])
    with col4:
        st.markdown("**Sleep Hours by Gender**")
        fig4, ax4 = plt.subplots(figsize=(6,5))
        sns.boxplot(data=df, x='gender', y='sleep_hours', palette='pastel', ax=ax4)
        st.pyplot(fig4)
        
    with col5:
        st.markdown("**Feature Correlation Heatmap**")
        fig5, ax5 = plt.subplots(figsize=(8,5))
        corr_matrix = df.select_dtypes(include=[np.number]).corr()
        sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', fmt=".2f", linewidths=0.5, ax=ax5)
        st.pyplot(fig5)

# --- TAB 2: MACHINE LEARNING ---
with tab2:
    st.header("Model Evaluation & Benchmarks")
    st.info("💡 Categorical features were processed using **One-Hot Encoding** to prevent mathematical hierarchy biases.")
    
    df_encoded = pd.get_dummies(df, columns=['gender', 'platform_usage', 'social_interaction_level'], drop_first=True)
    
    col_a, col_b = st.columns(2, gap="large")
    
    with col_a:
        st.subheader("📈 Linear Regression")
        st.markdown("**Target:** `sleep_hours` (Continuous)")
        
        X_lin = df_encoded.drop(['sleep_hours', 'rolling_mean_sleep'], axis=1)
        y_lin = df_encoded['sleep_hours']
        X_train_lin, X_test_lin, y_train_lin, y_test_lin = train_test_split(X_lin, y_lin, test_size=0.2, random_state=42)
        
        lin_reg = LinearRegression()
        lin_reg.fit(X_train_lin, y_train_lin)
        y_pred_lin = lin_reg.predict(X_test_lin)
        
        # Display metrics beautifully
        st.metric(label="Mean Squared Error (MSE)", value=f"{mean_squared_error(y_test_lin, y_pred_lin):.2f}")
        st.metric(label="R-Squared", value=f"{r2_score(y_test_lin, y_pred_lin):.2f}")
        st.error("⚠️ Low R-Squared indicates Linear Regression is not the ideal fit for this dataset.")

    with col_b:
        st.subheader("🤖 Logistic Regression")
        st.markdown("**Target:** `depression_label` (Classification)")
        
        X_log = df_encoded.drop(['depression_label', 'rolling_mean_sleep'], axis=1)
        y_log = df_encoded['depression_label']
        X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(X_log, y_log, test_size=0.2, random_state=42)
        
        log_reg = LogisticRegression(max_iter=1000)
        log_reg.fit(X_train_log, y_train_log)
        y_pred_log = log_reg.predict(X_test_log)
        
        conf_mat = confusion_matrix(y_test_log, y_pred_log)
        tn, fp, fn, tp = conf_mat.ravel()
        
        st.markdown("**Benchmark Extreme Rates:**")
        metric_col1, metric_col2 = st.columns(2)
        metric_col1.metric("True Pos (Sensitivity)", f"{tp / (tp + fn):.2%}")
        metric_col2.metric("True Neg (Specificity)", f"{tn / (tn + fp):.2%}")
        metric_col1.metric("False Positive Rate", f"{fp / (fp + tn):.2%}")
        metric_col2.metric("False Negative Rate", f"{fn / (fn + tp):.2%}")

# --- TAB 3: DATASET ---
with tab3:
    st.header("Raw Dataset Viewer")
    st.dataframe(df, use_container_width=True, height=600)
