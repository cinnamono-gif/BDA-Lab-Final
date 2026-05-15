import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix

# Set the page to be wide
st.set_page_config(layout="wide", page_title="BDA Lab Final", page_icon="🧠")

st.title("🧠 Mental Health Data & Machine Learning Dashboard")
st.write("Complete pipeline including EDA, Statistical Methods, Visualizations, and Predictive Modeling.")

# --- DATA PREPARATION ---
@st.cache_data
def load_and_prep_data():
    df = pd.read_csv("Mental_Health_Dataset.csv")
    
    # Fill Nulls
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col].fillna(df[col].mean(), inplace=True)
    for col in df.select_dtypes(include=['object']).columns:
        df[col].fillna(df[col].mode()[0], inplace=True)
        
    # Outliers (IQR on Social Media Hours)
    Q1 = df['daily_social_media_hours'].quantile(0.25)
    Q3 = df['daily_social_media_hours'].quantile(0.75)
    IQR = Q3 - Q1
    df['daily_social_media_hours'] = np.clip(df['daily_social_media_hours'], Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
    
    # Rolling Mean
    df_sorted = df.sort_values(by='age').reset_index(drop=True)
    df['rolling_mean_sleep'] = df_sorted['sleep_hours'].rolling(window=3).mean()
    
    return df

df = load_and_prep_data()

# --- CREATE TABS ---
tab1, tab2, tab3 = st.tabs(["📊 Exploratory Data Analysis", "⚙️ Machine Learning Models", "🗂️ View Dataset"])

# --- TAB 1: EDA ---
with tab1:
    st.header("Statistical Methods & Visualizations")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Statistical Summary (Mean, Std, Var)")
        stats_df = pd.DataFrame({
            'Mean': df.select_dtypes(include=[np.number]).mean(),
            'Std Dev': df.select_dtypes(include=[np.number]).std(),
            'Variance': df.select_dtypes(include=[np.number]).var()
        })
        st.dataframe(stats_df.round(2))
        
    with col2:
        st.subheader("Pie Chart: Platform Usage")
        fig1, ax1 = plt.subplots()
        df['platform_usage'].value_counts().plot.pie(autopct='%1.1f%%', cmap='Pastel1', ax=ax1)
        ax1.set_ylabel('')
        st.pyplot(fig1)

    st.markdown("---")
    
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Box Plot: Sleep by Gender")
        fig2, ax2 = plt.subplots()
        sns.boxplot(data=df, x='gender', y='sleep_hours', palette='Set2', ax=ax2)
        st.pyplot(fig2)
        
    with col4:
        st.subheader("Line Chart: Average Sleep by Age")
        fig3, ax3 = plt.subplots()
        age_sleep = df.groupby('age')['sleep_hours'].mean().reset_index()
        sns.lineplot(data=age_sleep, x='age', y='sleep_hours', marker='o', color='purple', ax=ax3)
        st.pyplot(fig3)

# --- TAB 2: MACHINE LEARNING ---
with tab2:
    st.header("Model Evaluation & Benchmarks")
    st.write("Categorical features were processed using **One-Hot Encoding**.")
    
    # Encoding
    df_encoded = pd.get_dummies(df, columns=['gender', 'platform_usage', 'social_interaction_level'], drop_first=True)
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("📈 Linear Regression")
        st.write("**Target:** `sleep_hours` (Continuous)")
        
        X_lin = df_encoded.drop(['sleep_hours', 'rolling_mean_sleep'], axis=1)
        y_lin = df_encoded['sleep_hours']
        X_train_lin, X_test_lin, y_train_lin, y_test_lin = train_test_split(X_lin, y_lin, test_size=0.2, random_state=42)
        
        lin_reg = LinearRegression()
        lin_reg.fit(X_train_lin, y_train_lin)
        y_pred_lin = lin_reg.predict(X_test_lin)
        
        st.metric(label="Mean Squared Error (MSE)", value=f"{mean_squared_error(y_test_lin, y_pred_lin):.2f}")
        st.metric(label="R-Squared", value=f"{r2_score(y_test_lin, y_pred_lin):.2f}")
        st.info("Low R-Squared indicates Linear Regression is not the ideal fit for this dataset.")

    with col_b:
        st.subheader("🤖 Logistic Regression")
        st.write("**Target:** `depression_label` (Classification)")
        
        X_log = df_encoded.drop(['depression_label', 'rolling_mean_sleep'], axis=1)
        y_log = df_encoded['depression_label']
        X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(X_log, y_log, test_size=0.2, random_state=42)
        
        log_reg = LogisticRegression(max_iter=1000)
        log_reg.fit(X_train_log, y_train_log)
        y_pred_log = log_reg.predict(X_test_log)
        
        conf_mat = confusion_matrix(y_test_log, y_pred_log)
        tn, fp, fn, tp = conf_mat.ravel()
        
        st.write("**Confusion Matrix:**")
        st.code(f"[[{tn}   {fp}]\n [  {fn}   {tp}]]")
        
        st.write("**Benchmark Extreme Rates:**")
        st.success(f"**True Positive Rate (Sensitivity):** {tp / (tp + fn):.4f}")
        st.error(f"**False Positive Rate:** {fp / (fp + tn):.4f}")
        st.success(f"**True Negative Rate (Specificity):** {tn / (tn + fp):.4f}")
        st.warning(f"**False Negative Rate:** {fn / (fn + tp):.4f}")

# --- TAB 3: DATASET ---
with tab3:
    st.header("Raw Dataset Viewer")
    st.dataframe(df)
