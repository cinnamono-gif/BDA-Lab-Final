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
    
    st.subheader("Statistical Summary (Mean, Std, Var)")
    stats_df = pd.DataFrame({
        'Mean': df.select_dtypes(include=[np.number]).mean(),
        'Std Dev': df.select_dtypes(include=[np.number]).std(),
        'Variance': df.select_dtypes(include=[np.number]).var()
    })
    st.dataframe(stats_df.round(2).T) # Transposed for a wider, cleaner view
    
    st.markdown("---")
    st.subheader("1. Distribution & Categorical Charts")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("**Pie Chart: Platform Usage**")
        fig1, ax1 = plt.subplots(figsize=(5,4))
        df['platform_usage'].value_counts().plot.pie(autopct='%1.1f%%', cmap='Pastel1', ax=ax1)
        ax1.set_ylabel('')
        st.pyplot(fig1)

    with col2:
        st.write("**Histogram: Sleep Hours Distribution**")
        fig2, ax2 = plt.subplots(figsize=(5,4))
        sns.histplot(df['sleep_hours'], bins=15, kde=True, color='skyblue', ax=ax2)
        st.pyplot(fig2)
        
    with col3:
        st.write("**Count Plot: Depression by Gender**")
        fig3, ax3 = plt.subplots(figsize=(5,4))
        sns.countplot(data=df, x='gender', hue='depression_label', palette='Set2', ax=ax3)
        st.pyplot(fig3)

    st.markdown("---")
    st.subheader("2. Relationship & Trend Charts")
    col4, col5 = st.columns(2)
    with col4:
        st.write("**Scatter Plot: Social Media Hours vs Stress Level**")
        fig4, ax4 = plt.subplots(figsize=(6,4))
        # Using a sample of 2000 so the scatter plot isn't too cluttered to read
        sns.scatterplot(data=df.sample(2000, random_state=42), x='daily_social_media_hours', y='stress_level', alpha=0.5, color='coral', ax=ax4)
        st.pyplot(fig4)
        
    with col5:
        st.write("**Line Chart: Average Sleep by Age**")
        fig5, ax5 = plt.subplots(figsize=(6,4))
        age_sleep = df.groupby('age')['sleep_hours'].mean().reset_index()
        sns.lineplot(data=age_sleep, x='age', y='sleep_hours', marker='o', color='purple', ax=ax5)
        st.pyplot(fig5)

    st.markdown("---")
    st.subheader("3. Statistical Dispersion (Box & Violin Plots)")
    col6, col7 = st.columns(2)
    with col6:
        st.write("**Box Plot: Sleep Hours by Gender**")
        fig6, ax6 = plt.subplots(figsize=(6,4))
        sns.boxplot(data=df, x='gender', y='sleep_hours', palette='Set3', ax=ax6)
        st.pyplot(fig6)
        
    with col7:
        st.write("**Violin Plot: Stress Level by Social Interaction**")
        fig7, ax7 = plt.subplots(figsize=(6,4))
        sns.violinplot(data=df, x='social_interaction_level', y='stress_level', palette='magma', ax=ax7)
        st.pyplot(fig7)
        
    st.markdown("---")
    st.subheader("4. Correlation Analysis")
    st.write("**Heatmap: Numeric Feature Correlations**")
    fig8, ax8 = plt.subplots(figsize=(10,6))
    corr_matrix = df.select_dtypes(include=[np.number]).corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax8)
    st.pyplot(fig8)

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
