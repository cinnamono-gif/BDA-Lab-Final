import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix

# --- PAGE SETUP ---
st.set_page_config(layout="wide", page_title="Psychological Data Analysis Portal", page_icon="🔬")

# Custom Professional Styling
st.markdown("""
    <style>
    .reportview-container { background: #fdfdfd; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #ececec; }
    .stat-box { background-color: #f1f4f9; padding: 20px; border-radius: 10px; border-left: 5px solid #2e59a8; }
    </style>
""", unsafe_allow_html=True)

# --- DATA ENGINE ---
@st.cache_data
def get_clean_data():
    # Use the path to your file
    df = pd.read_csv("Mental_Health_Dataset.csv")
    
    # REQUIREMENT: Fill Null Values
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col].fillna(df[col].mean(), inplace = True)
    for col in df.select_dtypes(include=['object']).columns:
        df[col].fillna(df[col].mode()[0], inplace = True)
        
    # REQUIREMENT: Outliers (IQR Method)
    Q1, Q3 = df['daily_social_media_hours'].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    df['daily_social_media_hours'] = np.clip(df['daily_social_media_hours'], Q1 - 1.5*IQR, Q3 + 1.5*IQR)
    
    # REQUIREMENT: Rolling Mean (Moving Average)
    # We sort by age to make the sequence logical for a rolling average
    df = df.sort_values(by='age').reset_index(drop=True)
    df['Rolling_Mean_Sleep'] = df['sleep_hours'].rolling(window=5).mean()
    
    return df

df = get_clean_data()

# --- HEADER SECTION ---
st.title("🔬 Psychological Indicators: Advanced Statistical Analysis")
st.write("An academic exploration of mental health metrics, behavioral patterns, and predictive modeling.")
st.markdown("---")

# --- TABBED INTERFACE ---
tab_stats, tab_viz, tab_ml, tab_raw = st.tabs([
    "📈 Statistical Methods", "🎨 Graphical Analysis", "🤖 Machine Learning", "📄 Data Ledger"
])

# --- TAB 1: ALL STATISTICAL METHODS ---
with tab_stats:
    st.header("Comprehensive Descriptive Statistics")
    st.info("Numerical breakdown of central tendency and dispersion metrics.")
    
    # REQUIREMENT: Mean, Std Dev, Variance, Median
    stats_summary = pd.DataFrame({
        'Mean': df.select_dtypes(include=[np.number]).mean(),
        'Median': df.select_dtypes(include=[np.number]).median(),
        'Std Deviation': df.select_dtypes(include=[np.number]).std(),
        'Variance': df.select_dtypes(include=[np.number]).var()
    }).dropna()
    
    st.table(stats_summary.round(3))
    
    st.markdown("---")
    st.subheader("Rolling Mean Calculation (Moving Average)")
    st.write("Below is the 5-point rolling average calculated for Sleep Hours across the age-sorted dataset:")
    st.dataframe(df[['age', 'sleep_hours', 'Rolling_Mean_Sleep']].dropna().head(15), use_container_width=True)

# --- TAB 2: VISUALIZATIONS ---
with tab_viz:
    st.header("Behavioral Visualization Suite")
    
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Pie Chart: Platform Distribution**")
        fig_pie, ax_pie = plt.subplots()
        df['platform_usage'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax_pie, startangle=90, cmap='viridis')
        ax_pie.set_ylabel('')
        st.pyplot(fig_pie)
        
        st.write("**Box Plot: Academic Performance vs Gender**")
        fig_box, ax_box = plt.subplots()
        sns.boxplot(data=df, x='gender', y='academic_performance', palette='coolwarm', ax=ax_box)
        st.pyplot(fig_box)

    with c2:
        st.write("**Scatter Plot: Screen Time vs Anxiety**")
        fig_scat, ax_scat = plt.subplots()
        sns.scatterplot(data=df.sample(min(len(df), 1000)), x='daily_social_media_hours', y='anxiety_level', alpha=0.4, color='teal')
        st.pyplot(fig_scat)

        st.write("**Line Chart: The Rolling Mean Trend**")
        fig_line, ax_line = plt.subplots()
        sns.lineplot(data=df, x='age', y='Rolling_Mean_Sleep', color='red', label='5-Point Rolling Avg', ax=ax_line)
        sns.scatterplot(data=df, x='age', y='sleep_hours', alpha=0.1, ax=ax_line)
        st.pyplot(fig_line)

# --- TAB 3: MACHINE LEARNING ---
with tab_ml:
    st.header("Comparative Model Benchmarking")
    
    # Preprocessing
    df_ml = pd.get_dummies(df, columns=['gender', 'platform_usage', 'social_interaction_level'], drop_first=True)
    
    # 1. LINEAR REGRESSION (Predicting Sleep)
    st.subheader("Linear Regression Analysis")
    X_lin = df_ml.drop(['sleep_hours', 'Rolling_Mean_Sleep'], axis=1)
    y_lin = df_ml['sleep_hours']
    x_train, x_test, y_train, y_test = train_test_split(X_lin, y_lin, test_size=0.2, random_state=7)
    
    l_model = LinearRegression().fit(x_train, y_train)
    y_res = l_model.predict(x_test)
    
    st.write(f"**Mean Squared Error (MSE):** `{mean_squared_error(y_test, y_res):.4f}`")
    st.write(f"**R-Squared Score:** `{r2_score(y_test, y_res):.4f}`")
    
    st.markdown("---")
    
    # 2. LOGISTIC REGRESSION (Predicting Depression)
    st.subheader("Logistic Regression (Classification Model)")
    X_log = df_ml.drop(['depression_label', 'Rolling_Mean_Sleep'], axis=1)
    y_log = df_ml['depression_label']
    x_train_c, x_test_c, y_train_c, y_test_c = train_test_split(X_log, y_log, test_size=0.2, random_state=7)
    
    c_model = LogisticRegression(max_iter=2000).fit(x_train_c, y_train_c)
    y_pred = c_model.predict(x_test_c)
    
    # Metrics
    cm = confusion_matrix(y_test_c, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    st.write("**Confusion Matrix Output:**")
    st.code(f"True Negatives: {tn} | False Positives: {fp}\nFalse Negatives: {fn} | True Positives: {tp}")
    
    st.markdown("**Extreme Rate Benchmarks:**")
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("True Pos Rate", f"{(tp/(tp+fn)):.2%}")
    mc2.metric("False Pos Rate", f"{(fp/(fp+tn)):.2%}")
    mc3.metric("True Neg Rate", f"{(tn/(tn+fp)):.2%}")
    mc4.metric("False Neg Rate", f"{(fn/(fn+tp)):.2%}")

# --- TAB 4: RAW DATA ---
with tab_raw:
    st.header("Processed Data Registry")
    st.dataframe(df, use_container_width=True)
