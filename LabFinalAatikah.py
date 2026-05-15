import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Title of the web app
st.title("Mental Health Dataset Dashboard")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv(r"C:\Users\hp\Downloads\Mental_Health_Dataset.csv")

df = load_data()

# Show a table of the data on the front end
st.write("### Dataset Preview", df.head())

# Sidebar filter
st.sidebar.header("Filter Data")
gender_filter = st.sidebar.selectbox("Select Gender", df['gender'].unique())
filtered_df = df[df['gender'] == gender_filter]

# Display Statistics
st.write(f"### Statistics for {gender_filter}")
st.write(filtered_df.describe())

# Display a chart
st.write("### Sleep Hours Distribution")
fig, ax = plt.subplots()
sns.histplot(filtered_df['sleep_hours'], bins=20, kde=True, ax=ax, color='skyblue')
st.pyplot(fig)