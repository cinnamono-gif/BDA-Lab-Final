import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, classification_report

# 1. Load Data
df = pd.read_csv(r"C:\Users\hp\Downloads\Mental_Health_Dataset.csv")

# 2. Fill missing values (Mean for numbers, Mode for text)
for col in df.select_dtypes(include=[np.number]).columns:
    df[col].fillna(df[col].mean(), inplace=True)
for col in df.select_dtypes(include=['object']).columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

# 3. Stats: Variance, Std Dev, and Rolling Mean
print(df.select_dtypes(include=[np.number]).var()) # Variance
df_sorted = df.sort_values(by='age').reset_index(drop=True)
df_sorted['rolling_mean_sleep'] = df_sorted['sleep_hours'].rolling(window=3).mean()

# 4. Outliers (Clipping using IQR)
Q1 = df['daily_social_media_hours'].quantile(0.25)
Q3 = df['daily_social_media_hours'].quantile(0.75)
IQR = Q3 - Q1
df['daily_social_media_hours'] = np.clip(df['daily_social_media_hours'], Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

# 5. One Hot Encoding
df_encoded = pd.get_dummies(df, columns=['gender', 'platform_usage', 'social_interaction_level'], drop_first=True)

# 6. Model: Logistic Regression (Classification)
X = df_encoded.drop('depression_label', axis=1)
y = df_encoded['depression_label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)

# 7. Benchmarks & Matrix
conf_mat = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_mat)

tn, fp, fn, tp = conf_mat.ravel()
print(f"True Positive Rate: {tp / (tp + fn):.4f}")
print(f"False Positive Rate: {fp / (fp + tn):.4f}")