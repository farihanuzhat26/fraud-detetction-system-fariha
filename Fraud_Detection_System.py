# === 1. IMPORTS ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from imblearn.over_sampling import SMOTE
import joblib

# === 2. LOAD DATA ===
import kagglehub

path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
csv_file = os.path.join(path, "creditcard.csv")
df = pd.read_csv(csv_file)

# === 3. EDA ===
print(df.head())
print(df.info())
print(df['Class'].value_counts())  # Class = 1 is fraud

# Optional: Visualize class imbalance
sns.countplot(x='Class', data=df)
plt.title("Class Distribution")
plt.show()

# === 4. DATA PREPROCESSING ===

# Separate features and labels
X = df.drop(['Class', 'Time'], axis=1)
y = df['Class']

# Scale 'Amount' (already PCA on other columns)
scaler = StandardScaler()
X['Amount'] = scaler.fit_transform(X['Amount'].values.reshape(-1, 1))

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Handle imbalance with SMOTE
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_train, y_train)

print(f"After SMOTE: {np.bincount(y_resampled)}")

# === 5. TRAIN MODELS ===

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_resampled, y_resampled)
lr_pred = lr.predict(X_test)
print("=== Logistic Regression ===")
print(classification_report(y_test, lr_pred))
print("ROC AUC:", roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1]))

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_resampled, y_resampled)
rf_pred = rf.predict(X_test)
print("=== Random Forest ===")
print(classification_report(y_test, rf_pred))
print("ROC AUC:", roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1]))

# === 6. EXPORT BEST MODEL ===
joblib.dump(rf, 'fraud_model.pkl')
