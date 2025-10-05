# ================================
# Exoplanet Prediction Model Training
# ================================

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# -----------------------------
# Load dataset
# -----------------------------
koi = pd.read_csv('KOI_CSV.csv')

print("\n* Columns in Dataset:\n", koi.columns.tolist())
print("\n* First 5 rows:\n", koi.head())
print("\n* Shape before preprocessing:", koi.shape)

# -----------------------------
# Inspect missing values
# -----------------------------
print("\n* Missing values per column:\n", koi.isnull().sum())

# -----------------------------
# Filter relevant labels only
# -----------------------------
koi = koi[koi['koi_disposition'].isin(['CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE'])]
print("\n* Counts of each label:\n", koi['koi_disposition'].value_counts())

# -----------------------------
# Drop unnecessary columns
# -----------------------------
drop_cols = ['kepid', 'kepoi_name', 'koi_pdisposition', 'kepler_name', 
             'koi_tce_plnt_num', 'koi_tce_delivname', 'ra', 'dec', 'koi_score']
koi = koi.drop(columns=drop_cols)

# Drop rows with missing values in selected features
koi = koi.dropna()
print("\n* Shape after dropping unnecessary columns and missing values:", koi.shape)

# -----------------------------
# Encode labels
# -----------------------------
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
koi['koi_disposition'] = le.fit_transform(koi['koi_disposition'])
print("\n* Label encoding mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

# -----------------------------
# Define features and target
# -----------------------------
features = ['koi_fpflag_ec', 'koi_model_snr', 'koi_prad', 
            'koi_impact', 'koi_fpflag_ss', 'koi_fpflag_co',
            'koi_duration', 'koi_period', 'koi_depth']

X = koi[features]
y = koi['koi_disposition']

# -----------------------------
# Split train/test
# -----------------------------
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\n* Training set shape:", X_train.shape, y_train.shape)
print("* Testing set shape:", X_test.shape, y_test.shape)

# -----------------------------
# Train Random Forest (balanced)
# -----------------------------
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# -----------------------------
# Predictions & Accuracy
# -----------------------------
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("\n* Accuracy: {:.2f}%".format(acc*100))
print("\n* Classification Report:\n", classification_report(y_test, y_pred))
print("\n* Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# -----------------------------
# Save trained model
# -----------------------------
joblib.dump(model, 'siftingexosaccurate_model.pkl')
print("\n* Model saved as 'siftingexosaccurate_model.pkl'")

# -----------------------------
# Feature Importance
# -----------------------------
importances = model.feature_importances_ * 100  # convert to %
importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=True)

# Plot horizontal bar chart
plt.figure(figsize=(10,6))
bars = plt.barh(importance_df['Feature'], importance_df['Importance'], color='purple')
plt.xlabel('Importance (%)')
plt.title('Feature Importance for Exoplanet Prediction Model')

# Add percentage labels to bars
for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.5, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', va='center')

plt.tight_layout()
plt.show()

# Print top features
print("\n* Feature Importance Table:\n", importance_df)
# ================================