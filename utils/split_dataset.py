import os 
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "feature_engineered_diabetes.csv")

import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["diabetes"])
y = df["diabetes"]

# Stratified split (preserves the class balance of 'diabetes')
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,         # 20% test set
    random_state=42,       # reproducibility
    stratify=y             # stratify on target
)

print("Train class distribution:\n", y_train.value_counts(normalize=True))
print("\nTest class distribution:\n", y_test.value_counts(normalize=True))

# Save train and test_data
pd.concat([X_train, y_train], axis=1).to_csv(os.path.join(BASE_DIR, "..", "data", "train_data.csv"), index=False)
pd.concat([X_test, y_test], axis=1).to_csv(os.path.join(BASE_DIR, "..", "data", "test_data.csv"), index=False)