#!/usr/bin/env python
# coding: utf-8

from dotenv import load_dotenv
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import pickle

# Load environment variables
load_dotenv(dotenv_path="data/.env.dev")  # Adjusted for GitHub Actions context

# Try loading data path from env file
data_path = os.getenv("DATA_PATH")

# Fallback if env variable not found
if not data_path:
    data_path = "data/loan_dev_data.csv"

# Show current path context
print("Current Directory:", os.getcwd())
print("Data Path from .env.dev or fallback:", data_path)

# Load CSV file
df = pd.read_csv(data_path)
print("📊 Sample data:")
print(df.head())

# Encode categorical variables
df = pd.get_dummies(df, drop_first=True)

# Drop missing values
df = df.dropna()

# Split features and target
X = df.drop("LoanAmount", axis=1)
y = df["LoanAmount"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)

if len(X_test) < 2:
    print("⚠️ Not enough data to compute R² score reliably.")
else:
    print("R² Score:", r2_score(y_test, y_pred))

print("MSE:", mean_squared_error(y_test, y_pred))

# ✅ Ensure model output directory exists
os.makedirs("data/model", exist_ok=True)

# Save the model
model_path = "data/model/loanmodel.pkl"
with open(model_path, "wb") as f:
    pickle.dump(model, f)

# Optional check to confirm model saved properly
with open(model_path, "rb") as f:
    loaded_model = pickle.load(f)

print("✅ Loaded model prediction on one sample:", loaded_model.predict(X_test[:1]))
print(f"\n✅ Model saved to {model_path}")
print("✅ Model training and saving complete.")