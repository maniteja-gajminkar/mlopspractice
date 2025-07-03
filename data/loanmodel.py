from dotenv import load_dotenv
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import pickle

# Load env vars
load_dotenv(dotenv_path="data/.env.dev")

# Get path from environment OR fallback
data_path = os.getenv("DATA_PATH", "data/loan_dev_data.csv")

print("Current Directory:", os.getcwd())
print("Data Path from .env.dev or fallback:", data_path)

# Load data
df = pd.read_csv(data_path)
print(df.head())

# Convert categorical to binary
df = pd.get_dummies(df, drop_first=True)
df = df.dropna()

# Feature and target
X = df.drop("LoanAmount", axis=1)
y = df["LoanAmount"]

# Train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("R² Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

# Save model
os.makedirs("data/model", exist_ok=True)
model_path = "data/model/loan-model.pkl"
with open(model_path, "wb") as f:
    pickle.dump(model, f)

# Test model
with open(model_path, "rb") as f:
    loaded_model = pickle.load(f)
print("Loaded model prediction:", loaded_model.predict(X_test[:1]))
