from dotenv import load_dotenv
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import pickle  # ✅ Move this to the top

# Load env vars
load_dotenv(dotenv_path="/loandemo/data/.env.dev")

# Get path from environment OR use fallback
data_path = os.getenv("DATA_PATH")
if not data_path:
    data_path = "./loandemo/data/loan_dev_data.csv"

print("Current Directory:", os.getcwd())
print("Data Path from .env.dev:", data_path)

# Load data
df = pd.read_csv(data_path)
print(df.head())

# Convert categorical 'Employed' to binary (1 = Yes)
df = pd.get_dummies(df, drop_first=True)

# Drop any rows with missing values (if any)
df = df.dropna()

# Feature-target split
X = df.drop("LoanAmount", axis=1)
y = df["LoanAmount"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("R² Score:", r2_score(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))

# Ensure model output directory exists
os.makedirs("data/model", exist_ok=True)

# Save the model
model_path = "data/model/loan-moodel.pkl"
with open(model_path, "wb") as f:
    pickle.dump(model, f)

# Load and test the model (optional check)
with open(model_path, "rb") as f:
    loaded_model = pickle.load(f)
print("Loaded model prediction:", loaded_model.predict(X_test[:1]))
