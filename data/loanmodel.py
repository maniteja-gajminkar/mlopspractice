#!/usr/bin/env python
# coding: utf-8

# In[1]:


from dotenv import load_dotenv
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt


# In[5]:


load_dotenv(dotenv_path=".env.dev")  

print("Current Directory:", os.getcwd())
print("Data Path from .env.dev:", data_path)
data_path = "loan_dev_data.csv"



# In[6]:


df = pd.read_csv(data_path)
print(df.head())


# In[ ]:


# Convert categorical 'Employed' to binary (1 = Yes)
df = pd.get_dummies(df, drop_first=True)

# Drop any rows with missing values (if any)
df = df.dropna()


# In[ ]:


X = df.drop("LoanAmount", axis=1)
y = df["LoanAmount"]


# In[ ]:


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("R² Score:", r2_score(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))





# In[ ]:


import pickle


# In[ ]:


# Save the model
with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)

# Load the model
with open("model/model.pkl", "rb") as f:
    loaded_model = pickle.load(f)

# Test loaded model
print("Loaded model prediction:", loaded_model.predict(X_test))

