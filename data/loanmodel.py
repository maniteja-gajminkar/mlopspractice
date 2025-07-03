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


# In[ ]:


load_dotenv(dotenv_path="data/.env.dev")

data_path = os.getenv("DATA_PATH")
print("Current Directory:", os.getcwd())
print("Data Path from .env.dev:", data_path)

