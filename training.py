#!/usr/bin/env python
# coding: utf-8

# In[3]:


# !pip install --upgrade pip


# In[4]:


# !pip install --upgrade setuptools


# In[12]:


from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, ParameterGrid
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("fish.csv")

# Preprocess the dataset
X = data.drop(columns=['Species', 'Weight'])
y = data['Weight']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4]
}

# Create parameter grid
grid = ParameterGrid(param_grid)

# Initialize an empty list to store the mean metrics
mean_metrics = []

# Initialize variables to store the best model and its parameters
best_model = None
best_params = None
best_mse = float('inf')  # Initialize with a large value

# Perform grid search and print the combination being trained along with the mean metric
for i, params in enumerate(grid):
    print(f"Run {i + 1}: Parameters - {params}")

    # Train a RandomForestRegressor model with current hyperparameters
    model = RandomForestRegressor(**params, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mean_metrics.append(mse)
    print(f"Mean Squared Error: {mse}")

    # Check if this model is the best so far
    if mse < best_mse:
        best_mse = mse
        best_model = model
        best_params = params

# Plot RMSE over the grid
plt.figure(figsize=(12, 8))
plt.plot(mean_metrics, marker='o')
plt.xlabel('Hyperparameter Combination')
plt.ylabel('Mean Squared Error')
plt.title('Grid Search Mean Squared Error')
plt.xticks(ticks=range(len(mean_metrics)), labels=range(1, len(mean_metrics) + 1))
plt.grid(True)
plt.show()

# Save the best trained model
joblib.dump(best_model, "fish_weight_prediction_model.pkl")
print("Best Model Parameters:", best_params)
print("Best Mean Squared Error:", best_mse)


