#!/usr/bin/env python
# coding: utf-8

# In[3]:


# !pip install --upgrade pip


# In[4]:


# !pip install --upgrade setuptools


# In[ ]:


from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV  # Import GridSearchCV for hyperparameter tuning
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import matplotlib.pyplot as plt  # Import matplotlib for plotting

app = Flask(__name__)

# Load the dataset
data = pd.read_csv("fish.csv")

# Preprocess the dataset
X = data.drop(columns=['Species', 'Weight'])
y = data['Weight']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Perform GridSearchCV to find the best hyperparameters
param_grid = {
    'n_estimators': [50, 100, 150, 200],  # Test different number of trees
    'max_depth': [None, 10, 20, 30],  # Test different maximum depths
    'min_samples_split': [2, 5, 10, 15],  # Test different minimum samples for split
    'min_samples_leaf': [1, 2, 4]  # Test different minimum samples for leaf node
}
grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Plot RMSE over the grid
results = pd.DataFrame(grid_search.cv_results_)
scores = -results['mean_test_score'].values.reshape(len(param_grid['n_estimators']), len(param_grid['max_depth']))
plt.figure(figsize=(12, 8))
plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot)
plt.xlabel('max_depth')
plt.ylabel('n_estimators')
plt.colorbar()
plt.xticks(ticks=range(len(param_grid['max_depth'])), labels=param_grid['max_depth'])
plt.yticks(ticks=range(len(param_grid['n_estimators'])), labels=param_grid['n_estimators'])
plt.title('Grid Search Mean Test Score (Negative MSE)')
plt.show()

# Get the best parameters and train the model with the best parameters
best_params = grid_search.best_params_
best_model = RandomForestRegressor(**best_params, random_state=42)
best_model.fit(X_train, y_train)

# Evaluate the best model
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error (Best Model):", mse)

# Save the best trained model
joblib.dump(best_model, "fish_weight_prediction_model.pkl")


# In[ ]:




