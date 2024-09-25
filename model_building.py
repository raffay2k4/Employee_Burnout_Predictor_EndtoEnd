# Import Libraries

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import numpy as np
from model_io import save_model,load_model

# Load Data

df=pd.read_csv("Data\Cleaned_ML_Dataset.csv")
df.head()

# Data Splitting

#Split into X and y
X=df.drop(columns=["Burn Rate"])
y=df["Burn Rate"]

#Train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=7)

# Model Building

#Initializing different models
models = {
    'Linear Regression' : LinearRegression(),
    'Random Forest': RandomForestRegressor(),
    'Decision Tree' : DecisionTreeRegressor()
}

#Checking which model gives the best performance
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)  
    r2 = r2_score(y_test, y_pred)
    
    print(f"{name} Performance:")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Root Mean Squared Error: {rmse:.2f}")
    print(f"R-squared: {r2:.2f}")
    print("\n")

# Selecting Random Forest: Hyperparameter Tuning

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

#Performing GridSearch
grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=5, scoring='neg_mean_absolute_error')
grid_search.fit(X_train, y_train)

#Best parameters and score
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score (MAE):", -grid_search.best_score_)  

# Final Eval of Best Param Random Forest

best_params_rf = grid_search.best_estimator_
best_params_rf.fit(X_train, y_train)

#Make predictions
y_pred = best_params_rf.predict(X_test)

#Calculating evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)  
r2 = r2_score(y_test, y_pred)

print("Final Random Forest Model Performance:")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R-squared: {r2:.2f}")

#Save Model

save_model(best_params_rf)
