# Lasso-Regression
Feature Selection and Model Evaluation using Lasso Regression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('training.csv') 
df_test = pd.read_csv('docking_simulation.csv')  

# Separate features and target variable
features = ['Gauss 1', 'Gauss 2', 'Repulsion', 'Hydrophobic', 'Hydrogen', 'Torsional']  # Selected features
X = df[features]
y = df['Affinity(kcal/mol)']  # Affinity is the target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Start Lasso regression
lasso = Lasso(alpha=0.01)

# Train the Lasso model with the training data
lasso.fit(X_train, y_train)

# Check the coefficients and select non-zero ones
lasso_coef = lasso.coef_
selected_features_lasso = X_train.columns[lasso_coef != 0]
print(f"Selected Features (Lasso): {selected_features_lasso}")

# Calculate model accuracy
y_train_pred_lasso = lasso.predict(X_train)
r2_lasso = r2_score(y_train, y_train_pred_lasso)
mse_lasso = mean_squared_error(y_train, y_train_pred_lasso)
mae_lasso = mean_absolute_error(y_train, y_train_pred_lasso)

# Print the results
print(f"Lasso R² Score: {r2_lasso:.4f}")
print(f"Lasso Mean Squared Error (MSE): {mse_lasso:.4f}")
print(f"Lasso Mean Absolute Error (MAE): {mae_lasso:.4f}")

# Make predictions on the test set
y_test_pred_lasso = lasso.predict(X_test)

# Calculate model accuracy on the test set
r2_test_lasso = r2_score(y_test, y_test_pred_lasso)
mse_test_lasso = mean_squared_error(y_test, y_test_pred_lasso)
mae_test_lasso = mean_absolute_error(y_test, y_test_pred_lasso)

# Print the results
print(f"Lasso Test R² Score: {r2_test_lasso:.4f}")
print(f"Lasso Test Mean Squared Error (MSE): {mse_test_lasso:.4f}")
print(f"Lasso Test Mean Absolute Error (MAE): {mae_test_lasso:.4f}")

# If you want to find better model parameters, you can search the alpha parameter using GridSearchCV
param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
grid_search = GridSearchCV(Lasso(), param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Print the best parameter and model
print(f"Best Alpha value: {grid_search.best_params_['alpha']}")
print(f"Best score: {grid_search.best_score_}")

# Retrain with the best model and print the results
best_lasso = grid_search.best_estimator_
y_train_pred_best = best_lasso.predict(X_train)
y_test_pred_best = best_lasso.predict(X_test)

r2_train_best = r2_score(y_train, y_train_pred_best)
mse_train_best = mean_squared_error(y_train, y_train_pred_best)
mae_train_best = mean_absolute_error(y_train, y_train_pred_best)

r2_test_best = r2_score(y_test, y_test_pred_best)
mse_test_best = mean_squared_error(y_test, y_test_pred_best)
mae_test_best = mean_absolute_error(y_test, y_test_pred_best)

print(f"Best Lasso Training R²: {r2_train_best:.4f}")
print(f"Best Lasso Training MSE: {mse_train_best:.4f}")
print(f"Best Lasso Training MAE: {mae_train_best:.4f}")

print(f"Best Lasso Test R²: {r2_test_best:.4f}")
print(f"Best Lasso Test MSE: {mse_test_best:.4f}")
print(f"Best Lasso Test MAE: {mae_test_best:.4f}")

# Visualize the actual vs predicted values
plt.figure(figsize=(10,6))
plt.scatter(y_test, y_test_pred_best, color='blue', label='Predicted')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Actual Values')
plt.xlabel('Actual Affinity (kcal/mol)')
plt.ylabel('Predicted Affinity (kcal/mol)')
plt.title('Actual vs Predicted Affinity (Best Lasso)')
plt.legend()
plt.show()
