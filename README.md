# Lasso Regression for Feature Selection and Model Evaluation

This project applies **Lasso Regression** to perform feature selection and evaluate the predictive performance of a model on molecular docking data.

## Requirements

Ensure you have the following Python libraries installed:

```bash
pip install pandas scikit-learn seaborn matplotlib
```

## Dataset

- **Training Data:** `training.csv`
- **Test Data:** `docking_simulation.csv`
- Features used: `Gauss 1`, `Gauss 2`, `Repulsion`, `Hydrophobic`, `Hydrogen`, `Torsional`
- Target variable: `Affinity (kcal/mol)`

## Implementation

### 1. Load Libraries and Data

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Load datasets
df = pd.read_csv('training.csv')
df_test = pd.read_csv('docking_simulation.csv')

# Define features and target
features = ['Gauss 1', 'Gauss 2', 'Repulsion', 'Hydrophobic', 'Hydrogen', 'Torsional']
X = df[features]
y = df['Affinity(kcal/mol)']
```

### 2. Split Data

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 3. Train Lasso Model

```python
lasso = Lasso(alpha=0.01)
lasso.fit(X_train, y_train)

# Identify selected features
selected_features = X_train.columns[lasso.coef_ != 0]
print(f"Selected Features (Lasso): {selected_features}")
```

### 4. Model Evaluation Function

```python
def evaluate_model(model, X_train, X_test, y_train, y_test):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    metrics = {
        'Train R²': r2_score(y_train, y_train_pred),
        'Train MSE': mean_squared_error(y_train, y_train_pred),
        'Train MAE': mean_absolute_error(y_train, y_train_pred),
        'Test R²': r2_score(y_test, y_test_pred),
        'Test MSE': mean_squared_error(y_test, y_test_pred),
        'Test MAE': mean_absolute_error(y_test, y_test_pred),
    }
    
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    
    return y_test, y_test_pred

# Evaluate Lasso Model
y_test, y_test_pred = evaluate_model(lasso, X_train, X_test, y_train, y_test)
```

### 5. Hyperparameter Tuning with GridSearchCV

```python
param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
grid_search = GridSearchCV(Lasso(), param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

print(f"Best Alpha: {grid_search.best_params_['alpha']}")

# Retrain with Best Model
best_lasso = grid_search.best_estimator_
y_test, y_test_pred_best = evaluate_model(best_lasso, X_train, X_test, y_train, y_test)
```

### 6. Visualization: Actual vs Predicted Affinity

```python
plt.figure(figsize=(10,6))
plt.scatter(y_test, y_test_pred_best, color='blue', label='Predicted')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Actual Values')
plt.xlabel('Actual Affinity (kcal/mol)')
plt.ylabel('Predicted Affinity (kcal/mol)')
plt.title('Actual vs Predicted Affinity (Best Lasso)')
plt.legend()
plt.show()
```

## Results

- The selected features using Lasso Regression are displayed in the output.
- The model's performance is measured in terms of R² score, Mean Squared Error (MSE), and Mean Absolute Error (MAE).
- The best alpha parameter for Lasso regression is identified using **GridSearchCV**.
- A scatter plot compares the actual vs. predicted affinity values.


