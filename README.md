# Lasso-Regression
Feature Selection and Model Evaluation using Lasso Regression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt

# Veriyi yükleyin
df = pd.read_csv('training.csv')  # Eğitim verinizin dosya adı
df_test = pd.read_csv('docking_simulation.csv')  # Test verinizin dosya adı

# Özellik ve hedef değişkenlerin ayrılması
features = ['Gauss 1', 'Gauss 2', 'Repulsion', 'Hydrophobic', 'Hydrogen', 'Torsional']  # Seçilecek özellikler
X = df[features]
y = df['Affinity(kcal/mol)']  # Affinity'yi hedef değişken olarak seçiyoruz

# Veriyi eğitim ve test olarak bölelim
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Lasso regresyonunu başlat
lasso = Lasso(alpha=0.01)

# Lasso'yu eğitim verisi ile eğit
lasso.fit(X_train, y_train)

# Katsayıları kontrol et ve sıfır olmayanları seç
lasso_coef = lasso.coef_
selected_features_lasso = X_train.columns[lasso_coef != 0]
print(f"Seçilen Özellikler (Lasso): {selected_features_lasso}")

# Modelin doğruluğunu hesapla
y_train_pred_lasso = lasso.predict(X_train)
r2_lasso = r2_score(y_train, y_train_pred_lasso)
mse_lasso = mean_squared_error(y_train, y_train_pred_lasso)
mae_lasso = mean_absolute_error(y_train, y_train_pred_lasso)

# Sonuçları yazdır
print(f"Lasso R² Skoru: {r2_lasso:.4f}")
print(f"Lasso Mean Squared Error (MSE): {mse_lasso:.4f}")
print(f"Lasso Mean Absolute Error (MAE): {mae_lasso:.4f}")

# Test seti için tahmin yap
y_test_pred_lasso = lasso.predict(X_test)

# Test setindeki model doğruluğunu hesapla
r2_test_lasso = r2_score(y_test, y_test_pred_lasso)
mse_test_lasso = mean_squared_error(y_test, y_test_pred_lasso)
mae_test_lasso = mean_absolute_error(y_test, y_test_pred_lasso)

# Sonuçları yazdır
print(f"Lasso Test R² Skoru: {r2_test_lasso:.4f}")
print(f"Lasso Test Mean Squared Error (MSE): {mse_test_lasso:.4f}")
print(f"Lasso Test Mean Absolute Error (MAE): {mae_test_lasso:.4f}")

# Eğer daha iyi bir model parametresi bulmak isterseniz GridSearchCV ile alpha parametresi üzerinde arama yapabilirsiniz
param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
grid_search = GridSearchCV(Lasso(), param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# En iyi parametreyi ve modeli yazdır
print(f"En iyi Alpha değeri: {grid_search.best_params_['alpha']}")
print(f"En iyi skor: {grid_search.best_score_}")

# En iyi modeli tekrar eğitip sonuçları yazdıralım
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

plt.figure(figsize=(10,6))
plt.scatter(y_test, y_test_pred_best, color='blue', label='Predicted')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Actual Values')
plt.xlabel('Actual Affinity (kcal/mol)')
plt.ylabel('Predicted Affinity (kcal/mol)')
plt.title('Actual vs Predicted Affinity (Best Lasso)')
plt.legend()
plt.show()
