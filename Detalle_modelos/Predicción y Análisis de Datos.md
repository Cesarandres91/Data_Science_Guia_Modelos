# Predicción y Análisis de Datos

## Linear Regression
Un modelo estadístico que se utiliza para predecir el valor de una variable dependiente basada en el valor de una o más variables independientes. Asume una relación lineal entre las variables.
- **Librería**: `scikit-learn`
- **Función**: `sklearn.linear_model.LinearRegression`

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

## Random Forest
Un conjunto de árboles de decisión que se entrenan utilizando diferentes subconjuntos del conjunto de datos y características. Se utiliza para clasificación y regresión, y es conocido por su alta precisión y capacidad para manejar datos faltantes y detectar interacciones entre variables.
- **Librería**: `scikit-learn`
- **Función**: `sklearn.ensemble.RandomForestRegressor` (para regresión), `sklearn.ensemble.RandomForestClassifier` (para clasificación)

```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

## Gradient Boosting Machines (GBM)
Una técnica de ensemble que construye modelos aditivos de manera secuencial. Cada modelo intenta corregir los errores de su predecesor. Es altamente efectivo para tareas de predicción y es muy utilizado en competencias de machine learning.
- **Librería**: `scikit-learn`
- **Función**: `sklearn.ensemble.GradientBoostingRegressor` (para regresión), `sklearn.ensemble.GradientBoostingClassifier` (para clasificación)

```python
from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

## XGBoost
Una implementación optimizada de gradient boosting que es altamente eficiente, flexible y portátil. Incluye mejoras de regularización, manejo de datos faltantes y procesamiento paralelo, lo que lo hace extremadamente rápido y preciso.
- **Librería**: `xgboost`
- **Función**: `xgboost.XGBRegressor` (para regresión), `xgboost.XGBClassifier` (para clasificación)

```python
import xgboost as xgb

model = xgb.XGBRegressor()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

## Support Vector Machines (SVM)
Un algoritmo de clasificación que encuentra el hiperplano óptimo que maximiza el margen entre las clases. Puede ser utilizado para problemas de clasificación lineal y no lineal y es eficaz en espacios de alta dimensión.
- **Librería**: `scikit-learn`
- **Función**: `sklearn.svm.SVR` (para regresión), `sklearn.svm.SVC` (para clasificación)

```python
from sklearn.svm import SVR

model = SVR()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```
