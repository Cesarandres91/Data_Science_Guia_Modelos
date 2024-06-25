# Detección de Anomalías y Fraudes

## Isolation Forest
Un algoritmo de detección de anomalías que utiliza árboles de decisión para aislar observaciones. Es eficaz para detectar puntos de datos que son pocos y diferentes.
- **Librería**: `scikit-learn`
- **Función**: `sklearn.ensemble.IsolationForest`

```python
from sklearn.ensemble import IsolationForest

model = IsolationForest(contamination=0.1)
model.fit(X_train)
anomalies = model.predict(X_test)
```

## One-Class SVM
Una variante del algoritmo SVM que se utiliza para la detección de anomalías. Entrena en un conjunto de datos "normal" y trata de identificar si nuevas observaciones son normales o anómalas.
- **Librería**: `scikit-learn`
- **Función**: `sklearn.svm.OneClassSVM`

```python
from sklearn.svm import OneClassSVM

model = OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
model.fit(X_train)
anomalies = model.predict(X_test)
```

## Autoencoders
Redes neuronales utilizadas para aprender representaciones eficientes de los datos, a menudo utilizadas para la detección de anomalías al reconstruir entradas y comparar con los datos originales.
- **Librería**: `TensorFlow`, `Keras`, `PyTorch`
- **Función**: `tf.keras.Sequential`, `torch.nn.Module`

```python
# Ejemplo con Keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

input_dim = X_train.shape[1]
encoding_dim = 14

input_layer = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu')(input_layer)
decoded = Dense(input_dim, activation='sigmoid')(encoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')
autoencoder.fit(X_train, X_train, epochs=50, batch_size=256, validation_data=(X_test, X_test))
```

## Gaussian Mixture Models (GMM)
Un modelo probabilístico que asume que los datos son generados a partir de una mezcla de varias distribuciones gaussianas. Puede ser utilizado para la detección de anomalías al identificar datos que tienen una baja probabilidad bajo el modelo.
- **Librería**: `scikit-learn`
- **Función**: `sklearn.mixture.GaussianMixture`

```python
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=2)
gmm.fit(X_train)
anomalies = gmm.predict(X_test)
```

## Random Forest
Un conjunto de árboles de decisión que puede ser utilizado para la detección de anomalías al medir la profundidad de los árboles necesarios para clasificar un punto de datos.
- **Librería**: `scikit-learn`
- **Función**: `sklearn.ensemble.RandomForestClassifier`

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```
