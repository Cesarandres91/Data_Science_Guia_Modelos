# Reducción de Dimensionalidad

## Principal Component Analysis (PCA)
Una técnica de reducción de dimensionalidad que transforma los datos a un nuevo sistema de coordenadas, de manera que la mayor varianza de los datos quede proyectada en los primeros componentes principales.
- **Librería**: `scikit-learn`
- **Función**: `sklearn.decomposition.PCA`

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)
```

## t-Distributed Stochastic Neighbor Embedding (t-SNE)
Una técnica de reducción de dimensionalidad no lineal que es especialmente buena para la visualización de datos de alta dimensión en 2 o 3 dimensiones.
- **Librería**: `scikit-learn`
- **Función**: `sklearn.manifold.TSNE`

```python
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2)
X_reduced = tsne.fit_transform(X)
```

## Linear Discriminant Analysis (LDA)
Una técnica supervisada de reducción de dimensionalidad que busca proyectar los datos en un espacio de menor dimensión tal que las clases estén lo más separadas posible.
- **Librería**: `scikit-learn`
- **Función**: `sklearn.discriminant_analysis.LinearDiscriminantAnalysis`

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis(n_components=2)
X_reduced = lda.fit_transform(X, y)
```

## Autoencoders
Redes neuronales que aprenden a comprimir datos en una representación de menor dimensión y luego reconstruir los datos originales a partir de esta representación.
- **Librería**: `TensorFlow`, `Keras`, `PyTorch`
- **Función**: `tf.keras.Sequential`, `torch.nn.Module`

```python
# Ejemplo con Keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

input_dim = X.shape[1]
encoding_dim = 32

input_layer = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu')(input_layer)
decoded = Dense(input_dim, activation='sigmoid')(encoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(X_train, X_train, epochs=50, batch_size=256, validation_data=(X_test, X_test))
```

## Independent Component Analysis (ICA)
Una técnica de análisis de datos que busca encontrar componentes independientes no gaussianas en los datos.
- **Librería**: `scikit-learn`
- **Función**: `sklearn.decomposition.FastICA`

```python
from sklearn.decomposition import FastICA

ica = FastICA(n_components=2)
X_reduced = ica.fit_transform(X)
```
