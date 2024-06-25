# Visualización y Exploración de Datos

## Principal Component Analysis (PCA)
Una técnica de reducción de dimensionalidad que transforma los datos a un nuevo sistema de coordenadas, de manera que la mayor varianza de los datos quede proyectada en los primeros componentes principales. Es útil para la exploración y visualización de datos en espacios de menor dimensión.
- **Librería**: `scikit-learn`
- **Función**: `sklearn.decomposition.PCA`

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

plt.scatter(X_reduced[:, 0], X_reduced[:, 1])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Visualization')
plt.show()
```

## t-Distributed Stochastic Neighbor Embedding (t-SNE)
Una técnica de reducción de dimensionalidad no lineal que es especialmente buena para la visualización de datos de alta dimensión en 2 o 3 dimensiones.
- **Librería**: `scikit-learn`
- **Función**: `sklearn.manifold.TSNE`

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

tsne = TSNE(n_components=2)
X_reduced = tsne.fit_transform(X)

plt.scatter(X_reduced[:, 0], X_reduced[:, 1])
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.title('t-SNE Visualization')
plt.show()
```

## Uniform Manifold Approximation and Projection (UMAP)
Una técnica de reducción de dimensionalidad que se utiliza para la exploración de datos de alta dimensión. Es similar a t-SNE pero generalmente más rápida y capaz de preservar más la estructura global de los datos.
- **Librería**: `umap-learn`
- **Función**: `umap.UMAP`

```python
import umap
import matplotlib.pyplot as plt

umap_model = umap.UMAP(n_components=2)
X_reduced = umap_model.fit_transform(X)

plt.scatter(X_reduced[:, 0], X_reduced[:, 1])
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.title('UMAP Visualization')
plt.show()
```

## Multidimensional Scaling (MDS)
Una técnica de reducción de dimensionalidad que intenta preservar las distancias entre los puntos en el espacio reducido. Es útil para la visualización de la similitud o disimilitud entre los datos.
- **Librería**: `scikit-learn`
- **Función**: `sklearn.manifold.MDS`

```python
from sklearn.manifold import MDS
import matplotlib.pyplot as plt

mds = MDS(n_components=2)
X_reduced = mds.fit_transform(X)

plt.scatter(X_reduced[:, 0], X_reduced[:, 1])
plt.xlabel('MDS Dimension 1')
plt.ylabel('MDS Dimension 2')
plt.title('MDS Visualization')
plt.show()
```

## Self-Organizing Maps (SOM)
Una red neuronal no supervisada utilizada para la visualización y el análisis de datos de alta dimensión. Organiza los datos en un mapa de menor dimensión, generalmente 2D, preservando las relaciones topológicas.
- **Librería**: `MiniSom`
- **Función**: `minisom.MiniSom`

```python
from minisom import MiniSom
import matplotlib.pyplot as plt

som = MiniSom(x=10, y=10, input_len=X.shape[1], sigma=1.0, learning_rate=0.5)
som.random_weights_init(X)
som.train_random(X, 100)

plt.figure(figsize=(10, 10))
for i, x in enumerate(X):
    w = som.winner(x)
    plt.plot(w[0] + 0.5, w[1] + 0.5, 'o', markerfacecolor='None', markeredgecolor='k', markersize=12, markeredgewidth=2)
plt.title('SOM Visualization')
plt.show()
```
