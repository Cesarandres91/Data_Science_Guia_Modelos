# Clustering y Segmentación

## K-Means Clustering
Un algoritmo de clustering que particiona los datos en K grupos, donde cada grupo está representado por el centroide de sus puntos. Es eficiente para grandes conjuntos de datos.
- **Librería**: `scikit-learn`
- **Función**: `sklearn.cluster.KMeans`

```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
labels = kmeans.predict(X)
```

## Hierarchical Clustering
Un método de clustering que construye una jerarquía de clústeres mediante la unión o división iterativa de datos. Puede ser aglomerativo (de abajo hacia arriba) o divisivo (de arriba hacia abajo).
- **Librería**: `scikit-learn`
- **Función**: `sklearn.cluster.AgglomerativeClustering`

```python
from sklearn.cluster import AgglomerativeClustering

hierarchical = AgglomerativeClustering(n_clusters=3)
labels = hierarchical.fit_predict(X)
```

## Gaussian Mixture Models (GMM)
Un modelo probabilístico que asume que los datos son generados a partir de una mezcla de varias distribuciones gaussianas. Es útil para modelar datos con estructuras complejas.
- **Librería**: `scikit-learn`
- **Función**: `sklearn.mixture.GaussianMixture`

```python
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=3)
gmm.fit(X)
labels = gmm.predict(X)
```

## DBSCAN
Un algoritmo de clustering basado en densidad que agrupa puntos que están cercanos entre sí y marca como ruido los puntos que están en áreas de baja densidad. Es robusto a la forma de los clústeres.
- **Librería**: `scikit-learn`
- **Función**: `sklearn.cluster.DBSCAN`

```python
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X)
```

## Mean Shift Clustering
Un algoritmo de clustering que busca los modos de una estimación de densidad de probabilidad mediante la actualización iterativa de los centroides hacia la media de los puntos en sus vecindades.
- **Librería**: `scikit-learn`
- **Función**: `sklearn.cluster.MeanShift`

```python
from sklearn.cluster import MeanShift

mean_shift = MeanShift()
mean_shift.fit(X)
labels = mean_shift.predict(X)
```
