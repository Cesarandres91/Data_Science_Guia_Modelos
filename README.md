# Modelos de Machine Learning, Deep Learning e Inteligencia Artificial por Categorías de Casos de Uso

## [Predicción y Análisis de Datos](Detalle_modelos/Predicción%20y%20Análisis%20de%20Datos.md)
#### Predecir resultados futuros y analizar patrones en datos históricos, incluyendo regresiones y métodos de ensamble.
- Linear Regression
- Random Forest
- Gradient Boosting Machines (GBM)
- XGBoost
- Support Vector Machines (SVM)

## [Clasificación de Texto e Imágenes](Detalle_modelos/Clasificación%20de%20Texto%20e%20Imágenes.md)
####  Clasificar y etiquetar automáticamente textos e imágenes usando regresiones logísticas hasta redes neuronales convolucionales y transformadores.
- Logistic Regression
- Convolutional Neural Networks (CNN)
- Naive Bayes
- Transformer Models (BERT, GPT)
- Support Vector Machines (SVM)

## [Reducción de Dimensionalidad](Detalle_modelos/Reducción%20de%20Dimensionalidad.md)
####  Métodos para reducir la cantidad de variables en un dataset, preservando la mayor cantidad de información posible.
- Principal Component Analysis (PCA)
- t-Distributed Stochastic Neighbor Embedding (t-SNE)
- Linear Discriminant Analysis (LDA)
- Autoencoders
- Independent Component Analysis (ICA)

## [Clustering y Segmentación](Detalle_modelos/Clustering%20y%20Segmentación.md)
####  Agrupar datos en clusters o segmentos basados en similitudes.
- K-Means Clustering
- Hierarchical Clustering
- Gaussian Mixture Models (GMM)
- DBSCAN
- Mean Shift Clustering

## [Detección de Anomalías y Fraudes](Detalle_modelos/Detección%20de%20Anomalías%20y%20Fraudes.md)
####  Identificar patrones inusuales que pueden indicar fraudes o anomalías en los datos.
- Isolation Forest
- One-Class SVM
- Autoencoders
- Gaussian Mixture Models (GMM)
- Random Forest

## [Procesamiento de Lenguaje Natural (NLP)](Detalle_modelos/Procesamiento%20de%20Lenguaje%20Natural%20(NLP).md)
####  Modelos y técnicas para el análisis y la generación de lenguaje natural, incluyendo transformadores como BERT y GPT.
- Transformer Models (BERT, GPT)
- Recurrent Neural Networks (RNN)
- Long Short-Term Memory (LSTM)
- Naive Bayes
- Word2Vec
## [Sistemas de Recomendación](Detalle_modelos/Sistemas%20de%20Recomendación.md)
####  Algoritmos para recomendar productos o contenidos a los usuarios, desde filtrado colaborativo y basado en contenido hasta técnicas de factoración matricial y aprendizaje por refuerzo.
- Collaborative Filtering
- Content-Based Filtering
- Matrix Factorization)
- Neural Collaborative Filtering (NCF)
- Reinforcement Learning for Recommender Systems

## [Análisis de Series Temporales](Detalle_modelos/Análisis%20de%20Series%20Temporales.md)
####  Análisis y predicción de datos secuenciales a lo largo del tiempo, tales como ARIMA, LSTM y Prophet.
- ARIMA
- LSTM
- Prophet
- Recurrent Neural Networks (RNN)
- Seasonal Decomposition of Time Series (STL)

## [Optimización y Búsqueda](Detalle_modelos/Optimización%20y%20Búsqueda.md)
####  Técnicas para encontrar soluciones óptimas en problemas complejos, incluyendo algoritmos genéticos, optimización bayesiana y búsqueda en cuadrícula.
- Genetic Algorithms
- Particle Swarm Optimization
- Simulated Annealing
- Bayesian Optimization
- Grid Search

## [Visualización y Exploración de Datos](Detalle_modelos/Visualización%20y%20Exploración%20de%20Datos.md)
####  Herramientas y métodos para visualizar y explorar dataset, facilitando la comprensión de la estructura y relaciones en los datos.
- Principal Component Analysis (PCA)
- t-Distributed Stochastic Neighbor Embedding (t-SNE)
- Uniform Manifold Approximation and Projection (UMAP)
- Multidimensional Scaling (MDS)
- Self-Organizing Maps (SOM)

# Transformación de variables

| **Nombre Transformación** | **Detalle** | **Modelos Habituales** | **Modelo (S: supervisado, N: no supervisado)** |
|---------------------------|-------------|------------------------|------------------------------------------------|
| Hash Encoding | Asigna a cada categoría un hash. Útil para manejar muchas categorías únicas. | Random Forest, Redes Neuronales, Isolation Forest, One-Class SVM | Supervisado, No supervisado |
| Word2Vec | Técnica de procesamiento de lenguaje natural que crea representaciones densas de palabras. Útil en modelos de NLP y redes neuronales. | Modelos de Procesamiento de Lenguaje Natural (NLP), Redes Neuronales, Autoencoders | Supervisado, No supervisado |
| Clustering (K-Means, DBSCAN) | Métodos como K-Means y DBSCAN son usados en análisis exploratorio y detección de anomalías. | Clustering, Análisis Exploratorio, Detección de Anomalías | No supervisado |
| One-Hot Encoding | Convierte cada categoría en una nueva columna binaria (0 o 1). Utilizado en modelos que pueden manejar alta dimensionalidad. | Random Forest, Redes Neuronales, Logistic Regression, Support Vector Machines (SVM), Gradient Boosting Machines (GBM), XGBoost | Supervisado |
| Label Encoding | Asigna un número entero único a cada categoría. Útil cuando las categorías tienen un orden natural. | Árboles de Decisión, Random Forest, Gradient Boosting Machines (GBM), XGBoost | Supervisado |
| Target Encoding | Asigna a cada categoría el promedio de la variable objetivo correspondiente a esa categoría. Captura la relación con la variable objetivo. | Regresión, Árboles de Decisión, Gradient Boosting Machines (GBM), XGBoost | Supervisado |
| Frequency Encoding | Asigna a cada categoría la frecuencia con la que aparece en el conjunto de datos. Útil cuando la frecuencia es indicativa. | Random Forest, Regresión, Gradient Boosting Machines (GBM), XGBoost | Supervisado |
| Ordinal Encoding | Asigna números enteros a las categorías en un orden específico. Útil cuando las categorías tienen un orden significativo. | Árboles de Decisión, Regresión, Gradient Boosting Machines (GBM), XGBoost | Supervisado |
| Binary Encoding | Transforma las categorías en una representación binaria. Reduce la dimensionalidad comparado con One-Hot Encoding. | Random Forest, Árboles de Decisión, Gradient Boosting Machines (GBM), XGBoost | Supervisado |
| Count Encoding | Asigna a cada categoría el número de veces que aparece en el conjunto de datos. Similar a Frequency Encoding. | Random Forest, Redes Neuronales, Isolation Forest, Gradient Boosting Machines (GBM), XGBoost | Supervisado |
| Mean Encoding | Similar a Target Encoding, pero puede utilizar otras estadísticas (como la media). | Regresión, Árboles de Decisión, Gradient Boosting Machines (GBM), XGBoost | Supervisado |
| Embedding Encoding | Utiliza técnicas de aprendizaje profundo para crear representaciones densas de categorías. Común en modelos de redes neuronales. | Redes Neuronales, Modelos de Deep Learning, Recurrent Neural Networks (RNN), Autoencoders | Supervisado |


