# Sistemas de Recomendación

## Collaborative Filtering
Una técnica de recomendación que se basa en las preferencias de los usuarios similares. Utiliza la información de calificaciones de múltiples usuarios para predecir la calificación de un usuario para un ítem no evaluado.
- **Librería**: `surprise`
- **Función**: `surprise.prediction_algorithms.matrix_factorization.SVD`

```python
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

data = Dataset.load_builtin('ml-100k')
trainset, testset = train_test_split(data, test_size=0.25)

model = SVD()
model.fit(trainset)
predictions = model.test(testset)
accuracy.rmse(predictions)
```

## Content-Based Filtering
Un método de recomendación que utiliza las características de los ítems para hacer recomendaciones. Comparte los ítems a recomendar con los ítems previamente gustados por el usuario.
- **Librería**: `scikit-learn`
- **Función**: `sklearn.metrics.pairwise.cosine_similarity`

```python
from sklearn.metrics.pairwise import cosine_similarity

# Ejemplo de matriz de características de los ítems
item_features = X  # matriz de características de los ítems
user_profile = y  # perfil del usuario basado en ítems anteriores

similarity = cosine_similarity(user_profile, item_features)
recommendations = similarity.argsort()[0][-5:]  # obtener las 5 recomendaciones principales
```

## Matrix Factorization
Un enfoque de descomposición de la matriz de calificaciones del usuario para encontrar patrones latentes. Utiliza técnicas como SVD (Descomposición en Valores Singulares) para reducir la dimensión y predecir calificaciones.
- **Librería**: `surprise`
- **Función**: `surprise.prediction_algorithms.matrix_factorization.SVD`

```python
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

data = Dataset.load_builtin('ml-100k')
trainset, testset = train_test_split(data, test_size=0.25)

model = SVD()
model.fit(trainset)
predictions = model.test(testset)
accuracy.rmse(predictions)
```

## Neural Collaborative Filtering (NCF)
Un enfoque basado en redes neuronales para el filtrado colaborativo que aprende las interacciones entre usuarios e ítems utilizando una arquitectura de red neuronal.
- **Librería**: `TensorFlow`, `Keras`
- **Función**: `tf.keras.Model`, `tf.keras.layers.Dense`

```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate

user_input = Input(shape=(1,))
item_input = Input(shape=(1,))

user_embedding = Embedding(input_dim=num_users, output_dim=50)(user_input)
item_embedding = Embedding(input_dim=num_items, output_dim=50)(item_input)

user_vecs = Flatten()(user_embedding)
item_vecs = Flatten()(item_embedding)

input_vecs = Concatenate()([user_vecs, item_vecs])
x = Dense(128, activation='relu')(input_vecs)
x = Dense(64, activation='relu')(x)
y = Dense(1)(x)

model = Model([user_input, item_input], y)
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit([user_train, item_train], rating_train, epochs=5, batch_size=64, validation_data=([user_test, item_test], rating_test))
```

## Reinforcement Learning for Recommender Systems
Un enfoque de recomendación que utiliza técnicas de aprendizaje por refuerzo para aprender una política de recomendación óptima basada en la interacción del usuario con el sistema.
- **Librería**: `Keras-RL`, `TensorFlow`
- **Función**: `keras_rl.agents.DQNAgent`, `keras_rl.policy.BoltzmannQPolicy`

```python
from keras.models import Sequential
from keras.layers import Dense, Flatten
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

# Construir modelo
model = Sequential()
model.add(Flatten(input_shape=(1, state_size)))
model.add(Dense(24, activation='relu'))
model.add(Dense(24, activation='relu'))
model.add(Dense(action_size, activation='linear'))

# Configurar agente DQN
memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=action_size, memory=memory, nb_steps_warmup=10, target_model_update=1e-2, policy=policy)
dqn.compile(optimizer='adam', metrics=['mae'])

# Entrenar agente
dqn.fit(env, nb_steps=5000, visualize=False, verbose=2)
```
