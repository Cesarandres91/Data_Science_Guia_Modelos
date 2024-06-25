# Clasificación de Texto e Imágenes

## Logistic Regression
Un modelo estadístico utilizado para tareas de clasificación binaria y multiclase, que utiliza una función logística para modelar la probabilidad de pertenencia a una clase.
- **Librería**: `scikit-learn`
- **Función**: `sklearn.linear_model.LogisticRegression`

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

## Convolutional Neural Networks (CNN)
Una arquitectura de red neuronal profunda diseñada para procesar datos con una estructura de cuadrícula, como las imágenes. Las CNN son especialmente efectivas para tareas de reconocimiento y clasificación de imágenes.
- **Librería**: `TensorFlow`, `Keras`, `PyTorch`
- **Función**: `tf.keras.Sequential`, `torch.nn.Conv2d`

```python
# Ejemplo con Keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
```

## Naive Bayes
Un conjunto de algoritmos de clasificación basados en el teorema de Bayes, con la suposición de independencia entre las características. Es muy utilizado para tareas de clasificación de texto.
- **Librería**: `scikit-learn`
- **Función**: `sklearn.naive_bayes.GaussianNB`, `sklearn.naive_bayes.MultinomialNB`

```python
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

## Transformer Models (BERT, GPT)
Modelos de aprendizaje profundo basados en la arquitectura Transformer, que son altamente efectivos para tareas de procesamiento de lenguaje natural (NLP) como la clasificación de texto, generación de texto y traducción automática.
- **Librería**: `transformers`
- **Función**: `transformers.BertForSequenceClassification`, `transformers.GPT2LMHeadModel`

```python
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

inputs = tokenizer(X_train, return_tensors='pt', padding=True, truncation=True)
labels = torch.tensor(y_train)

training_args = TrainingArguments(output_dir='./results', num_train_epochs=3)
trainer = Trainer(model=model, args=training_args, train_dataset=inputs, eval_dataset=inputs)
trainer.train()
```

## Support Vector Machines (SVM)
Un algoritmo de clasificación que encuentra el hiperplano óptimo que maximiza el margen entre las clases. Es eficaz en espacios de alta dimensión y puede ser utilizado tanto para problemas de clasificación lineal como no lineal.
- **Librería**: `scikit-learn`
- **Función**: `sklearn.svm.SVC`

```python
from sklearn.svm import SVC

model = SVC()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```
