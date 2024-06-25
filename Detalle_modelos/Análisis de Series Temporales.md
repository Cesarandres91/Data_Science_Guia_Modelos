# Análisis de Series Temporales

## ARIMA
Un modelo estadístico utilizado para el análisis y la predicción de series temporales. Combina componentes de autorregresión (AR), media móvil (MA) e integración (I) para modelar la autocorrelación en los datos.
- **Librería**: `statsmodels`
- **Función**: `statsmodels.tsa.arima.model.ARIMA`

```python
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# Ejemplo de serie temporal
data = pd.read_csv('time_series.csv', index_col='date', parse_dates=True)
series = data['value']

model = ARIMA(series, order=(5, 1, 0))
model_fit = model.fit()
forecast = model_fit.forecast(steps=10)
```

## LSTM
Una mejora de las RNN estándar que puede aprender dependencias a largo plazo. Son especialmente efectivas para tareas de predicción de series temporales.
- **Librería**: `TensorFlow`, `Keras`, `PyTorch`
- **Función**: `tf.keras.layers.LSTM`, `torch.nn.LSTM`

```python
# Ejemplo con Keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(timesteps, input_dim)),
    LSTM(50),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))
```

## Prophet
Una herramienta de modelado de series temporales desarrollada por Facebook que es robusta frente a datos faltantes y cambios en la tendencia, y que maneja automáticamente los efectos estacionales.
- **Librería**: `prophet`
- **Función**: `prophet.Prophet`

```python
from prophet import Prophet
import pandas as pd

# Ejemplo de serie temporal
data = pd.read_csv('time_series.csv')
data.columns = ['ds', 'y']  # Prophet requiere columnas 'ds' (fecha) y 'y' (valor)

model = Prophet()
model.fit(data)
future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)
```

## Recurrent Neural Networks (RNN)
Un tipo de red neuronal diseñada para reconocer patrones en secuencias de datos. Es útil para tareas como el análisis de texto y series temporales.
- **Librería**: `TensorFlow`, `Keras`, `PyTorch`
- **Función**: `tf.keras.layers.SimpleRNN`, `torch.nn.RNN`

```python
# Ejemplo con Keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

model = Sequential([
    SimpleRNN(50, input_shape=(timesteps, input_dim)),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))
```

## Seasonal Decomposition of Time Series (STL)
Un método para descomponer una serie temporal en componentes de tendencia, estacionalidad y residuales. Es útil para analizar los patrones subyacentes en los datos de series temporales.
- **Librería**: `statsmodels`
- **Función**: `statsmodels.tsa.seasonal.STL`

```python
import pandas as pd
from statsmodels.tsa.seasonal import STL

# Ejemplo de serie temporal
data = pd.read_csv('time_series.csv', index_col='date', parse_dates=True)
series = data['value']

stl = STL(series, seasonal=13)
result = stl.fit()
trend = result.trend
seasonal = result.seasonal
resid = result.resid
```
