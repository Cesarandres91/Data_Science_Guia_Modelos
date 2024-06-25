# Optimización y Búsqueda

## Genetic Algorithms
Un método de optimización basado en los principios de la evolución natural y la genética. Utiliza operadores como la selección, cruce y mutación para iterativamente mejorar una población de soluciones.
- **Librería**: `deap`
- **Función**: `deap.algorithms.eaSimple`

```python
from deap import base, creator, tools, algorithms
import random

# Definir el problema a resolver
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

def eval_func(individual):
    return sum(individual),

toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=100)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", eval_func)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

population = toolbox.population(n=300)
algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.2, ngen=40, verbose=False)
```

## Particle Swarm Optimization
Un algoritmo de optimización inspirado en el comportamiento social de las aves y los peces. Los individuos de la población, llamados partículas, se mueven en el espacio de soluciones buscando la mejor posición basada en su conocimiento y el de sus vecinos.
- **Librería**: `pyswarm`
- **Función**: `pyswarm.pso`

```python
from pyswarm import pso

def objective_function(x):
    return sum(x**2)

lb = [-5, -5]
ub = [5, 5]
xopt, fopt = pso(objective_function, lb, ub)
```

## Simulated Annealing
Un algoritmo de optimización inspirado en el proceso de recocido en metalurgia. Enfría iterativamente el sistema, explorando soluciones cercanas y aceptando soluciones peores con cierta probabilidad para evitar quedar atrapado en óptimos locales.
- **Librería**: `scipy`
- **Función**: `scipy.optimize.anneal` (para versiones antiguas de scipy) o `scipy.optimize.dual_annealing`

```python
from scipy.optimize import dual_annealing

def objective_function(x):
    return sum(x**2)

bounds = [(-5, 5), (-5, 5)]
result = dual_annealing(objective_function, bounds)
```

## Bayesian Optimization
Un método de optimización global que utiliza modelos probabilísticos (generalmente Gaussian Processes) para encontrar el mínimo de funciones costosas de evaluar. Es útil para la optimización de hiperparámetros en modelos de machine learning.
- **Librería**: `bayesian-optimization`
- **Función**: `bayes_opt.BayesianOptimization`

```python
from bayes_opt import BayesianOptimization

def black_box_function(x, y):
    return -x ** 2 - (y - 1) ** 2 + 1

pbounds = {'x': (-2, 2), 'y': (-3, 3)}

optimizer = BayesianOptimization(
    f=black_box_function,
    pbounds=pbounds,
    random_state=1,
)
optimizer.maximize(init_points=2, n_iter=3)
```

## Grid Search
Un método de búsqueda exhaustiva de hiperparámetros para modelos de machine learning. Prueba todas las combinaciones posibles de un conjunto especificado de valores de hiperparámetros.
- **Librería**: `scikit-learn`
- **Función**: `sklearn.model_selection.GridSearchCV`

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30]
}

model = RandomForestClassifier()
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
```
