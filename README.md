# Experimentación y seguimiento con MLflow

**Estudiante:** Daniel Salinas Romero (A01797657)

**Repositorio:** <https://github.com/salinasromerodaniel/ExperimentacionBaseConMLFlow>

Este proyecto migra el modelo de regresión logística del notebook baseline a un script parametrizado. Usa el conjunto Wine incluido en scikit-learn, conserva la imputación por mediana y `RobustScaler` dentro del `Pipeline`, y registra cada corrida en el experimento `TC5061_Experimentacion_Wine_SalinasDaniel`.

## Instalación

Se recomienda Python 3.11 o posterior.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Ejecución

El script funciona con valores por defecto:

```bash
python train.py
```

Comando parametrizado de ejemplo:

```bash
python train.py --run-name baseline_C1_l2_none --C 1 --penalty l2 --class-weight none --max-iter 5000 --seed 42
```

Los argumentos `--C`, `--penalty`, `--class-weight` y `--max-iter` controlan hiperparámetros del modelo; `--seed` fija la partición y la aleatoriedad del optimizador. Consulta todos los argumentos con `python train.py --help`.

Para ejecutar las seis configuraciones comparables:

```bash
chmod +x run_experiments.sh
PYTHON_BIN=python ./run_experiments.sh
```

## Interfaz de MLflow

Las corridas se guardan en `mlflow_tracking.db`. Inicia la interfaz desde esta carpeta:

```bash
mlflow server --backend-store-uri sqlite:///mlflow_tracking.db --host 127.0.0.1 --port 5000
```

Abre <http://127.0.0.1:5000>, selecciona el experimento y marca las corridas para compararlas. Cada corrida contiene los parámetros, `accuracy`, `f1_macro`, `balanced_accuracy`, los artefactos de evaluación y el `Pipeline` entrenado.

Las capturas verificadas se incluyen en `evidencias/`.

## Reproducibilidad

El dataset se distribuye con scikit-learn y no requiere una descarga externa. La misma semilla controla NumPy, Python, la partición estratificada y `LogisticRegression`. Con las mismas dependencias y argumentos, dos ejecuciones producen las mismas métricas; el identificador y la hora de cada corrida sí cambian.
