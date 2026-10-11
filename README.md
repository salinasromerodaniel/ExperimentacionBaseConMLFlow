# Experimentación y seguimiento con MLflow

**Estudiante:** Daniel Salinas Romero (A01797657)

**Repositorio:** <https://github.com/salinasromerodaniel/ExperimentacionBaseConMLFlow>

Este proyecto migra el modelo de regresión logística del notebook baseline a un script parametrizado. Usa el conjunto Wine incluido en scikit-learn, conserva la imputación por mediana y `RobustScaler` dentro del `Pipeline`, y registra cada corrida en el experimento `TC5061_Experimentacion_Wine_SalinasDaniel`.

## Instalación

Se requiere Python 3.12 o posterior por las dependencias fijadas. Entorno validado: Python 3.14.7. Ejecuta los comandos desde la raíz de este repositorio.

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

Abre <http://127.0.0.1:5000>, selecciona el experimento y entra en **Training runs**. Marca las seis configuraciones distintas y pulsa **Compare**. En la comparación, contrae **Visualizations** y activa **Show diff only** en **Parameters** para ver `C`, `penalty` y `class_weight` junto con las tres métricas. En cada corrida, **Artifacts > evaluation** contiene los archivos de evaluación y **Overview > Logged models** muestra el `Pipeline` entrenado como `wine_classifier`.

Las capturas verificadas se incluyen en `evidencias/`.

## Notebook exploratorio

`TC5061_Notebook_EDA_SalinasDaniel.ipynb` conserva la exploración del notebook anterior: descripción, distribuciones, correlaciones, faltantes, duplicados y valores extremos. No entrena modelos ni registra corridas. Ábrelo en VS Code/Jupyter con el mismo entorno de Python; si necesitas un kernel, instala las dependencias opcionales con `python -m pip install -r requirements-notebook.txt`.

Se mantiene la regresión logística, la imputación por mediana y `RobustScaler` del trabajo anterior. Para comparar L1 y L2 con el mismo optimizador se utiliza `solver="saga"`, en lugar del `lbfgs` predeterminado del notebook original, y `max_iter=5000` en lugar de 1000. La corrida denominada `baseline_C1_l2_none` es la referencia de este nuevo experimento, no una reproducción literal de todos los parámetros del notebook anterior.

## Reproducibilidad

El dataset se distribuye con scikit-learn y no requiere una descarga externa. La misma semilla controla NumPy, Python, la partición estratificada y `LogisticRegression`. Con las mismas dependencias y argumentos, dos ejecuciones producen las mismas métricas; el identificador y la hora de cada corrida sí cambian.

Después de las seis configuraciones, repite la mejor con:

```bash
python train.py --run-name reproducibilidad_C01_l2_none --C 0.1 --penalty l2 --class-weight none --max-iter 5000 --seed 42
```

En MLflow, compara esta corrida con `C01_l2_none`: ambas deben tener `accuracy=1.0`, `f1_macro=1.0` y `balanced_accuracy=1.0` en el entorno validado. Las capturas entregadas corresponden a las corridas originales; una nueva ejecución genera otros identificadores.
