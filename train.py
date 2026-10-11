"""Entrenamiento reproducible del clasificador Wine con seguimiento en MLflow."""

from __future__ import annotations

import argparse
import json
import random
import tempfile
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import load_wine
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_EXPERIMENT = "TC5061_Experimentacion_Wine_SalinasDaniel"


def parse_args() -> argparse.Namespace:
    """Define los argumentos de la corrida y valida sus valores básicos."""
    parser = argparse.ArgumentParser(
        description="Entrena una regresión logística sobre UCI Wine y registra la corrida en MLflow."
    )
    parser.add_argument("--C", type=float, default=1.0, help="Inverso de regularización (> 0).")
    parser.add_argument(
        "--penalty",
        choices=("l1", "l2"),
        default="l2",
        help="Tipo de regularización del clasificador.",
    )
    parser.add_argument(
        "--class-weight",
        choices=("none", "balanced"),
        default="none",
        help="Ponderación de clases; 'balanced' usa frecuencias del entrenamiento.",
    )
    parser.add_argument(
        "--max-iter", type=int, default=5000, help="Máximo de iteraciones del optimizador (> 0)."
    )
    parser.add_argument("--seed", type=int, default=42, help="Semilla de partición y entrenamiento.")
    parser.add_argument(
        "--test-size", type=float, default=0.20, help="Proporción estratificada de prueba (0, 1)."
    )
    parser.add_argument("--run-name", default=None, help="Nombre opcional de la corrida en MLflow.")
    parser.add_argument(
        "--experiment-name", default=DEFAULT_EXPERIMENT, help="Experimento que agrupa las corridas."
    )
    parser.add_argument(
        "--tracking-uri",
        default=f"sqlite:///{(PROJECT_DIR / 'mlflow_tracking.db').as_posix()}",
        help="URI de MLflow Tracking.",
    )
    args = parser.parse_args()
    if args.C <= 0:
        parser.error("--C debe ser mayor que 0")
    if args.max_iter <= 0:
        parser.error("--max-iter debe ser mayor que 0")
    if not 0 < args.test_size < 1:
        parser.error("--test-size debe estar entre 0 y 1")
    return args


def fix_seed(seed: int) -> None:
    """Fija las fuentes de aleatoriedad usadas por el script."""
    random.seed(seed)
    np.random.seed(seed)


def load_and_clean_data() -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    """Carga Wine y aplica las mismas reglas de limpieza del notebook baseline."""
    wine = load_wine(as_frame=True)
    data = wine.frame.copy()
    features = list(wine.feature_names)
    for column in features:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["target"] = pd.to_numeric(data["target"], errors="coerce")
    data = data.dropna(subset=["target"]).drop_duplicates().copy()
    data["target"] = data["target"].astype(int)
    return data[features], data["target"], features, list(wine.target_names)


def split_data(
    X: pd.DataFrame, y: pd.Series, test_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Crea una partición estratificada y reproducible."""
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


def build_model(args: argparse.Namespace) -> Pipeline:
    """Construye el flujo de preprocesamiento y clasificación."""
    class_weight = None if args.class_weight == "none" else args.class_weight
    return Pipeline(
        steps=[
            ("imputation", SimpleImputer(strategy="median")),
            ("scaling", RobustScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=args.C,
                    penalty=args.penalty,
                    solver="saga",
                    class_weight=class_weight,
                    max_iter=args.max_iter,
                    random_state=args.seed,
                    tol=1e-4,
                ),
            ),
        ]
    )


def evaluate_model(
    model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, target_names: list[str]
) -> tuple[dict[str, float], dict[str, Any], Any]:
    """Calcula métricas y crea la matriz de confusión."""
    predictions = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1_macro": float(f1_score(y_test, predictions, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
    }
    report = classification_report(
        y_test, predictions, target_names=target_names, output_dict=True, zero_division=0
    )
    figure, axis = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, predictions, display_labels=target_names, cmap="Blues", ax=axis
    )
    axis.set_title("Matriz de confusión")
    figure.tight_layout()
    return metrics, report, figure


def log_run(
    args: argparse.Namespace,
    model: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    metrics: dict[str, float],
    report: dict[str, Any],
    figure: Any,
) -> str:
    """Registra parámetros, métricas, artefactos y modelo en MLflow."""
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run(run_name=args.run_name) as run:
        mlflow.log_params(
            {
                "dataset": "UCI Wine / sklearn.load_wine",
                "model": "LogisticRegression",
                "C": args.C,
                "penalty": args.penalty,
                "solver": "saga",
                "class_weight": args.class_weight,
                "max_iter": args.max_iter,
                "tol": 1e-4,
                "seed": args.seed,
                "test_size": args.test_size,
                "train_rows": len(X_train),
                "test_rows": len(X_test),
                "features": X_train.shape[1],
                "sklearn_version": sklearn.__version__,
                "mlflow_version": mlflow.__version__,
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.set_tags(
            {
                "student": "Daniel Salinas Romero",
                "student_id": "A01797657",
                "data_source": "https://archive.ics.uci.edu/dataset/109/wine",
            }
        )
        with tempfile.TemporaryDirectory(prefix="tc5061_artifacts_") as temp_dir:
            artifact_dir = Path(temp_dir)
            metrics_path = artifact_dir / "metrics.json"
            report_path = artifact_dir / "classification_report.json"
            figure_path = artifact_dir / "confusion_matrix.png"
            metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            figure.savefig(figure_path, dpi=160, bbox_inches="tight")
            mlflow.log_artifacts(str(artifact_dir), artifact_path="evaluation")
        mlflow.sklearn.log_model(
            sk_model=model,
            name="wine_classifier",
            input_example=X_test.head(3),
            serialization_format="cloudpickle",
        )
        return run.info.run_id


def main() -> None:
    args = parse_args()
    fix_seed(args.seed)
    X, y, _, target_names = load_and_clean_data()
    X_train, X_test, y_train, y_test = split_data(X, y, args.test_size, args.seed)
    model = build_model(args)
    model.fit(X_train, y_train)
    metrics, report, figure = evaluate_model(model, X_test, y_test, target_names)
    try:
        run_id = log_run(args, model, X_train, X_test, metrics, report, figure)
    finally:
        plt.close(figure)
    print(
        json.dumps(
            {
                "run_id": run_id,
                "experiment": args.experiment_name,
                "tracking_uri": args.tracking_uri,
                "parameters": {
                    "C": args.C,
                    "penalty": args.penalty,
                    "class_weight": args.class_weight,
                    "max_iter": args.max_iter,
                    "seed": args.seed,
                },
                "metrics": metrics,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
