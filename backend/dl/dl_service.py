"""Deep Learning module skeleton.

Architecture mirrors the ML pipeline:
    Data → Preprocessing → Model → Training → Evaluation → Prediction

This module is preserved as a structured placeholder for existing DL functionality.
The DL architecture remains intact and is NOT replaced by the ML/SVM implementation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class DLDataConfig:
    """Configuration for DL data loading."""
    dataset_name: str = "Iris"
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42
    batch_size: int = 32


@dataclass
class DLPreprocessingConfig:
    """Configuration for DL preprocessing."""
    normalize: bool = True
    one_hot_encode_labels: bool = True
    augment: bool = False


@dataclass
class DLModelConfig:
    """Configuration for DL model architecture."""
    model_type: str = "MLP"
    input_dim: int = 4
    hidden_layers: list = field(default_factory=lambda: [64, 32])
    output_dim: int = 3
    activation: str = "relu"
    dropout_rate: float = 0.2


@dataclass
class DLTrainingConfig:
    """Configuration for DL training."""
    epochs: int = 100
    learning_rate: float = 0.001
    optimizer: str = "adam"
    loss: str = "categorical_crossentropy"
    early_stopping: bool = True
    patience: int = 10


@dataclass
class DLEvaluationResult:
    """Results from DL evaluation."""
    accuracy: float = 0.0
    loss: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    confusion_matrix: Optional[list] = None


class DLPipeline:
    """
    Deep Learning pipeline: Data → Preprocessing → Model → Training → Evaluation → Prediction.

    This class provides the structured skeleton for the DL workflow.
    Concrete implementations should subclass or extend this.
    """

    def __init__(
        self,
        data_config: Optional[DLDataConfig] = None,
        preprocessing_config: Optional[DLPreprocessingConfig] = None,
        model_config: Optional[DLModelConfig] = None,
        training_config: Optional[DLTrainingConfig] = None,
    ):
        self.data_config = data_config or DLDataConfig()
        self.preprocessing_config = preprocessing_config or DLPreprocessingConfig()
        self.model_config = model_config or DLModelConfig()
        self.training_config = training_config or DLTrainingConfig()
        self._model = None
        self._is_trained = False

    # --- Phase 1: Data ---
    def load_data(self) -> dict:
        """Load dataset. Returns raw data dict."""
        from sklearn import datasets
        iris = datasets.load_iris()
        return {
            "X": iris.data.tolist(),
            "y": iris.target.tolist(),
            "feature_names": list(iris.feature_names),
            "target_names": list(iris.target_names),
        }

    # --- Phase 2: Preprocessing ---
    def preprocess(self, data: dict) -> dict:
        """Preprocess data: normalize, split, encode."""
        import numpy as np
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split

        X = np.array(data["X"])
        y = np.array(data["y"])

        if self.preprocessing_config.normalize:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.data_config.test_size,
            random_state=self.data_config.random_state,
        )

        return {
            "X_train": X_train.tolist(),
            "X_test": X_test.tolist(),
            "y_train": y_train.tolist(),
            "y_test": y_test.tolist(),
        }

    # --- Phase 3: Model ---
    def build_model(self) -> Any:
        """Build concrete neural network model architecture using MLPClassifier."""
        from sklearn.neural_network import MLPClassifier

        # Map optimizer name if needed
        solver = self.training_config.optimizer.lower()
        if solver not in ("adam", "lbfgs", "sgd"):
            solver = "adam"

        model = MLPClassifier(
            hidden_layer_sizes=tuple(self.model_config.hidden_layers),
            activation=self.model_config.activation,
            solver=solver,
            max_iter=300,
            early_stopping=self.training_config.early_stopping,
            n_iter_no_change=self.training_config.patience,
            random_state=self.data_config.random_state,
        )
        self._model = model
        return model

    # --- Phase 4: Training ---
    def train(self, model: Any, preprocessed_data: dict) -> dict:
        """Train the model on preprocessed training data. Returns training history."""
        import numpy as np

        X_train = np.array(preprocessed_data["X_train"])
        y_train = np.array(preprocessed_data["y_train"])

        model.fit(X_train, y_train)
        self._model = model
        self._is_trained = True

        best_loss = getattr(model, "best_loss_", None)
        if best_loss is None:
            best_loss = getattr(model, "loss_", 0.0)
        if best_loss is None:
            best_loss = 0.0

        return {
            "epochs": getattr(model, "n_iter_", self.training_config.epochs),
            "loss": float(best_loss),
            "status": "trained",
            "n_layers": getattr(model, "n_layers_", len(self.model_config.hidden_layers) + 2),
            "n_outputs": getattr(model, "n_outputs_", self.model_config.output_dim),
        }

    # --- Phase 5: Evaluation ---
    def evaluate(self, model: Any, preprocessed_data: dict) -> DLEvaluationResult:
        """Evaluate trained model on test data partition."""
        import numpy as np
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        )

        X_test = np.array(preprocessed_data["X_test"])
        y_test = np.array(preprocessed_data["y_test"])

        y_pred = model.predict(X_test)

        acc = float(accuracy_score(y_test, y_pred))
        best_loss = getattr(model, "best_loss_", None)
        if best_loss is None:
            best_loss = getattr(model, "loss_", 0.0)
        loss = float(best_loss or 0.0)

        prec = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
        rec = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
        f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
        cm = confusion_matrix(y_test, y_pred).tolist()

        return DLEvaluationResult(
            accuracy=acc,
            loss=loss,
            precision=prec,
            recall=rec,
            f1_score=f1,
            confusion_matrix=cm,
        )

    # --- Phase 6: Prediction ---
    def predict(self, features: list[float], scaler: Any = None) -> dict:
        """Run inference on new input with genuine Softmax probability distribution."""
        import numpy as np

        if not self._is_trained or self._model is None:
            # If not yet trained, run pipeline to train
            self.run_pipeline()

        X_raw = np.array([features])
        if scaler is not None:
            X_input = scaler.transform(X_raw)
        else:
            X_input = X_raw

        pred_class = int(self._model.predict(X_input)[0])
        probas = self._model.predict_proba(X_input)[0].tolist()

        species_map = {0: "Iris setosa", 1: "Iris versicolor", 2: "Iris virginica"}
        return {
            "predicted_class_idx": pred_class,
            "predicted_species": species_map.get(pred_class, "Unknown"),
            "probabilities": {
                "setosa": round(probas[0] * 100, 2),
                "versicolor": round(probas[1] * 100, 2),
                "virginica": round(probas[2] * 100, 2),
            },
            "confidence": round(max(probas) * 100, 2),
            "model_type": "Deep Learning MLP",
            "status": "success",
        }

    def run_pipeline(self) -> dict:
        """Execute full pipeline end-to-end."""
        raw = self.load_data()
        processed = self.preprocess(raw)
        model = self.build_model()
        history = self.train(model, processed)
        eval_result = self.evaluate(model, processed)
        return {
            "data_loaded": True,
            "preprocessed": True,
            "model": model,
            "training_history": history,
            "evaluation": {
                "accuracy": eval_result.accuracy,
                "precision": eval_result.precision,
                "recall": eval_result.recall,
                "f1_score": eval_result.f1_score,
                "confusion_matrix": eval_result.confusion_matrix,
            },
            "status": "pipeline complete",
        }


def train_mlp_classifier(X_scaled, y, X_train, X_test, y_train, y_test, scaler) -> dict:
    """Train Deep Learning MLP model using the exact same evaluation protocol as SVM models.

    Uses architecture: [64, 32], ReLU activation, Adam optimizer, early stopping.
    Returns standard metric dictionary matching the ML/SVM pipeline.
    """
    import numpy as np
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, classification_report,
    )

    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        max_iter=300,
        early_stopping=True,
        n_iter_no_change=10,
        random_state=42,
    )

    cv_scores = cross_val_score(mlp, X_scaled, y, cv=5, scoring="accuracy")
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)

    return {
        "kernel": "mlp",
        "kernel_display": "Deep Learning MLP",
        "cv_mean": float(np.mean(cv_scores)),
        "cv_std": float(np.std(cv_scores)),
        "cv_scores": cv_scores.tolist(),
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
        "model": mlp,
        "scaler": scaler,
    }


def get_dl_status() -> dict:
    """Return current DL module status."""
    return {
        "module": "Deep Learning",
        "status": "ready",
        "pipeline_stages": [
            "Data",
            "Preprocessing",
            "Model",
            "Training",
            "Evaluation",
            "Prediction",
        ],
        "supported_models": ["MLP"],
        "dataset": "Iris",
    }

