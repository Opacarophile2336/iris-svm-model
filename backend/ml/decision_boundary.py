"""Decision boundary generator for SVM (2D projection using PCA or feature pairs)."""
from __future__ import annotations

import base64
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA

import config

# Professional, readable color palette
COLORS = ["#4E79A7", "#F28E2B", "#59A14F"]
CMAP = mcolors.ListedColormap(["#AEC6E8", "#FBDAB1", "#B5D9AD"])


def generate_decision_boundary(kernel: str, feature_x: int = 2, feature_y: int = 3) -> str:
    """Return base64-encoded PNG of 2D decision boundary.

    feature_x/y indices: 0=sepal_length, 1=sepal_width, 2=petal_length, 3=petal_width
    """
    if kernel not in config.ALLOWED_KERNELS:
        raise ValueError(f"Kernel '{kernel}' not supported. Choose from: {config.ALLOWED_KERNELS}")

    iris = datasets.load_iris()
    X = iris.data[:, [feature_x, feature_y]]
    y = iris.target
    feature_names = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = SVC(kernel=kernel, probability=True, random_state=42, gamma="scale", C=1.0)
    model.fit(X_scaled, y)

    x_min, x_max = X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5
    y_min, y_max = X_scaled[:, 1].min() - 0.5, X_scaled[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.contourf(xx, yy, Z, cmap=CMAP, alpha=0.7)

    for cls_idx, cls_name in enumerate(iris.target_names):
        mask = y == cls_idx
        ax.scatter(
            X_scaled[mask, 0], X_scaled[mask, 1],
            c=COLORS[cls_idx], label=cls_name.capitalize(),
            edgecolors="white", linewidths=0.5, s=60, zorder=3,
        )

    ax.set_xlabel(feature_names[feature_x], fontsize=12)
    ax.set_ylabel(feature_names[feature_y], fontsize=12)
    ax.set_title(f"SVM Decision Boundary — {kernel.upper()} Kernel", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_facecolor("#F8F9FA")
    fig.patch.set_facecolor("white")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()
