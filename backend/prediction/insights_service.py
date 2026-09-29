"""Model Insights & Morphological Diagnostics Engine.

Provides deep analytical explanation of flower measurements:
- Statistical comparison against canonical Iris dataset distributions
- Z-score and percentile analysis across Setosa, Versicolor, Virginica
- Kernel agreement and boundary margin diagnostics
- Academic diagnostic explanation
"""
from __future__ import annotations

import numpy as np
from sklearn import datasets

# Precomputed empirical statistics from Fisher's 150-sample Iris dataset
_iris = datasets.load_iris()
_X = _iris.data
_y = _iris.target

# Statistics per class (0=Setosa, 1=Versicolor, 2=Virginica)
CLASS_NAMES = ["Iris setosa", "Iris versicolor", "Iris virginica"]
FEATURE_KEYS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
FEATURE_LABELS = {
    "sepal_length": "Sepal Length",
    "sepal_width": "Sepal Width",
    "petal_length": "Petal Length",
    "petal_width": "Petal Width",
}

EMPIRICAL_STATS = {}
for cls_idx, cls_name in enumerate(CLASS_NAMES):
    mask = (_y == cls_idx)
    cls_data = _X[mask]
    EMPIRICAL_STATS[cls_name] = {
        FEATURE_KEYS[i]: {
            "mean": round(float(np.mean(cls_data[:, i])), 2),
            "std": round(float(np.std(cls_data[:, i])), 2),
            "min": round(float(np.min(cls_data[:, i])), 2),
            "max": round(float(np.max(cls_data[:, i])), 2),
        }
        for i in range(4)
    }


def analyze_sample_insights(
    sepal_length: float,
    sepal_width: float,
    petal_length: float,
    petal_width: float,
    multi_kernel_results: dict,
) -> dict:
    """Analyze sample measurements against canonical botanical distributions and kernel consensus."""
    sample = {
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width,
    }

    # 1. Morphological distance (standardized squared distance / Mahalanobis-like metric per class)
    species_alignment = {}
    for cls_name in CLASS_NAMES:
        stats = EMPIRICAL_STATS[cls_name]
        z_scores = {}
        total_sq_z = 0.0
        for f_key in FEATURE_KEYS:
            mean = stats[f_key]["mean"]
            std = max(stats[f_key]["std"], 0.01)
            val = sample[f_key]
            z = (val - mean) / std
            z_scores[f_key] = round(float(z), 2)
            total_sq_z += z ** 2

        # Distance index (lower is closer)
        dist = round(float(np.sqrt(total_sq_z)), 2)
        species_alignment[cls_name] = {
            "z_scores": z_scores,
            "morphological_distance": dist,
            "within_typical_range": all(abs(z) <= 2.5 for z in z_scores.values()),
        }

    # Best matched botanical species by morphological distance
    closest_botanical = min(species_alignment, key=lambda s: species_alignment[s]["morphological_distance"])

    # 2. Key Discriminative Feature Analysis
    # Petal dimensions are known in botany as the primary separator between Setosa and other Iris
    pl = sample["petal_length"]
    pw = sample["petal_width"]

    discriminant_notes = []
    if pl <= 2.2:
        discriminant_notes.append(
            f"Petal Length ({pl} cm) is below 2.2 cm, which in Fisher's taxonomy is an absolute, linearly separable indicator for Iris setosa."
        )
    elif pl >= 4.8 and pw >= 1.7:
        discriminant_notes.append(
            f"Petal Length ({pl} cm) and Petal Width ({pw} cm) are large, strongly placing the sample within the upper cluster characteristic of Iris virginica."
        )
    elif 3.0 <= pl <= 5.1 and 1.0 <= pw <= 1.8:
        discriminant_notes.append(
            f"Petal dimensions ({pl} cm × {pw} cm) fall in the intermediate zone where Iris versicolor and Iris virginica feature distributions overlap."
        )
    else:
        discriminant_notes.append(
            f"Measurements exhibit an atypical aspect ratio (Petal: {pl} cm × {pw} cm, Sepal: {sample['sepal_length']} cm × {sample['sepal_width']} cm)."
        )

    # 3. Dynamic Multi-Classifier Consensus & Model Diagnostics (All 5 Classifiers)
    from collections import Counter
    results = multi_kernel_results.get("results", {})
    consensus = multi_kernel_results.get("consensus", "unknown")
    highest_conf = multi_kernel_results.get("highest_confidence", {})

    total_models = len(results) if results else 5
    predicted_species_list = [r.get("predicted_species", "") for r in results.values() if r.get("predicted_species")]
    counts = Counter(predicted_species_list) if predicted_species_list else Counter()

    if counts:
        top_species, top_count = counts.most_common(1)[0]
    else:
        top_species, top_count = highest_conf.get("predicted_species", "Unknown"), total_models

    model_diagnostics = []

    # Dynamic consensus note
    if consensus == "unanimous" or top_count == total_models:
        model_diagnostics.append(
            f"All {total_models} classifiers (SVM RBF, Linear, Polynomial, Sigmoid, and Deep Learning MLP) "
            f"reached unanimous consensus on {top_species}. This confirms absolute high confidence with clear "
            f"decision boundary separation across both non-linear kernel mappings and neural network representations."
        )
    elif consensus == "majority" or (top_count > total_models / 2):
        model_diagnostics.append(
            f"The classifiers demonstrated majority agreement ({top_count}/{total_models} models agree on {top_species}). "
            f"The divergence indicates the sample lies near the decision boundary margin where non-linear kernel curvature "
            f"(RBF, Polynomial, Sigmoid), maximum-margin flat hyperplanes (Linear), and deep neural representations "
            f"(MLP) resolve ambiguous boundary points with slight variances."
        )
    else:
        model_diagnostics.append(
            f"Classifiers diverged across models ({top_count}/{total_models} maximum agreement on {top_species}). "
            f"The sample lies directly within an overlapping morphological transition zone. "
            f"Refer to the highest-confidence classifier ({highest_conf.get('kernel_display', 'Best Model')}) "
            f"or inspect statistical morphological distance."
        )

    # SVM Sigmoid specific diagnostic
    if "sigmoid" in results:
        sig_info = results["sigmoid"]
        sig_spec = sig_info.get("predicted_species", "Unknown")
        sig_conf = sig_info.get("confidence", 0.0)
        model_diagnostics.append(
            f"SVM Sigmoid evaluates the sample via a hyperbolic tangent kernel function, predicting {sig_spec} "
            f"({sig_conf}% confidence). This non-linear mapping models S-curve boundary transitions akin to sigmoidal transfer."
        )

    # Deep Learning MLP specific diagnostic (explicitly identified as a Neural Network, NOT an SVM kernel)
    if "mlp" in results:
        mlp_info = results["mlp"]
        mlp_spec = mlp_info.get("predicted_species", "Unknown")
        mlp_conf = mlp_info.get("confidence", 0.0)
        model_diagnostics.append(
            f"Deep Learning MLP utilizes a multi-layer perceptron neural network (hidden layers: 64, 32) with ReLU activations "
            f"and Softmax distribution, predicting {mlp_spec} ({mlp_conf}% confidence). Its classification stems from latent "
            f"multi-dimensional feature transformations rather than kernel distance functions."
        )

    # Architecture-specific explanations
    model_explanations = {
        "rbf": "Radial Basis Function maps input features into an infinite-dimensional space, capturing localized Gaussian clusters.",
        "linear": "Linear SVM constructs maximum-margin flat hyperplanes, optimal for linearly separable classes.",
        "poly": "Polynomial SVM (degree 3) projects features into cross-product dimensions, detecting curved decision boundaries.",
        "sigmoid": "Sigmoid SVM employs hyperbolic tangent kernel transformations, modeling S-curve boundary transitions.",
        "mlp": "Deep Learning Multi-Layer Perceptron processes features through feedforward neural layers with ReLU activations and normalized Softmax probabilities.",
    }

    # Outlier detection
    outliers = []
    for f_key in FEATURE_KEYS:
        val = sample[f_key]
        if val < 0.5 or val > 15.0:
            outliers.append(f"{FEATURE_LABELS[f_key]} ({val} cm) is outside expected botanical boundaries.")

    return {
        "sample": sample,
        "closest_botanical_match": closest_botanical,
        "species_alignment": species_alignment,
        "discriminant_notes": discriminant_notes,
        "kernel_diagnostics": model_diagnostics,
        "model_diagnostics": model_diagnostics,
        "model_explanations": model_explanations,
        "consensus": consensus,
        "consensus_text": multi_kernel_results.get("consensus_text", ""),
        "highest_confidence": highest_conf,
        "empirical_benchmarks": EMPIRICAL_STATS,
        "outliers": outliers,
    }
