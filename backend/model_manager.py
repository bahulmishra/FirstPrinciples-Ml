"""
ModelManager — Tier 3 of the FirstPrinciple ML pipeline.

All algorithms are implemented using pure NumPy — NO scikit-learn estimators.

Algorithms
──────────
Regression:
  • Linear Regression  (Gradient Descent on MSE)
  • Hard-Margin SVM    (Hinge Loss, subgradient descent)
  • Soft-Margin SVM    (Slack Variable with C parameter)

Classification:
  • K-Nearest Neighbors  (Euclidean distance)
  • Fuzzy C-Means        (FCM — membership degree matrix)
  • Possibilistic C-Means (PCM — typicality, outlier-robust)

Cross-Validation:
  • K-Fold (custom loop, no sklearn)

Metrics:
  • R² score  (regression)
  • Accuracy  (classification)
"""
from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple


# ══════════════════════════════════════════════════════════════════════
# Helper utilities
# ══════════════════════════════════════════════════════════════════════

def _augment_bias(X: np.ndarray) -> np.ndarray:
    """Prepend a column of 1s to X for the intercept/bias term."""
    return np.hstack([np.ones((X.shape[0], 1)), X])


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² = 1 - SS_res / SS_tot"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return float(1 - ss_res / ss_tot)


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def _euclidean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vectorised Euclidean distance: each row in a vs every row in b."""
    # shape: (n_a, n_b)
    diff = a[:, np.newaxis, :] - b[np.newaxis, :, :]  # (n_a, n_b, d)
    return np.sqrt(np.sum(diff ** 2, axis=2))


# ══════════════════════════════════════════════════════════════════════
# K-Fold Cross-Validation
# ══════════════════════════════════════════════════════════════════════

class KFoldCV:
    """Custom K-Fold cross-validation (no sklearn)."""

    def __init__(self, k: int = 5, random_seed: int = 42):
        self.k = k
        self.rng = np.random.default_rng(random_seed)

    def split(
        self, X: np.ndarray, y: np.ndarray
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Yield (X_train, X_val, y_train, y_val) for each fold."""
        n = len(X)
        idx = self.rng.permutation(n)
        fold_sizes = np.full(self.k, n // self.k)
        fold_sizes[: n % self.k] += 1

        folds = []
        current = 0
        for size in fold_sizes:
            val_idx = idx[current : current + size]
            train_idx = np.concatenate([idx[:current], idx[current + size :]])
            folds.append((X[train_idx], X[val_idx], y[train_idx], y[val_idx]))
            current += size
        return folds


# ══════════════════════════════════════════════════════════════════════
# Regression Algorithms
# ══════════════════════════════════════════════════════════════════════

class LinearRegressionGD:
    """
    Linear Regression optimised with Batch Gradient Descent.

    Update rule: θ = θ - α · (1/m) · Xᵀ(Xθ - y)
    Derivative of MSE with respect to θ.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        epochs: int = 1000,
        tolerance: float = 1e-4,
    ):
        self.lr = learning_rate
        self.epochs = epochs
        self.tolerance = tolerance
        self.theta: Optional[np.ndarray] = None
        self.loss_history: List[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressionGD":
        X_b = _augment_bias(X)
        m, n = X_b.shape
        self.theta = np.zeros(n)
        prev_loss = float("inf")

        for _ in range(self.epochs):
            y_pred = X_b @ self.theta
            error = y_pred - y
            loss = float(np.mean(error ** 2))
            self.loss_history.append(loss)

            # Derivative of mean(error^2) = 2/m * (X^T @ error)
            gradient = (2 / m) * (X_b.T @ error)
            self.theta -= self.lr * gradient

            if abs(prev_loss - loss) < self.tolerance:
                break
            prev_loss = loss
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return _augment_bias(X) @ self.theta

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return _r2_score(y, self.predict(X))


class HardSVMRegressor:
    """
    Hard-Margin SVM for Regression using Hinge Loss (subgradient descent).
    Loss = max(0, |y - (w·x + b)| - ε)
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        epochs: int = 1000,
        epsilon: float = 0.1,
        tolerance: float = 1e-4,
    ):
        self.lr = learning_rate
        self.epochs = epochs
        self.epsilon = epsilon
        self.tolerance = tolerance
        self.w: Optional[np.ndarray] = None
        self.b: float = 0.0
        self.loss_history: List[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HardSVMRegressor":
        m, n = X.shape
        self.w = np.zeros(n)
        self.b = 0.0
        prev_loss = float("inf")

        for _ in range(self.epochs):
            y_pred = X @ self.w + self.b
            residuals = y - y_pred
            # Hinge: loss per sample
            loss = float(np.mean(np.maximum(0, np.abs(residuals) - self.epsilon)))
            self.loss_history.append(loss)

            # Subgradient
            mask_pos = residuals < -self.epsilon   # prediction too high
            mask_neg = residuals > self.epsilon    # prediction too low
            grad_w = np.zeros(n)
            grad_b = 0.0

            grad_w += X[mask_pos].sum(axis=0)     # push w up
            grad_w -= X[mask_neg].sum(axis=0)     # push w down
            grad_b += mask_pos.sum() - mask_neg.sum()

            grad_w /= m
            grad_b /= m

            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b

            if abs(prev_loss - loss) < self.tolerance:
                break
            prev_loss = loss
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.w + self.b

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return _r2_score(y, self.predict(X))


class SoftSVMRegressor:
    """
    Soft-Margin SVM for Regression with slack variable (C parameter).
    Loss = (1/2)||w||² + C · Σ max(0, |y - (w·x + b)| - ε)
    Uses subgradient descent on the combined objective.
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        epochs: int = 1000,
        epsilon: float = 0.1,
        C: float = 1.0,
        tolerance: float = 1e-4,
    ):
        self.lr = learning_rate
        self.epochs = epochs
        self.epsilon = epsilon
        self.C = C
        self.tolerance = tolerance
        self.w: Optional[np.ndarray] = None
        self.b: float = 0.0
        self.loss_history: List[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SoftSVMRegressor":
        m, n = X.shape
        self.w = np.zeros(n)
        self.b = 0.0
        prev_loss = float("inf")

        for _ in range(self.epochs):
            y_pred = X @ self.w + self.b
            residuals = y - y_pred

            hinge = np.maximum(0, np.abs(residuals) - self.epsilon)
            
            # Use objective scaled by 1/m to prevent explosion relative to gradients:
            # Loss = (1/(2m)) ||w||^2 + C * mean(hinge)
            loss = (0.5 / m) * float(np.dot(self.w, self.w)) + self.C * float(np.mean(hinge))
            self.loss_history.append(loss)

            mask_pos = residuals < -self.epsilon
            mask_neg = residuals > self.epsilon

            # Gradients of the objective w.r.t w and b
            grad_w = self.w.copy() / m
            grad_b = 0.0
            grad_w += self.C / m * (X[mask_pos].sum(axis=0) - X[mask_neg].sum(axis=0))
            grad_b += self.C / m * float(mask_pos.sum() - mask_neg.sum())

            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b

            if abs(prev_loss - loss) < self.tolerance:
                break
            prev_loss = loss
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.w + self.b

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return _r2_score(y, self.predict(X))


# ══════════════════════════════════════════════════════════════════════
# Classification Algorithms
# ══════════════════════════════════════════════════════════════════════

class KNNClassifier:
    """
    K-Nearest Neighbours (Euclidean distance, majority vote).
    """

    def __init__(self, k: int = 5):
        self.k = k
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNClassifier":
        self.X_train = X
        self.y_train = y
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        dist_matrix = _euclidean(X, self.X_train)      # (n_test, n_train)
        k_idx = np.argsort(dist_matrix, axis=1)[:, : self.k]
        k_labels = self.y_train[k_idx]                 # (n_test, k)
        # Majority vote per row
        predictions = np.array([
            np.bincount(row.astype(int)).argmax() for row in k_labels
        ])
        return predictions

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return _accuracy(y, self.predict(X))


class FuzzyCMeans:
    """
    Fuzzy C-Means Clustering.
    Each point has a degree of membership to every cluster.
    Update equations:
      u_ij = 1 / Σ_k (d_ij / d_ik)^(2/(m-1))
      c_j  = Σ_i u_ij^m · x_i / Σ_i u_ij^m
    """

    def __init__(
        self,
        n_clusters: int = 3,
        m: float = 2.0,
        max_iter: int = 300,
        tolerance: float = 1e-4,
        random_seed: int = 42,
    ):
        self.c = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.tol = tolerance
        self.rng = np.random.default_rng(random_seed)
        self.centroids: Optional[np.ndarray] = None
        self.U: Optional[np.ndarray] = None          # membership matrix (n, c)
        self.labels_: Optional[np.ndarray] = None

    def _init_membership(self, n: int) -> np.ndarray:
        U = self.rng.random((n, self.c))
        U /= U.sum(axis=1, keepdims=True)
        return U

    def _update_centroids(self, X: np.ndarray) -> np.ndarray:
        U_m = self.U ** self.m                        # (n, c)
        return (U_m.T @ X) / U_m.sum(axis=0)[:, np.newaxis]  # (c, d)

    def _update_membership(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        dists = _euclidean(X, self.centroids)         # (n, c)
        dists = np.maximum(dists, 1e-10)              # avoid div/0
        exp = 2.0 / (self.m - 1)
        U_new = np.zeros((n, self.c))
        for j in range(self.c):
            ratios = (dists[:, j:j+1] / dists) ** exp   # (n, c)
            U_new[:, j] = 1.0 / ratios.sum(axis=1)
        return U_new

    def fit(self, X: np.ndarray) -> "FuzzyCMeans":
        n = X.shape[0]
        self.U = self._init_membership(n)

        for _ in range(self.max_iter):
            self.centroids = self._update_centroids(X)
            U_new = self._update_membership(X)
            if np.max(np.abs(U_new - self.U)) < self.tol:
                self.U = U_new
                break
            self.U = U_new

        self.labels_ = np.argmax(self.U, axis=1)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        U = self._update_membership(X)
        return np.argmax(U, axis=1)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Accuracy requires mapping cluster labels to true class labels."""
        pred = self.predict(X)
        return _cluster_accuracy(y.astype(int), pred, self.c)


class PossibilisticCMeans:
    """
    Possibilistic C-Means (PCM) — outlier-robust variant of FCM.
    Uses typicality τ instead of constrained membership.
    τ_ij = 1 / (1 + (d_ij² / η_j)^(1/(m-1)))
    η_j estimated from FCM membership as: η_j = Σ u_ij^m d_ij² / Σ u_ij^m
    """

    def __init__(
        self,
        n_clusters: int = 3,
        m: float = 2.0,
        max_iter: int = 300,
        tolerance: float = 1e-4,
        random_seed: int = 42,
    ):
        self.c = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.tol = tolerance
        self.rng = np.random.default_rng(random_seed)
        self.centroids: Optional[np.ndarray] = None
        self.T: Optional[np.ndarray] = None          # typicality matrix (n, c)
        self.labels_: Optional[np.ndarray] = None
        self._eta: Optional[np.ndarray] = None       # (c,) bandwidth params

    def _init_from_fcm(self, X: np.ndarray) -> None:
        """Bootstrap centroids + η from a quick FCM run."""
        fcm = FuzzyCMeans(n_clusters=self.c, m=self.m, max_iter=50, random_seed=42)
        fcm.fit(X)
        self.centroids = fcm.centroids.copy()
        # Compute η_j per cluster
        dists_sq = _euclidean(X, self.centroids) ** 2  # (n, c)
        U_m = fcm.U ** self.m
        self._eta = (U_m * dists_sq).sum(axis=0) / (U_m.sum(axis=0) + 1e-10)
        self._eta = np.maximum(self._eta, 1e-10)

    def _update_typicality(self, X: np.ndarray) -> np.ndarray:
        dists_sq = _euclidean(X, self.centroids) ** 2  # (n, c)
        exp = 1.0 / (self.m - 1)
        T = 1.0 / (1.0 + (dists_sq / self._eta) ** exp)
        return T

    def _update_centroids(self, X: np.ndarray) -> np.ndarray:
        T_m = self.T ** self.m
        return (T_m.T @ X) / (T_m.sum(axis=0)[:, np.newaxis] + 1e-10)

    def fit(self, X: np.ndarray) -> "PossibilisticCMeans":
        self._init_from_fcm(X)
        n = X.shape[0]
        self.T = self._update_typicality(X)

        for _ in range(self.max_iter):
            new_centroids = self._update_centroids(X)
            T_new = self._update_typicality(X)
            if np.max(np.abs(T_new - self.T)) < self.tol:
                self.T = T_new
                self.centroids = new_centroids
                break
            self.T = T_new
            self.centroids = new_centroids

        self.labels_ = np.argmax(self.T, axis=1)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        T = self._update_typicality(X)
        return np.argmax(T, axis=1)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        pred = self.predict(X)
        return _cluster_accuracy(y.astype(int), pred, self.c)


# ══════════════════════════════════════════════════════════════════════
# Helper for cluster → class label mapping
# ══════════════════════════════════════════════════════════════════════

def _cluster_accuracy(y_true: np.ndarray, y_pred: np.ndarray, n_clusters: int) -> float:
    """
    Map cluster indices to class labels via a confusion-matrix greedy assignment.
    Uses the Hungarian-like greedy approach: for each cluster pick the most
    common true class.
    """
    n_classes = len(np.unique(y_true))
    mapping = {}
    for cluster_id in range(n_clusters):
        mask = y_pred == cluster_id
        if mask.sum() == 0:
            mapping[cluster_id] = 0
            continue
        labels_in_cluster = y_true[mask]
        mapping[cluster_id] = int(np.bincount(labels_in_cluster, minlength=n_classes).argmax())
    mapped = np.array([mapping[p] for p in y_pred])
    return _accuracy(y_true, mapped)


# ══════════════════════════════════════════════════════════════════════
# ModelManager — orchestrates all of the above
# ══════════════════════════════════════════════════════════════════════

class ModelManager:
    """
    High-level interface that:
    1. Selects the right algorithm(s) based on task type
    2. Runs K-Fold CV to report stability scores
    3. Trains on the full train set and evaluates on test set
    4. Returns metrics + plotting data for the frontend
    """

    REGRESSION_ALGORITHMS = ["linear_regression", "hard_svm", "soft_svm"]
    CLASSIFICATION_ALGORITHMS = ["knn", "fcm", "pcm"]

    def __init__(self, hyperparams: dict):
        """
        hyperparams is a dictionary of parameter lists. E.g.:
        {
          "linear_regression": {"learning_rate": [0.1, 0.01], "epochs": [1000]},
          "knn": {"k": [3, 5, 7]}
        }
        """
        self.hyperparams_grid = hyperparams

    def _generate_grid(self, alg: str) -> List[Dict[str, Any]]:
        """Cartesian product of hyperparams for an algorithm. Returns a list of parameter dicts."""
        import itertools
        grid = self.hyperparams_grid.get(alg, {})
        if not grid:
            return [{}]  # Default single run with no specific params
            
        keys, values = zip(*grid.items())
        # values should be lists
        list_values = [v if isinstance(v, list) else [v] for v in values]
        combinations = [dict(zip(keys, v)) for v in itertools.product(*list_values)]
        return combinations

    def _make_model(self, algorithm: str, hp: dict):
        """Instantiate a model with a specific set of hyperparameters."""
        if algorithm == "linear_regression":
            return LinearRegressionGD(
                learning_rate=float(hp.get("learning_rate", 0.01)),
                epochs=int(hp.get("epochs", 1000)),
                tolerance=float(hp.get("tolerance", 1e-4)),
            )
        elif algorithm == "hard_svm":
            return HardSVMRegressor(
                learning_rate=float(hp.get("learning_rate", 0.001)),
                epochs=int(hp.get("epochs", 1000)),
                epsilon=float(hp.get("epsilon", 0.1)),
                tolerance=float(hp.get("tolerance", 1e-4)),
            )
        elif algorithm == "soft_svm":
            return SoftSVMRegressor(
                learning_rate=float(hp.get("learning_rate", 0.001)),
                epochs=int(hp.get("epochs", 1000)),
                epsilon=float(hp.get("epsilon", 0.1)),
                C=float(hp.get("C", 1.0)),
                tolerance=float(hp.get("tolerance", 1e-4)),
            )
        elif algorithm == "knn":
            return KNNClassifier(k=int(hp.get("k", 5)))
        elif algorithm == "fcm":
            return FuzzyCMeans(
                n_clusters=int(hp.get("n_clusters", 3)),
                m=float(hp.get("m", 2.0)),
                max_iter=int(hp.get("epochs", 300)),
                tolerance=float(hp.get("tolerance", 1e-4)),
            )
        elif algorithm == "pcm":
            return PossibilisticCMeans(
                n_clusters=int(hp.get("n_clusters", 3)),
                m=float(hp.get("m", 2.0)),
                max_iter=int(hp.get("epochs", 300)),
                tolerance=float(hp.get("tolerance", 1e-4)),
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")



    def run(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        task_type: str,            # "regression" | "classification"
        algorithms: List[str],
        k_folds: int = 5,
    ) -> Dict:
        """
        Train each algorithm, run K-Fold Grid Search over hyperparams, 
        select best params, and compute test metrics.
        """
        kfold = KFoldCV(k=max(2, k_folds), random_seed=42)
        folds = kfold.split(X_train, y_train)

        results = {}
        for alg in algorithms:
            # 1. Grid search over combinations
            param_combinations = self._generate_grid(alg)
            best_hp = None
            best_cv_mean = -float("inf")
            best_cv_scores = []

            for hp in param_combinations:
                cv_scores = []
                for X_tr, X_val, y_tr, y_val in folds:
                    try:
                        m = self._make_model(alg, hp)
                        if task_type == "regression":
                            m.fit(X_tr, y_tr)
                        else:
                            if alg in ("fcm", "pcm"):
                                m.fit(X_tr)
                            else:
                                m.fit(X_tr, y_tr)
                        cv_scores.append(m.score(X_val, y_val))
                    except Exception:
                        cv_scores.append(0.0)
                
                mean_cv = np.mean(cv_scores) if cv_scores else 0.0
                if mean_cv > best_cv_mean or best_hp is None:
                    best_cv_mean = float(mean_cv)
                    best_cv_scores = cv_scores
                    best_hp = hp
            
            if not best_hp:
                best_hp = {}

            # 2. Final model on full train set using best_hp
            model = self._make_model(alg, best_hp)
            try:
                if task_type == "regression":
                    model.fit(X_train, y_train)
                    test_score = model.score(X_test, y_test)
                    y_pred = model.predict(X_test)
                    loss_history = getattr(model, "loss_history", [])
                else:
                    if alg in ("fcm", "pcm"):
                        model.fit(X_train)
                    else:
                        model.fit(X_train, y_train)
                    test_score = model.score(X_test, y_test)
                    y_pred = model.predict(X_test)
                    loss_history = []
            except Exception as e:
                test_score = 0.0
                y_pred = np.zeros_like(y_test)
                loss_history = []

            results[alg] = {
                "test_score": round(float(test_score), 4),
                "cv_scores": [round(float(s), 4) for s in best_cv_scores],
                "cv_mean": round(best_cv_mean, 4),
                "cv_std": round(float(np.std(best_cv_scores)) if best_cv_scores else 0.0, 4),
                "best_hyperparameters": best_hp,
                "y_pred": y_pred.tolist(),
                "y_test": y_test.tolist(),
                "loss_history": loss_history,
                "model": model,
            }

        # Build leaderboard based on test_score (or cv_mean, but test_score is final evaluation)
        best_alg = max(results, key=lambda a: results[a]["test_score"])
        leaderboard = sorted(
            results.items(), key=lambda x: x[1]["test_score"], reverse=True
        )

        return {
            "results": {k: {kk: vv for kk, vv in v.items() if kk != "model"}
                        for k, v in results.items()},
            "best_algorithm": best_alg,
            "leaderboard": [
                {
                    "algorithm": a, 
                    "test_score": r["test_score"], 
                    "cv_mean": r["cv_mean"],
                    "best_hyperparameters": r["best_hyperparameters"]
                }
                for a, r in leaderboard
            ],
            "models": {k: v["model"] for k, v in results.items()},
        }
