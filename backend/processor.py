"""
DataProcessor — Tier 2 of the FirstPrinciple ML pipeline.

Steps (in order):
  1. Categorical encoding  (One-Hot / Label)
  2. Missing-value imputation  (mean fill or row drop)
  3. Null guard on user-selected features
  4. Feature scaling  (Z-score standardization or Min-Max)
  5. Correlation pruning  (remove multi-collinear features |r| > 0.9)
  6. 65:35 train-test split  (no sklearn)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Literal, Optional, Tuple


class DataProcessor:
    """Transform a raw DataFrame into clean train/test numpy arrays."""

    def __init__(self):
        self._encoding_map: dict = {}       # col → {original_value: encoded_value}
        self._scale_params: dict = {}       # col → (mean, std) or (min, max)
        self._dropped_cols: List[str] = []  # columns removed by correlation pruning
        self._ohe_columns: List[str] = []   # new OHE column names

    # ------------------------------------------------------------------
    # 1. Categorical Encoding
    # ------------------------------------------------------------------

    def encode_categoricals(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        strategy: Literal["onehot", "label"] = "onehot",
        columns: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Detect object/category dtype columns and encode them based on training data.
        """
        if columns is None:
            columns = df_train.select_dtypes(include=["object", "category"]).columns.tolist()

        if not columns:
            return df_train, df_test

        if strategy == "onehot":
            # Combine to ensure identical columns, then split back
            n_train = len(df_train)
            combined = pd.concat([df_train, df_test], axis=0)
            combined = pd.get_dummies(combined, columns=columns, drop_first=True, dtype=float)
            
            self._ohe_columns = [c for c in combined.columns if any(c.startswith(col) for col in columns)]
            
            df_train_enc = combined.iloc[:n_train].copy()
            df_test_enc = combined.iloc[n_train:].copy()
            return df_train_enc, df_test_enc
        else:
            for col in columns:
                unique_vals = df_train[col].dropna().unique().tolist()
                mapping = {v: i for i, v in enumerate(unique_vals)}
                self._encoding_map[col] = mapping
                
                df_train[col] = df_train[col].map(mapping).fillna(-1) # Handle unseen gently
                df_test[col] = df_test[col].map(mapping).fillna(-1)

        return df_train, df_test

    def split_raw(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.65,
        random_seed: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split raw dataframe into train and test sets early in the pipeline."""
        rng = np.random.default_rng(random_seed)
        idx = rng.permutation(len(df))
        split_point = int(len(df) * train_ratio)

        train_idx = idx[:split_point]
        test_idx = idx[split_point:]

        return df.iloc[train_idx].copy().reset_index(drop=True), df.iloc[test_idx].copy().reset_index(drop=True)

    def validate_and_impute(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        missing_threshold: float = 0.05
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Check if any column in Train has > missing_threshold (5%) nulls.
        If yes, raise ValueError.
        If no, impute missing values using Train's mean (numeric) or mode (categorical)
        and apply to both Train and Test.
        """
        n_train = len(df_train)
        if n_train == 0:
            return df_train, df_test

        null_counts = df_train.isnull().sum()
        null_ratios = null_counts / n_train

        # 1. Check > 5% missing
        problem_cols = null_ratios[null_ratios > missing_threshold]
        if not problem_cols.empty:
            details = ", ".join(f"{col} ({cnt} nulls, {ratio*100:.1f}%)" 
                                for col, cnt, ratio in zip(problem_cols.index, null_counts[problem_cols.index], problem_cols))
            raise ValueError(
                f"Data quality rejected: Columns have >{missing_threshold*100}% missing values: {details}. "
                "Please provide proper data."
            )

        # 2. Impute < 5% missing using Train stats
        for col in df_train.columns:
            if null_counts[col] > 0:
                if pd.api.types.is_numeric_dtype(df_train[col]):
                    fill_val = df_train[col].mean()
                else:
                    mode_series = df_train[col].mode()
                    fill_val = mode_series[0] if not mode_series.empty else "Unknown"
                
                df_train[col] = df_train[col].fillna(fill_val)
                # Apply same train fill_val to test
                if col in df_test.columns:
                    df_test[col] = df_test[col].fillna(fill_val)

        # Catch any remaining nulls in test (e.g., test had nulls but train didn't)
        # We fill them with train mean/mode as well
        test_nulls = df_test.isnull().sum()
        for col in test_nulls[test_nulls > 0].index:
            if pd.api.types.is_numeric_dtype(df_train[col]):
                fill_val = df_train[col].mean()
            else:
                mode_series = df_train[col].mode()
                fill_val = mode_series[0] if not mode_series.empty else "Unknown"
            # Fill test; if fill_val is NaN (e.g. all train was NaN, but caught above), fallback to 0
            if pd.isna(fill_val): fill_val = 0
            df_test[col] = df_test[col].fillna(fill_val)

        return df_train, df_test

    def check_nulls(self, df: pd.DataFrame, features: List[str], target: str) -> None:
        """
        Raise ValueError if any user-selected column still contains nulls.
        This acts as the explicit error halt required by the spec.
        """
        selected = features + [target]
        null_counts = df[selected].isnull().sum()
        problem_cols = null_counts[null_counts > 0]
        if not problem_cols.empty:
            details = ", ".join(f"{col}({cnt} nulls)" for col, cnt in problem_cols.items())
            raise ValueError(
                f"Selected features/target still contain missing values: {details}. "
                "Please run imputation first or drop the affected rows."
            )

    # ------------------------------------------------------------------
    # 4. Feature Scaling
    # ------------------------------------------------------------------

    def scale_features(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        features: List[str],
        strategy: Literal["zscore", "minmax"] = "zscore",
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Scale numeric feature columns in-place fitted on train, applied to both.
        """
        for col in features:
            if col not in df_train.columns:
                continue
            if not pd.api.types.is_numeric_dtype(df_train[col]):
                continue

            if strategy == "zscore":
                mu = df_train[col].mean()
                sigma = df_train[col].std()
                if sigma == 0:
                    sigma = 1  # avoid division by zero for constant columns
                self._scale_params[col] = (mu, sigma)
                
                df_train[col] = (df_train[col] - mu) / sigma
                if col in df_test.columns:
                    df_test[col] = (df_test[col] - mu) / sigma
            else:  # minmax
                mn = df_train[col].min()
                mx = df_train[col].max()
                rng = mx - mn if mx != mn else 1
                self._scale_params[col] = (mn, mx)
                
                df_train[col] = (df_train[col] - mn) / rng
                if col in df_test.columns:
                    df_test[col] = (df_test[col] - mn) / rng

        return df_train, df_test

    # ------------------------------------------------------------------
    # 5. Correlation Pruning
    # ------------------------------------------------------------------

    def prune_correlated(
        self, df_train: pd.DataFrame, df_test: pd.DataFrame, features: List[str], threshold: float = 0.9
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        """
        Compute Pearson correlation matrix on Training Data.
        For each pair with |r| > threshold, remove the second column from Train and Test.
        """
        numeric_feats = [f for f in features if pd.api.types.is_numeric_dtype(df_train[f])]
        if len(numeric_feats) < 2:
            return df_train, df_test, features

        corr = df_train[numeric_feats].corr().abs()
        upper_tri = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

        to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > threshold)]
        self._dropped_cols = to_drop

        df_train = df_train.drop(columns=to_drop, errors="ignore")
        df_test = df_test.drop(columns=to_drop, errors="ignore")
        surviving = [f for f in features if f not in to_drop]
        return df_train, df_test, surviving

    def get_correlation_matrix(self, df: pd.DataFrame, features: List[str]) -> dict:
        """Return correlation matrix as a JSON-serialisable dict for the frontend heatmap."""
        numeric_feats = [f for f in features if f in df.columns and pd.api.types.is_numeric_dtype(df[f])]
        if not numeric_feats:
            return {"columns": [], "values": []}
        corr = df[numeric_feats].corr().round(3)
        return {
            "columns": corr.columns.tolist(),
            "values": corr.values.tolist(),
        }

    # ------------------------------------------------------------------
    # 6. Extract numpy arrays
    # ------------------------------------------------------------------

    def extract_xy(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        features: List[str],
        target: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract X and y numpy arrays from the final dataframes.
        """
        X_train = df_train[features].values.astype(float)
        y_train = df_train[target].values.astype(float)
        
        X_test = df_test[features].values.astype(float)
        y_test = df_test[target].values.astype(float)

        return X_train, X_test, y_train, y_test
