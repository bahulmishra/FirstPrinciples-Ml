"""
FirstPrinciple ML — FastAPI Application
Entry point: uvicorn main:app --reload --port 8000
"""
from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import session_store as store
from ingestor import DataIngestor
from model_manager import ModelManager
from processor import DataProcessor

# ── App init ──────────────────────────────────────────────────────────
app = FastAPI(title="FirstPrinciple ML API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
SAMPLE_DIR = Path(__file__).parent.parent / "sample_data"

# Serve static frontend files
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ── Pydantic request models ───────────────────────────────────────────

class URLUploadRequest(BaseModel):
    url: str


class ConfigureRequest(BaseModel):
    session_id: str
    features: List[str]
    target: str
    task_type: str              # "regression" | "classification"
    encoding_strategy: str = "onehot"   # "onehot" | "label"
    scaling_strategy: str = "zscore"    # "zscore" | "minmax"


class TrainRequest(BaseModel):
    session_id: str
    algorithms: List[str]
    k_folds: int = 5
    hyperparams: Dict[str, Any] = {}


# ── Utility ───────────────────────────────────────────────────────────

def _safe_json(obj):
    """Recursively convert numpy types → Python native for JSON serialisation."""
    if isinstance(obj, dict):
        return {k: _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_safe_json(i) for i in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _pca_2d(X: np.ndarray) -> np.ndarray:
    """Manual PCA to 2 components (used for decision-surface visualisation)."""
    X_c = X - X.mean(axis=0)
    cov = (X_c.T @ X_c) / max(len(X) - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Sort eigenvectors by descending eigenvalue
    order = np.argsort(eigenvalues)[::-1]
    components = eigenvectors[:, order[:2]]
    return X_c @ components


def _build_decision_surface(model, X_2d, y, task_type, algorithm):
    """Build a grid for decision boundary or regression line (2-D projection)."""
    x1_min, x1_max = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
    x2_min, x2_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5

    grid_res = 60
    xx1, xx2 = np.meshgrid(
        np.linspace(x1_min, x1_max, grid_res),
        np.linspace(x2_min, x2_max, grid_res),
    )
    grid = np.c_[xx1.ravel(), xx2.ravel()]

    try:
        if task_type == "classification":
            if algorithm in ("fcm", "pcm"):
                zz = model.predict(grid)
            else:
                zz = model.predict(grid)
        else:
            zz = model.predict(grid)
    except Exception:
        zz = np.zeros(len(grid))

    return {
        "x1": xx1[0].tolist(),
        "x2": xx2[:, 0].tolist(),
        "z": zz.reshape(grid_res, grid_res).tolist(),
        "scatter_x1": X_2d[:, 0].tolist(),
        "scatter_x2": X_2d[:, 1].tolist(),
        "scatter_y": y.tolist(),
    }


# ── Routes ────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "FirstPrinciple ML API is running. Frontend not found."}


@app.get("/api/sample-data")
async def list_sample_data():
    if not SAMPLE_DIR.exists():
        return {"datasets": []}
    files = [f.name for f in SAMPLE_DIR.iterdir() if f.suffix in (".csv", ".json")]
    return {"datasets": files}


@app.get("/api/sample-data/{filename}")
async def download_sample_data(filename: str):
    path = SAMPLE_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Sample dataset not found.")
    return FileResponse(str(path), filename=filename)


# ── Column values (for frontend distribution chart) ───────────────────

@app.post("/api/column-values")
async def column_values(body: dict):
    """Return all raw values for the requested columns from the session Training DataFrame."""
    sid = body.get("session_id")
    cols = body.get("columns", [])
    if not sid:
        raise HTTPException(status_code=400, detail="session_id required.")
    df_train: pd.DataFrame = store.get_key(sid, "df_train")
    if df_train is None:
        raise HTTPException(status_code=404, detail="Session not found.")

    result = {}
    for col in cols:
        if col in df_train.columns and pd.api.types.is_numeric_dtype(df_train[col]):
            result[col] = df_train[col].dropna().tolist()
    return JSONResponse(_safe_json({"values": result}))


# ── Upload ────────────────────────────────────────────────────────────

@app.post("/api/upload/file")
async def upload_file(file: UploadFile = File(...)):
    ingestor = DataIngestor()
    proc = DataProcessor()
    content = await file.read()
    fname = file.filename or ""
    try:
        if fname.endswith(".json"):
            df = ingestor.load_json(content)
        else:
            # Try CSV by default
            try:
                df = ingestor.load_csv(content)
            except Exception:
                df = ingestor.load_json(content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {e}")

    # 1. Split immediately
    df_train, df_test = proc.split_raw(df)
    
    # 2. Check nulls & auto-impute (<5% -> train mean/mode; >5% -> reject)
    try:
        df_train, df_test = proc.validate_and_impute(df_train, df_test)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    sid = store.create_session()
    store.set_key(sid, "df_train", df_train)
    store.set_key(sid, "df_test", df_test)
    store.set_key(sid, "filename", fname)

    # Use training dataframe for column types and preview
    columns = ingestor.get_columns(df_train)
    preview = df_train.head(5).to_dict(orient="records")

    return JSONResponse(
        _safe_json({
            "session_id": sid,
            "columns": columns,
            "preview": preview,
            "shape": {"rows": len(df_train), "cols": len(df_train.columns)},
        })
    )


@app.post("/api/upload/url")
async def upload_url(body: URLUploadRequest):
    ingestor = DataIngestor()
    proc = DataProcessor()
    try:
        df = ingestor.load_url(body.url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {e}")

    # 1. Split immediately
    df_train, df_test = proc.split_raw(df)
    
    # 2. Check nulls & auto-impute (<5% -> train mean; >5% -> reject)
    try:
        df_train, df_test = proc.validate_and_impute(df_train, df_test)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    sid = store.create_session()
    store.set_key(sid, "df_train", df_train)
    store.set_key(sid, "df_test", df_test)
    store.set_key(sid, "filename", body.url)

    columns = ingestor.get_columns(df_train)
    preview = df_train.head(5).to_dict(orient="records")

    return JSONResponse(
        _safe_json({
            "session_id": sid,
            "columns": columns,
            "preview": preview,
            "shape": {"rows": len(df_train), "cols": len(df_train.columns)},
        })
    )


# ── Configure ─────────────────────────────────────────────────────────

@app.post("/api/configure")
async def configure(req: ConfigureRequest):
    session = store.get_session(req.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")

    df_train: pd.DataFrame = store.get_key(req.session_id, "df_train")

    missing_cols = [c for c in req.features + [req.target] if c not in df_train.columns]
    if missing_cols:
        raise HTTPException(
            status_code=400,
            detail=f"Columns not found in dataset: {missing_cols}",
        )

    store.set_key(req.session_id, "features", req.features)
    store.set_key(req.session_id, "target", req.target)
    store.set_key(req.session_id, "task_type", req.task_type)
    store.set_key(req.session_id, "encoding_strategy", req.encoding_strategy)
    store.set_key(req.session_id, "scaling_strategy", req.scaling_strategy)

    return {"status": "configured", "session_id": req.session_id}


# ── Preprocess ────────────────────────────────────────────────────────

@app.post("/api/preprocess")
async def preprocess(body: dict):
    sid = body.get("session_id")
    if not sid:
        raise HTTPException(status_code=400, detail="session_id required.")
    session = store.get_session(sid)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")

    df_train: pd.DataFrame = store.get_key(sid, "df_train").copy()
    df_test: pd.DataFrame = store.get_key(sid, "df_test").copy()
    features: List[str] = store.get_key(sid, "features")
    target: str = store.get_key(sid, "target")
    task_type: str = store.get_key(sid, "task_type")
    encoding_strategy = store.get_key(sid, "encoding_strategy", "onehot")
    scaling_strategy = store.get_key(sid, "scaling_strategy", "zscore")

    proc = DataProcessor()

    # 1. Encode categoricals
    cat_columns = df_train.select_dtypes(include=["object", "category"]).columns.tolist()
    if target in cat_columns and encoding_strategy == "onehot":
        cat_columns.remove(target)
        
    df_train, df_test = proc.encode_categoricals(df_train, df_test, strategy=encoding_strategy, columns=cat_columns)

    all_cols = df_train.columns.tolist()
    expanded_features = []
    for f in features:
        if f in all_cols:
            expanded_features.append(f)
        else:
            expanded_features.extend([c for c in all_cols if c.startswith(f + "_")])

    # Target: encode if categorical
    if target in df_train.columns and not pd.api.types.is_numeric_dtype(df_train[target]):
        unique_vals = df_train[target].dropna().unique().tolist()
        t_mapping = {v: i for i, v in enumerate(unique_vals)}
        df_train[target] = df_train[target].map(t_mapping)
        df_test[target] = df_test[target].map(t_mapping)
        store.set_key(sid, "target_mapping", t_mapping)

    # 3. Null guard - should not trigger if validation at upload did its job properly, but good as a sanity check
    try:
        proc.check_nulls(df_train, expanded_features, target)
        proc.check_nulls(df_test, expanded_features, target)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # 4. Correlation matrix (from train)
    corr_data = proc.get_correlation_matrix(df_train, expanded_features)

    # 5. Prune correlated features
    df_train, df_test, surviving_features = proc.prune_correlated(df_train, df_test, expanded_features)

    # 6. Scale features
    df_train, df_test = proc.scale_features(df_train, df_test, surviving_features, strategy=scaling_strategy)

    # 7. Extract arrays
    X_train, X_test, y_train, y_test = proc.extract_xy(df_train, df_test, surviving_features, target)

    # Store processed data
    store.set_key(sid, "X_train", X_train)
    store.set_key(sid, "X_test", X_test)
    store.set_key(sid, "y_train", y_train)
    store.set_key(sid, "y_test", y_test)
    store.set_key(sid, "surviving_features", surviving_features)
    store.set_key(sid, "dropped_cols", proc._dropped_cols)

    return JSONResponse(
        _safe_json({
            "status": "preprocessed",
            "train_shape": list(X_train.shape),
            "test_shape": list(X_test.shape),
            "surviving_features": surviving_features,
            "dropped_cols": proc._dropped_cols,
            "correlation_matrix": corr_data,
        })
    )


# ── Train ─────────────────────────────────────────────────────────────

@app.post("/api/train")
async def train(req: TrainRequest):
    session = store.get_session(req.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")

    X_train = store.get_key(req.session_id, "X_train")
    X_test = store.get_key(req.session_id, "X_test")
    y_train = store.get_key(req.session_id, "y_train")
    y_test = store.get_key(req.session_id, "y_test")
    task_type = store.get_key(req.session_id, "task_type")
    surviving_features = store.get_key(req.session_id, "surviving_features")

    if X_train is None:
        raise HTTPException(
            status_code=400, detail="Data not preprocessed. Call /api/preprocess first."
        )

    # Validate algorithms match task type
    REGRESSION_ALGS = {"linear_regression", "hard_svm", "soft_svm"}
    CLASSIFICATION_ALGS = {"knn", "fcm", "pcm"}
    for alg in req.algorithms:
        if task_type == "regression" and alg not in REGRESSION_ALGS:
            raise HTTPException(
                status_code=400,
                detail=f"Algorithm '{alg}' is not a regression algorithm.",
            )
        if task_type == "classification" and alg not in CLASSIFICATION_ALGS:
            raise HTTPException(
                status_code=400,
                detail=f"Algorithm '{alg}' is not a classification algorithm.",
            )

    manager = ModelManager(hyperparams=req.hyperparams)
    try:
        run_result = manager.run(
            X_train, X_test, y_train, y_test,
            task_type=task_type,
            algorithms=req.algorithms,
            k_folds=req.k_folds,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training error: {e}")

    # Build 2D visualisation using PCA on full dataset
    X_full = np.vstack([X_train, X_test])
    y_full = np.concatenate([y_train, y_test])

    n_features = X_full.shape[1]
    if n_features >= 2:
        X_2d = _pca_2d(X_full)
    elif n_features == 1:
        # Pad second dimension with zeros for compatibility
        X_2d = np.hstack([X_full, np.zeros((len(X_full), 1))])
    else:
        X_2d = np.zeros((len(X_full), 2))

    # Decision surface for best algorithm
    best_alg = run_result["best_algorithm"]
    best_hp = run_result["results"][best_alg]["best_hyperparameters"]

    # Retrain best model on 2D PCA projection for surface plot
    surface = None
    try:
        n_train = len(X_train)
        X_2d_train = X_2d[:n_train]
        y_2d_train = y_full[:n_train]
        model_2d = manager._make_model(best_alg, best_hp)
        if task_type == "regression":
            model_2d.fit(X_2d_train, y_2d_train)
        else:
            if best_alg in ("fcm", "pcm"):
                model_2d.fit(X_2d_train)
            else:
                model_2d.fit(X_2d_train, y_2d_train)
        surface = _build_decision_surface(model_2d, X_2d, y_full, task_type, best_alg)
    except Exception:
        surface = None

    # Scatter data for plotting
    scatter_data = {
        "x1": X_2d[:, 0].tolist(),
        "x2": X_2d[:, 1].tolist(),
        "y": y_full.tolist(),
    }

    response = _safe_json({
        "status": "done",
        "task_type": task_type,
        "results": run_result["results"],
        "leaderboard": run_result["leaderboard"],
        "best_algorithm": best_alg,
        "decision_surface": surface,
        "scatter_data": scatter_data,
        "surviving_features": surviving_features,
    })

    store.set_key(req.session_id, "last_result", response)
    return JSONResponse(response)


# ── Results (cached) ──────────────────────────────────────────────────

@app.get("/api/results/{session_id}")
async def get_results(session_id: str):
    result = store.get_key(session_id, "last_result")
    if result is None:
        raise HTTPException(status_code=404, detail="No results yet for this session.")
    return JSONResponse(result)
