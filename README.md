# FirstPrinciple ML

**FirstPrinciple ML** is a full-stack exploration computing platform designed to bridge the gap between high-level AI libraries and fundamental mathematical optimization. It operates essentially as a transparent ML playground where algorithms are written, processed, and optimized from the foundational principles of Calculus, Linear Algebra, and NumPy matrix multiplication.

---

## 🏗️ Architecture & Codebase Explanation

The project is structured efficiently into a separated modern `frontend/` and a Python-powered `backend/`.

### 1. `backend/main.py`
This is the core FastAPI routing entry point. It oversees:
- **Session initialization**: Connects REST endpoints to global session tracking objects.
- **`/api/upload/...`**: Receives CSV, JSON, or fetch URLs, immediately requesting strict 65:35 Train-Test split and Null checking.
- **`/api/preprocess`**: Bridges UI configurations to backend data processing (Encoding, Scaling, Pruning).
- **`/api/train`**: Hands off prepared data matrices to the mathematical framework. It orchestrates Grid Searches, coordinates PCA dimensionality reduction for the frontend, and packages final statistical outputs.

### 2. `backend/processor.py`
Responsible for Data Sanity & Transformations. Crucially, almost all functionalities here restrict measurement fitting to the `X_train` dataset, mapping purely transformed results to `X_test` in order to safeguard against data leakage.
- `split_raw()`: Implements RNG permutations for initial indexing.
- `validate_and_impute()`: Examines columns for >5% thresholds, utilizing Train-set mean/mode approximations for missing cells.
- `encode_categoricals()`: Creates structured One-Hot grids or integer mapping layouts.
- `scale_features()`: Evaluates internal constants `μ` and `σ` for respective standardization.
- `prune_correlated()`: Computes upper-triangle Pearson coefficients, purging attributes demonstrating excessive multi-collinearity (`r > 0.9`).

### 3. `backend/model_manager.py`
The absolute computational heart of the platform.
- **Custom Algorithms (`LinearRegressionGD`, `SoftSVMRegressor`, `KNNClassifier`, etc.)**: Manually written mathematical formulations. For instance, the Gradients are computed absolutely via properties like `(1/m) * (X_b.T @ error)`. Mathematical definitions (such as Euclidean distances or Fuzzy Membership probabilities) reside functionally within these classes without external library aid.
- `KFoldCV`: An organic implementation of cross-validation slicing.
- `ModelManager`: Exposes the central `.run()` controller. It synthesizes Cartesian Parameter Grids out of client requests, runs successive Model instantiations through `KFoldCV.split`, catalogs internal iteration scores, deduces "Best" structural metrics, and runs final test evaluations.

### 4. `backend/ingestor.py`
A lightweight ingestion wrapper mapping CSV strings, JSON streams, and web URLs to initial generic Pandas DataFrames.

### 5. `backend/session_store.py`
Because DataFrames and NumPy tensors cannot be optimally passed back and forth to HTTP stateless clients every fraction of a second, `session_store` acts as an ephemeral in-memory dictionary. Clients retain a UUID string and the backend securely maps this UUID to the precise datasets and transformations under management.

---

### 6. `frontend/` (UI Layer)
The UI comprises a pure HTML/CSS/VanillaJS stack utilizing *Plotly.js* for high-performance canvas rendering.
- `index.html`: Contains the structural skeleton, separating steps into sequential `panel` classes with fluid animation transitions.
- `style.css`: Configured around a polished, bright, glassmorphism-inspired "Clean Science" aesthetic. Features intense layout grids, distinct step trackers, and responsive component wrapping.
- `app.js`: Coordinates the entire User Experience sequence.
  - Sends configuration choices to FastAPI endpoints.
  - Iterates dynamically rendered hyperparameter fields to support arrays of values.
  - Parses statistical returns into HTML Leaderboards (`renderLeaderboard`).
  - Orchestrates interactive Plotly graphs, translating JSON payload matrices into vivid scatter distributions, 2D PCA regression surfaces, and algorithmic training loss descents.
