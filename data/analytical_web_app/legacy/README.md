# Legacy MVP

The original six-tab Streamlit app, kept runnable for reference and superseded by
Groundtruth in the parent directory.

    streamlit run legacy/app_mvp.py
    python -m pytest legacy/test_core_mvp.py

| File | Superseded by |
|---|---|
| `app_mvp.py` | `../app.py` |
| `analytics_core.py` | `groundtruth/{connectors,semantic,stats,report}.py` |
| `ml_core.py` | `groundtruth/ml.py` |
| `test_core_mvp.py` | `../tests/` |

Kept because it still runs and documents where the project started. It carries the
limitations Groundtruth was built to fix: in-memory pandas only, AND-only filters,
a stateless alert worker, impurity-based feature importances, and an AI tab that
estimates from 30 sample rows instead of computing.
