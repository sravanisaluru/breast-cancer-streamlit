
# Breast Cancer Classification App

This Streamlit app trains six ML models and reports classification metrics and ROC curves. It uses the built-in scikit-learn **Breast Cancer** dataset, so you can run it without uploading data.

## Quick start
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Repository structure
- `app.py` – Streamlit UI and orchestration
- `utils/BITS_Classification_Assignment.py` – Model definitions, training and evaluation utilities
- `utils/preprocessing.py` – Placeholder for future preprocessing steps
- `model/` – Put any saved models here (e.g., `trained_model.pkl`)
- `notebooks/` – Jupyter notebooks for experiments
- `datasets/` – Data files (not used by the default demo)

## Notes
- XGBoost is included to match assignment requirements. If you prefer to omit it, remove it from `requirements.txt` and update `get_models()` accordingly.
- For deployment on Streamlit Community Cloud: ensure `app.py` and `requirements.txt` are in the repo root.
