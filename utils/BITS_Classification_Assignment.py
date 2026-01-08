
# Utilities for the BITS Classification Assignment.
# Implements six models and evaluation metrics: Accuracy, AUC, Precision, Recall, F1, MCC.
from typing import Dict, Tuple
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
)

try:
    from xgboost import XGBClassifier  # type: ignore
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False


def get_models(random_state: int = 42) -> Dict[str, object]:
    # Return the dict of untrained models required by the assignment.
    models: Dict[str, object] = {
        'Logistic Regression': LogisticRegression(solver='liblinear', random_state=random_state),
        'Decision Tree': DecisionTreeClassifier(random_state=random_state),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'GaussianNB': GaussianNB(),
        'Random Forest': RandomForestClassifier(n_estimators=200, random_state=random_state),
    }
    if _HAS_XGB:
        models['XGBoost'] = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=random_state,
            eval_metric='logloss',
            use_label_encoder=False,
        )
    return models


def train_models(models: Dict[str, object], X_train, y_train) -> Dict[str, object]:
    # Fit each model and return a dict of trained models.
    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model
    return trained


def _predict_proba_or_decision(model, X_test) -> Tuple[np.ndarray, bool]:
    # Return (proba, has_proba). If only decision_function is available, convert to 0-1 range.
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X_test)
        return proba[:, 1], True
    if hasattr(model, 'decision_function'):
        scores = model.decision_function(X_test)
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)
        return scores, False
    return np.array([]), False


def evaluate_models(models: Dict[str, object], X_test, y_test) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    # Evaluate trained models and return (metrics_df, roc_data).
    rows = []
    roc_data: Dict[str, Dict[str, float]] = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        proba, has_proba = _predict_proba_or_decision(model, X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)

        if proba.size > 0:
            auc = roc_auc_score(y_test, proba)
            fpr, tpr, _ = roc_curve(y_test, proba)
            roc_data[name] = {'fpr': fpr, 'tpr': tpr, 'auc': auc}
        else:
            auc = float('nan')

        rows.append({
            'Model': name,
            'Accuracy': acc,
            'AUC': auc,
            'Precision': prec,
            'Recall': rec,
            'F1': f1,
            'MCC': mcc,
        })

    results_df = pd.DataFrame(rows)
    for col in ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']:
        results_df[col] = results_df[col].astype(float).round(4)
    results_df = results_df.sort_values(by='AUC', ascending=False, na_position='last').reset_index(drop=True)
    return results_df, roc_data
