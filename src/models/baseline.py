"""
Baseline sklearn models (Random Forest, Logistic Regression) and shared
evaluation metrics, used for the Local ML and Centralized ML experiments
(Phase 3) and as the reference upper/lower bounds for the later FedAvg and
Personalized FL experiments.

Random Forest is the primary baseline (matches the SHAP TreeExplainer
planned for Phase 6). Logistic Regression is included as a simpler
reference model, per the locked tech stack.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _single_class_fallback(y_train: pd.Series):
    """
    Some hospitals under extreme Dirichlet skew have zero training examples
    of one class (disease_rate of 0.0 or 1.0). Logistic Regression's solver
    cannot fit on a single class at all, and a Random Forest fit on a single
    class degenerates to always predicting it anyway -- so in both cases we
    use an explicit majority-class DummyClassifier rather than crashing or
    silently producing a misleading "trained" model.
    """
    model = DummyClassifier(strategy="most_frequent")
    model.fit(pd.DataFrame(index=y_train.index), y_train)
    return model


def train_random_forest(
    x_train: pd.DataFrame, y_train: pd.Series, seed: int = 42
) -> RandomForestClassifier:
    if y_train.nunique() < 2:
        return _single_class_fallback(y_train)
    model = RandomForestClassifier(n_estimators=100, random_state=seed)
    model.fit(x_train, y_train)
    return model


def train_logistic_regression(
    x_train: pd.DataFrame, y_train: pd.Series, seed: int = 42
) -> LogisticRegression:
    if y_train.nunique() < 2:
        return _single_class_fallback(y_train)
    model = LogisticRegression(max_iter=1000, random_state=seed)
    model.fit(x_train, y_train)
    return model


MODEL_FACTORIES = {
    "random_forest": train_random_forest,
    "logistic_regression": train_logistic_regression,
}


def evaluate_model(model, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Computes Accuracy, Precision, Recall, F1, and AUC for a fitted binary
    classifier on a held-out test set.

    AUC and precision/recall/F1 are undefined (returned as NaN) when the
    test set contains only one class -- this happens for some hospitals
    under extreme Dirichlet skew and is reported honestly rather than
    hidden, since it reflects a real limitation of evaluating on tiny,
    highly non-IID local test sets.
    """
    n_test = len(y_test)
    n_classes_in_test = y_test.nunique()

    if n_test == 0:
        return {
            "n_test": 0, "accuracy": np.nan, "precision": np.nan,
            "recall": np.nan, "f1": np.nan, "auc": np.nan,
        }

    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    n_classes_in_model = len(getattr(model, "classes_", []))
    if n_classes_in_test < 2 or n_classes_in_model < 2:
        # AUC is undefined either when the test set has only one true class,
        # or when the model was trained on only one class (e.g. a hospital
        # with disease_rate 0.0 or 1.0 under extreme Dirichlet skew) and so
        # cannot produce a probability for the missing class.
        auc = np.nan
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_proba = model.predict_proba(x_test)[:, 1]
            auc = roc_auc_score(y_test, y_proba)

    return {
        "n_test": n_test,
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "auc": round(float(auc), 4) if not np.isnan(auc) else np.nan,
    }
