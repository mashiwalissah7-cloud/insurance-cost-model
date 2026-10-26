"""Measure how good the predictions are."""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def score(y_true, y_pred):
    """Return the three numbers used to compare models.

    r2: share of the variation in charges the model explains (1.0 is perfect).
    mae: average size of a prediction error, in dollars.
    rmse: like mae, but punishes large misses more.
    """
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def naive_baseline(y_train, y_test):
    """Score a model that always predicts the average charge.

    Any real model has to beat this to be worth using.
    """
    guess = np.full(len(y_test), float(np.mean(y_train)))
    return score(y_test, guess)
