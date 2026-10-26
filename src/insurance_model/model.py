"""Train the regression models."""

from sklearn.linear_model import LinearRegression


def train_linear_model(X_train, y_train):
    """Fit a linear regression: predicted charges = a weighted sum of the features."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def coefficient_table(model, feature_names):
    """List how many dollars each feature adds to the prediction, largest first."""
    pairs = list(zip(feature_names, model.coef_))
    return sorted(pairs, key=lambda pair: abs(pair[1]), reverse=True)
