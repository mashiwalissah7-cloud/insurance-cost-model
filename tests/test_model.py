from sklearn.model_selection import train_test_split

from conftest import DATA
from insurance_model.data import BASE_FEATURES, RISK_FEATURES, add_features, load_data, split_features
from insurance_model.evaluate import naive_baseline, score
from insurance_model.model import train_linear_model


def _fit(features):
    df = add_features(load_data(DATA))
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    X_train, y_train = split_features(train, features)
    X_test, y_test = split_features(test, features)
    model = train_linear_model(X_train, y_train)
    return score(y_test, model.predict(X_test)), naive_baseline(y_train, y_test)


def test_base_model_beats_always_guessing_the_average():
    result, naive = _fit(BASE_FEATURES)
    assert result["mae"] < naive["mae"]
    assert result["r2"] > 0.5


def test_risk_features_improve_the_model():
    base, _ = _fit(BASE_FEATURES)
    risk, _ = _fit(RISK_FEATURES)
    assert risk["r2"] > base["r2"]
    assert risk["mae"] < base["mae"]
