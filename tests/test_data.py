import pandas as pd
import pytest

from conftest import DATA
from insurance_model.data import RISK_FEATURES, add_features, load_data


def test_load_data_has_no_missing_values():
    df = load_data(DATA)
    assert len(df) > 1000
    assert not df.isna().any().any()


def test_load_data_rejects_missing_columns(tmp_path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"age": [30], "bmi": [25.0]}).to_csv(bad, index=False)
    with pytest.raises(ValueError):
        load_data(bad)


def test_add_features_encodes_text_as_numbers():
    df = pd.DataFrame(
        {"age": [40, 25], "sex": ["male", "female"], "bmi": [32.0, 22.0], "children": [1, 0],
         "smoker": ["yes", "no"], "region": ["southeast", "northeast"], "charges": [40000.0, 3000.0]}
    )
    out = add_features(df)
    assert out["is_smoker"].tolist() == [1, 0]
    assert out["is_male"].tolist() == [1, 0]
    assert out["smoker_obese"].tolist() == [1, 0]  # smoker with BMI 32 counts, non-smoker does not
    assert out["region_southeast"].tolist() == [1, 0]
    assert out["age_squared"].tolist() == [1600, 625]
    for feature in RISK_FEATURES:
        assert feature in out.columns
