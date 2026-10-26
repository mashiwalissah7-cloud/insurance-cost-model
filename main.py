"""Train and compare two models that predict medical insurance charges.

Run:  python main.py
Options: --data path/to/insurance.csv  --out outputs  --seed 42
"""

import argparse
import json
import sys
from pathlib import Path

from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent / "src"))

from insurance_model.data import BASE_FEATURES, RISK_FEATURES, add_features, load_data, split_features  # noqa: E402
from insurance_model.evaluate import naive_baseline, score  # noqa: E402
from insurance_model.model import coefficient_table, train_linear_model  # noqa: E402
from insurance_model.plots import actual_vs_predicted, charges_by_bmi_and_smoking  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Predict medical insurance charges.")
    parser.add_argument("--data", default="data/insurance.csv")
    parser.add_argument("--out", default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = add_features(load_data(args.data))
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=args.seed)
    print(f"Rows: {len(df)} total, {len(train_df)} for training, {len(test_df)} for testing\n")

    results = {"naive_average": naive_baseline(train_df["charges"], test_df["charges"])}
    models = {}
    for name, features in [("base_model", BASE_FEATURES), ("risk_model", RISK_FEATURES)]:
        X_train, y_train = split_features(train_df, features)
        X_test, y_test = split_features(test_df, features)
        model = train_linear_model(X_train, y_train)
        predictions = model.predict(X_test)
        results[name] = score(y_test, predictions)
        models[name] = (model, features, predictions)

    print(f"{'Model':<16}{'R2':>8}{'MAE ($)':>12}{'RMSE ($)':>12}")
    for name, s in results.items():
        print(f"{name:<16}{s['r2']:>8.3f}{s['mae']:>12,.0f}{s['rmse']:>12,.0f}")

    risk_model, risk_features, risk_pred = models["risk_model"]
    print("\nWhat the risk model learned (dollars added per unit of each feature):")
    for feature, coef in coefficient_table(risk_model, risk_features):
        print(f"  {feature:<18}{coef:>12,.0f}")

    actual_vs_predicted(test_df["charges"], models["base_model"][2], "Base model: actual vs predicted", out / "base_actual_vs_predicted.png")
    actual_vs_predicted(test_df["charges"], risk_pred, "Risk model: actual vs predicted", out / "risk_actual_vs_predicted.png")
    charges_by_bmi_and_smoking(df, out / "charges_by_bmi_and_smoking.png")
    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\nCharts and results saved to {out}/")


if __name__ == "__main__":
    main()
