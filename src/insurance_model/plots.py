"""Charts that show what the model learned."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files, no window needed
import matplotlib.pyplot as plt


def actual_vs_predicted(y_true, y_pred, title, path):
    """Points on the diagonal line are perfect predictions."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.5, s=14)
    low, high = float(min(y_true.min(), y_pred.min())), float(max(y_true.max(), y_pred.max()))
    ax.plot([low, high], [low, high], linestyle="--", linewidth=1)
    ax.set_xlabel("Actual charges ($)")
    ax.set_ylabel("Predicted charges ($)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(Path(path), dpi=150)
    plt.close(fig)


def charges_by_bmi_and_smoking(df, path):
    """Shows why the smoker and obesity feature matters."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, group in [("Non-smoker", df[df["is_smoker"] == 0]), ("Smoker", df[df["is_smoker"] == 1])]:
        ax.scatter(group["bmi"], group["charges"], alpha=0.5, s=14, label=label)
    ax.axvline(30, linestyle="--", linewidth=1)
    ax.text(30.5, ax.get_ylim()[1] * 0.95, "BMI 30 (obesity)", va="top")
    ax.set_xlabel("BMI")
    ax.set_ylabel("Annual medical charges ($)")
    ax.set_title("Charges by BMI and smoking status")
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(path), dpi=150)
    plt.close(fig)
