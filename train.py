"""Train, compare and save models, and write a Kaggle-style submission file.

Usage:
    python train.py
"""
import pandas as pd

from titanic.data import FEATURES, engineer_features, load_test, load_train
from titanic.model import MODELS_DIR, load_model, train_and_save


def main() -> None:
    train_df = load_train()
    print(f"Training on {len(train_df)} passengers ...")
    m = train_and_save(train_df)

    print("\nCross-validated comparison (training split only):")
    print(pd.DataFrame(m["comparison"]).round(4).to_string(index=False))
    print(f"\nBest model: {m['best_model']}")
    print(f"Hold-out accuracy : {m['test']['accuracy']:.3f}  (ROC-AUC {m['test']['roc_auc']:.3f})")
    base = m["baseline_original_notebook"]
    print(f"Original notebook : {base['accuracy']:.3f}  (ROC-AUC {base['roc_auc']:.3f})")

    test_df = load_test()
    preds = load_model().predict(engineer_features(test_df)[FEATURES])
    sub = pd.DataFrame({"PassengerId": test_df["PassengerId"], "Survived": preds})
    sub.to_csv(MODELS_DIR / "submission.csv", index=False)
    print(f"\nSaved model + metrics + Kaggle submission to {MODELS_DIR}/")


if __name__ == "__main__":
    main()
