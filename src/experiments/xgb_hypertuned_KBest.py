from src.experiments.utils import *
from src.features.feature_engineering_2 import select_features
from sklearn.model_selection import train_test_split
import xgboost as xgb


def main():
    data = pd.read_csv("data/datasets/csv_files/KBest_2016-2019.csv")
    X, y = data.drop(['is_goal'], axis=1), data['is_goal']
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = xgb.XGBClassifier(
        subsample=0.6,
        n_estimators=200,
        min_child_weight=5,
        max_depth=15,
        learning_rate=0.2,
        colsample_bytree=0.5,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]

    exp = create_experiment("xgb_hypertuned_KBest")
    log_metrics(exp, y_val, y_pred, y_pred_proba)
    log_plots(exp, y_val.values, y_pred_proba)
    log_model(
        exp,
        model,
        "xgb_hypertuned_KBest",
        ["XGBoost", "Model hypertuned on F1 score", "Features: KBest 10 features"],
    )
    exp.end()


if __name__ == "__main__":
    main()
