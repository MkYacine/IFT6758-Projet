from src.experiments.utils import *
from sklearn.model_selection import train_test_split
import xgboost as xgb


def main():
    data = pd.read_csv("data/datasets/csv_files/2016-2019-v2.csv")
    X, y = data[['distance_to_net', 'shot_angle']], data['is_goal']
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]

    exp = create_experiment("xgb_base")
    log_metrics(exp, y_val, y_pred, y_pred_proba)
    log_plots(exp, y_val.values, y_pred_proba)
    log_model(
        exp,
        model,
        "xgb_base",
        ["XGBoost", "Basic Model, no hypertuning", "Features: shot angle, shot distance"],
    )
    exp.end()


if __name__ == "__main__":
    main()
