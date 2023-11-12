from src.experiments.utils import *
from src.features.feature_engineering_2 import select_features
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def main():
    data = pd.read_csv("../../data/datasets/csv_files/2016-2019-v2.csv")
    df1 = data[["distance_to_net", "shot_angle", "is_goal"]]
    df1 = df1.dropna()

    X, y = df1[["distance_to_net", "shot_angle"]].values, df1["is_goal"].values

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]

    exp = create_experiment("logit_dist_angle")
    log_metrics(exp, y_val, y_pred, y_pred_proba)
    log_plots(exp, y_val, y_pred_proba)
    log_model(
        exp,
        model,
        "logit_dist_angle",
        ["LogisticRegression", "Features: distance_to_net, shot_angle"],
    )
    exp.end()


if __name__ == "__main__":
    main()
