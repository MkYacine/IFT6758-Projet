from src.experiments.utils import *
from src.features.feature_engineering_2 import select_features
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import src.features.feature_selection as fsel


def main():
    input_path = os.path.join("data\datasets\csv_files", "2016-2019-v2.csv")
    df = pd.read_csv(input_path)

    df = fsel.remove_nan_from_strength(df)

    categorical_columns_1 = [
        "period_type",
        "attacking_team_name",
        "shooter",
        "goalie",
        "rebound",
        "last_event_type",
        "home_team",
    ]
    shot_type_classified = [
        ["Wrap-around", 0],
        ["Slap Shot", 1],
        ["Snap Shot", 2],
        ["Wrist Shot", 3],
        ["Backhand", 4],
        ["Deflected", 5],
        ["Tip-In", 6],
    ]
    strength_classified = [["Short Handed", 0], ["Even", 1], ["Power Play", 2]]

    df = df.dropna()
    df = fsel.encode_categorical_features(
        df, categorical_columns_1, shot_type_classified, strength_classified
    )

    X = df.drop(columns=["is_goal", "period_time"])
    Y = df["is_goal"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    model = MLPClassifier(max_iter=1000, hidden_layer_sizes=(100, 50), alpha=0.0001)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]

    exp = create_experiment("mlp_all_features")
    log_metrics(exp, y_val, y_pred, y_pred_proba)
    log_plots(exp, y_val.values, y_pred_proba)
    log_model(
        exp,
        model,
        "mlp_all_features",
        ["MLPClassifier", "Features: All features"],
    )
    exp.end()


if __name__ == "__main__":
    main()
