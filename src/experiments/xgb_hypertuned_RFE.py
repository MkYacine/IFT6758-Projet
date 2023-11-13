from src.experiments.utils import *
from src.features.feature_engineering_2 import select_features
from sklearn.model_selection import train_test_split
import xgboost as xgb


def main():
    data = pd.read_csv("data/datasets/csv_files/2016-2019-v2.csv")
    X, y = select_features(data)
    X = X[['period', 'game_seconds', 'x_coordinate', 'y_coordinate',
       'distance_to_net', 'shot_angle', 'shot_type', 'last_event_x',
       'last_event_y', 'time_since_last_event', 'distance_from_last_event',
       'rebound', 'angle_change', 'speed', 'powerplay_duration',
       'home_team_players', 'away_team_players',
       'last_event_type_Blocked Shot', 'last_event_type_Faceoff',
       'last_event_type_Game Official', 'last_event_type_Giveaway',
       'last_event_type_Goal', 'last_event_type_Hit',
       'last_event_type_Missed Shot', 'last_event_type_Official Challenge',
       'last_event_type_Penalty', 'last_event_type_Period End',
       'last_event_type_Period Ready', 'last_event_type_Period Start',
       'last_event_type_Shootout Complete', 'last_event_type_Shot',
       'last_event_type_Stoppage', 'last_event_type_Takeaway']]
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

    exp = create_experiment("xgb_hypertuned_RFE")
    log_metrics(exp, y_val, y_pred, y_pred_proba)
    log_plots(exp, y_val.values, y_pred_proba)
    log_model(
        exp,
        model,
        "xgb_hypertuned",
        ["XGBoost", "Model hypertuned on F1", "Features: RFE best features"],
    )
    exp.end()


if __name__ == "__main__":
    main()
