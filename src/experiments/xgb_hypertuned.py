from src.experiments.utils import *
from src.features.feature_engineering_2 import select_features
from sklearn.model_selection import train_test_split
import xgboost as xgb

def main():
    data = pd.read_csv('data/datasets/csv_files/2016-2019-v2.csv')
    X, y = select_features(data)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = xgb.XGBClassifier(subsample= 0.8, n_estimators= 500, min_child_weight= 2, max_depth= 6, learning_rate= 0.05, colsample_bytree= 1)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]

    exp = create_experiment('xgb_hypertuned')
    log_metrics(exp, y_val, y_pred, y_pred_proba)
    log_plots(exp, y_val.values, y_pred_proba)
    log_model(exp, model, 'xgb_hypertuned', ["XGBoost", "Model hypertuned on accuracy", "Features: Q4 set"])
    exp.end()

if __name__ == "__main__":
    main()
    