from src.experiments.utils import *
from src.features.feature_engineering_2 import select_features
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def main():
    # Training set
    X_train = pd.read_csv('data/datasets/csv_files/Oversampled_train_dataset.csv')
    y_train = pd.read_csv('data/datasets/csv_files/Oversampled_train_labels.csv')
    # Validation set
    X_val = pd.read_csv('data/datasets/csv_files/Validation_set_features.csv')
    y_val = pd.read_csv('data/datasets/csv_files/Validation_set_labels.csv')

    model = RandomForestClassifier(random_state = 42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]

    exp = create_experiment('RandomForest_kbest_features_SMOTE')
    log_metrics(exp, y_val, y_pred, y_pred_proba)
    log_plots(exp, y_val.values, y_pred_proba)
    log_model(exp, model, 'RandomForest_Kbest_features_oversampled', ["RandomForest", "Model With Oversampled data and features from K-best", "Features: Q4 set"])
    exp.end()

if __name__ == "__main__":
    main()
    