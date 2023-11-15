from dotenv import load_dotenv
from comet_ml import Experiment
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score
import os
import joblib
from src.features.q3_plots import *


def init():
    load_dotenv()
    try:
        if not os.path.exists("data\models"):
            os.mkdir("data\models")
    except OSError:
        print("Erreur lors de la création du répertoire")


def create_experiment(name: str):
    init()

    experiment = Experiment(
        api_key=os.environ.get("COMET_API_KEY"),
        project_name="ift6758-project",
        workspace="tedoul",
    )

    experiment.set_name(name)

    return experiment


def log_plots(experiment: Experiment, y_val, y_pred_proba):
    experiment.log_figure(
        figure=plot_roc_curve(y_val, y_pred_proba), figure_name="Courbe ROC"
    )
    plt.close()
    experiment.log_figure(
        figure=plot_goal_rate(y_val, y_pred_proba), figure_name="Goal rate"
    )
    plt.close()
    experiment.log_figure(
        figure=plot_cumulative_percent_goal(y_val, y_pred_proba),
        figure_name="Cumulative percent of goal",
    )
    plt.close()
    experiment.log_figure(
        figure=plot_fiability_diagram_resize(y_val, y_pred_proba),
        figure_name="Diagrame de fiabilité",
    )
    plt.close()


def log_metrics(experiment: Experiment, y_val, y_pred, y_pred_proba):
    accuracy = accuracy_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_pred_proba)

    experiment.log_metrics(
        {"accuracy": accuracy, "recall": recall, "f1-score": f1, "auc": auc}
    )
    experiment.log_confusion_matrix(y_val, y_pred)


def log_model(experiment: Experiment, model, name, tags: list):
    joblib.dump(model, f"data/models/{name}.joblib")
    experiment.log_model(name=f"{name}", file_or_folder=f"data/models/{name}.joblib")
    experiment.add_tags(tags)
