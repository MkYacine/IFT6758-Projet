import numpy as np
from src.features.feature_engineering_1 import *
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.calibration import CalibrationDisplay


def plot_roc_curve(y_val: np.ndarray, y_pred_proba: np.ndarray):
    # https://www.statology.org/plot-roc-curve-python/
    """
    Fonction qui plot la courbe ROC et la métrique AUC

    Paramètres:
    y_val (np.ndarray): les vrais labels sur l'ensemble de validation,
    y_pred_proba (np.ndarray): les probabilités de but sur les données de l'ensemble de validation

    Retour:
    La courbe ROC et la métrique AUC
    """

    fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
    auc = roc_auc_score(y_val, y_pred_proba)

    # create ROC curve
    plt.plot(fpr, tpr, label="AUC=" + str(auc))
    plt.plot(
        np.linspace(0, 1), np.linspace(0, 1), c="red", linestyle="--", label="AUC=0.5"
    )
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.title("Courbe ROC")
    plt.grid()
    plt.legend(loc=4)

    fig = plt.gcf()


def plot_goal_rate(y_val: np.ndarray, y_pred_proba: np.ndarray):
    """
    Fonction qui plot le taux de buts (#buts / (#non_buts + #buts))
    comme une fonction du centile de la probabilité de tir

    Paramètres:
    y_val (np.ndarray): les vrais labels sur l'ensemble de validation,
    y_pred_proba (np.ndarray): les probabilités de but sur les données de l'ensemble de validation

    Retour:
    La courbe du taux de buts comme une fonction du centile de la probabilité de tir
    """

    proba_but = y_pred_proba

    percentiles = np.linspace(0, 100, 10)
    percentile_but = np.percentile(proba_but, np.linspace(0, 100, 11))
    taux_buts = []

    for i in range(len(percentiles)):
        percentile_inf = percentile_but[i]
        percentile_sup = percentile_but[i + 1]

        indices = np.where(
            (proba_but >= percentile_inf) & (proba_but <= percentile_sup)
        )

        goals_in_percentile = np.sum(y_val[indices])
        total_shots_percentile = y_val[indices].shape[0]
        taux_buts.append((goals_in_percentile / total_shots_percentile) * 100)

    plt.plot(np.linspace(0, 100, 10), taux_buts)
    plt.ylim(0, 100)
    plt.xlim(110, -10)
    plt.xlabel("Shot Probability model percentile")
    plt.ylabel("Goals / (Goals + Shots)")
    plt.title("Goal Rate")
    plt.grid()

    fig = plt.gcf()


def plot_cumulative_percent_goal(y_val: np.ndarray, y_pred_proba: np.ndarray):
    """
    Fonction qui plot la proportion cumulée de buts
    comme une fonction du centile de la probabilité de tir

    Paramètres:
    y_val (np.ndarray): les vrais labels sur l'ensemble de validation,
    y_pred_proba (np.ndarray): les probabilités de but sur les données de l'ensemble de validation

    Retour:
    La courbe de proportion cumulée de buts comme une fonction du centile de la probabilité de tir
    """

    proba_but = y_pred_proba

    percentiles = np.linspace(0, 100, 10)
    percentile_but = np.percentile(proba_but, np.linspace(0, 100, 10))
    proportions_buts = []

    for i in range(len(percentiles)):
        percentile_i = percentile_but[i]

        indices = np.where(proba_but >= percentile_i)

        goals_in_percentile = np.sum(y_val[indices])

        total_shots_percentile = np.sum(y_val)

        proportions_buts.append((goals_in_percentile / total_shots_percentile) * 100)

    plt.plot(np.linspace(0, 100, 10), proportions_buts)
    plt.ylim(0, 110)
    plt.xlim(110, -10)
    plt.xlabel("Shot Probability model percentile")
    plt.ylabel("Proportion")
    plt.title("Cumulative % of goals")
    plt.grid()

    fig = plt.gcf()


def plot_fiability_diagram(y_val: np.ndarray, y_pred_proba: np.ndarray):
    """
    Fonction qui plot le diagramme de fiabilité du modèle

    Paramètres:
    y_val (np.ndarray): les vrais labels sur l'ensemble de validation,
    y_pred_proba (np.ndarray): les probabilités de but sur les données de l'ensemble de validation

    Retour:
    Fonction qui plot le diagramme de fiabilité du modèle
    """

    proba_but = y_pred_proba

    ax = plt.subplot(2, 2, 4)
    ax.set_title("Diagramme de fiabilité")

    CalibrationDisplay.from_predictions(y_val, proba_but, ax=ax)

    fig = plt.gcf()


def plot_metrics(y_val: np.ndarray, y_pred_proba: np.ndarray):
    """
    Fonction qui plot la courbe ROC, le taux de buts, la proportion cumulée de buts et le diagramme de fiabilité

    Paramètres:
    y_val (np.ndarray): les vrais labels sur l'ensemble de validation,
    y_pred_proba (np.ndarray): les probabilités de but sur les données de l'ensemble de validation

    Retour:
    La courbe ROC, le taux de buts, la proportion cumulée de buts et le diagramme de fiabilité
    """

    # Créez une nouvelle figure avec une grille 2x2 d
    plt.figure(figsize=(12, 8))

    # Créez chaque sous-tracé individuellement
    plt.subplot(2, 2, 1)
    plot_roc_curve(y_val, y_pred_proba)

    plt.subplot(2, 2, 2)
    plot_goal_rate(y_val, y_pred_proba)

    plt.subplot(2, 2, 3)
    plot_cumulative_percent_goal(y_val, y_pred_proba)

    # plt.subplot(2, 2, 4)
    plot_fiability_diagram(y_val, y_pred_proba)

    # Ajustez l'espacement entre les sous-tracés pour éviter le chevauchement
    plt.tight_layout()

    # Affichez la figure
    plt.show()


# only for question 3 on milestone 2


def plot_all_roc_curve(list_y_val: list, list_y_pred_proba: list, models: list):
    """
    Fonction qui plot les courbes ROC et les métriques AUC

    Paramètres:
    list_y_val (list(nd.array)): liste des vrais labels sur l'ensemble de validation,
    list_y_pred_proba (list(nd.array)): liste des probabilités de but sur les données de
    l'ensemble de validation

    Retour:
    Plot descourbes ROC et les métriques AUC
    """
    plt.figure(figsize=(10, 6))

    for i in range(len(models)):
        fpr, tpr, _ = roc_curve(list_y_val[i], list_y_pred_proba[i])
        auc = roc_auc_score(list_y_val[i], list_y_pred_proba[i])

        # create ROC curve
        plt.plot(fpr, tpr, label=f"{models[i]}=" + str(round(auc, 5)))

    plt.plot(
        np.linspace(0, 1), np.linspace(0, 1), c="red", linestyle="--", label="AUC=0.5"
    )
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.title("Courbe ROC")
    plt.grid()
    plt.legend(loc=2)


def plot_all_goal_rate(list_y_val: list, list_y_pred_proba: list, models: list):
    """
    Fonction qui plot les taux de buts (#buts / (#non_buts + #buts))
    comme une fonction du centile de la probabilité de tir des modèles

    Paramètres:
    list_y_val (list(nd.array)): liste des vrais labels sur l'ensemble de validation,
    list_y_pred_proba (list(nd.array)): liste des probabilités de but sur les données de
    l'ensemble de validation

    Retour:
    Plot des taux de buts (#buts / (#non_buts + #buts))
    comme une fonction du centile de la probabilité de tir des modèles
    """

    plt.figure(figsize=(10, 6))

    for i in range(len(models)):
        proba_but = list_y_pred_proba[i]
        y_val = list_y_val[i]
        model = models[i]

        percentiles = np.linspace(0, 100, 10)
        percentile_but = np.percentile(proba_but, np.linspace(0, 100, 11))
        taux_buts = []

        for i in range(len(percentiles)):
            percentile_inf = percentile_but[i]
            percentile_sup = percentile_but[i + 1]

            indices = np.where(
                (proba_but >= percentile_inf) & (proba_but <= percentile_sup)
            )

            goals_in_percentile = np.sum(y_val[indices])
            total_shots_percentile = y_val[indices].shape[0]
            taux_buts.append((goals_in_percentile / total_shots_percentile) * 100)

        plt.plot(np.linspace(0, 100, 10), taux_buts, label=f"{model}")

    plt.ylim(0, 100)
    plt.xlim(110, -10)
    plt.xlabel("Shot Probability model percentile")
    plt.ylabel("Goals / (Goals + Shots)")
    plt.title("Goal Rate")
    plt.legend()
    plt.grid()


def plot_all_cumulative_percent_goal(
    list_y_val: list, list_y_pred_proba: list, models: list
):
    """
    Fonction qui plot les proportions cumulées de buts
    comme une fonction du centile de la probabilité de tir des modèles

    Paramètres:
    list_y_val (list(nd.array)): liste des vrais labels sur l'ensemble de validation,
    list_y_pred_proba (list(nd.array)): liste des probabilités de but sur les données de
    l'ensemble de validation

    Retour:
    Plot des proportions cumulées de buts
    comme une fonction du centile de la probabilité de tir des modèles
    """
    plt.figure(figsize=(10, 6))

    for i in range(len(models)):
        proba_but = list_y_pred_proba[i]
        y_val = list_y_val[i]
        model = models[i]

        percentiles = np.linspace(0, 100, 10)
        percentile_but = np.percentile(proba_but, np.linspace(0, 100, 10))
        proportions_buts = []

        for i in range(len(percentiles)):
            percentile_i = percentile_but[i]

            indices = np.where(proba_but >= percentile_i)

            goals_in_percentile = np.sum(y_val[indices])

            total_shots_percentile = np.sum(y_val)

            proportions_buts.append(
                (goals_in_percentile / total_shots_percentile) * 100
            )

        plt.plot(np.linspace(0, 100, 10), proportions_buts, label=f"{model}")
    plt.ylim(0, 110)
    plt.xlim(110, -10)
    plt.xlabel("Shot Probability model percentile")
    plt.ylabel("Proportion")
    plt.title("Cumulative % of goals")
    plt.legend()
    plt.grid()


def plot_all_fiability_diagram(list_y_val: list, list_y_pred_proba: list, models: list):
    """
    Fonction qui plot les diagrammes de fiabilité des modèles

    Paramètres:
    list_y_val (list(nd.array)): liste des vrais labels sur l'ensemble de validation,
    list_y_pred_proba (list(nd.array)): liste des probabilités de but sur les données de
    l'ensemble de validation

    Retour:
    Plot des diagramme de fiabilité des modèles
    """
    plt.figure(figsize=(10, 6))

    for i in range(len(models)):
        proba_but = list_y_pred_proba[i]
        y_val = list_y_val[i]
        model = models[i]

        ax = plt.subplot(1, 1, 1)
        ax.set_title("Diagramme de fiabilité")

        CalibrationDisplay.from_predictions(y_val, proba_but, ax=ax, name=f"{model}")

    plt.legend(loc=2)


def plot_fiability_diagram_resize(y_val: np.ndarray, y_pred_proba: np.ndarray):
    """
    Fonction qui plot le diagramme de fiabilité du modèle

    Paramètres:
    y_val (np.ndarray): les vrais labels sur l'ensemble de validation,
    y_pred_proba (np.ndarray): les probabilités de but sur les données de l'ensemble de validation

    Retour:
    Fonction qui plot le diagramme de fiabilité du modèle
    """

    proba_but = y_pred_proba

    ax = plt.subplot(1, 1, 1)
    ax.set_title("Diagramme de fiabilité")

    CalibrationDisplay.from_predictions(y_val, proba_but, ax=ax)

    fig = plt.gcf()
