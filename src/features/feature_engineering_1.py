import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

import src.visualization.simple_visualization as sv


def get_all_season_data(years : list):
    df = pd.DataFrame()

    for year in years :
        path = os.path.join('../data/datasets/csv_files/', f'{year}.csv')
        aux_df = pd.read_csv(path)
        df = pd.concat([df,aux_df], ignore_index = True)
    
    return df

def compute_shot_angle(coordinate_x : float, coordinate_y : float): 
    # If the shot is made on the right side
    if coordinate_x >= 0 :
        x_net = 89.
        if coordinate_x == 89. : # Shot perpendicular to the net
            return 90
    
        x_dist_abs = np.abs(coordinate_x - x_net)
        if coordinate_x > 89. : # Shot behind the net
            return 90 + np.rad2deg(np.arctan(coordinate_y/x_dist_abs))
        
        return np.rad2deg(np.arctan(coordinate_y/x_dist_abs))
    
    else: # If the shot is made on the left side
        x_net = -89.
        if coordinate_x == -89 : # Shot perpendicular to the net
            return 90
        
        x_dist_abs = np.abs(coordinate_x - x_net)
        if coordinate_x < -89. : # Shot behind the net
            return 90 + np.rad2deg(np.arctan(coordinate_y/x_dist_abs))
        
        return np.rad2deg(np.arctan(coordinate_y/x_dist_abs))
    

def add_new_features(df : pd.DataFrame):
    # Ajout de la colonne distance au DataFrame
    df['distance_to_net'] = df.apply(lambda x : sv.compute_distance_to_net(x['x_coordinate'],x['y_coordinate']), axis=1)
    df['distance_to_net'] = df['distance_to_net'].apply(lambda x : round(x,0))

    # Ajout de la colonne 'shot_angle'
    df['shot_angle'] = df.apply(lambda x : compute_shot_angle(x['x_coordinate'], x['y_coordinate']), axis = 1)
    df['shot_angle'] = df['shot_angle'].apply(lambda x : round(x,0))

    # Ajout de la colonne is_goal
    df['is_goal'] = df['play_type'].apply(lambda x : 1 if x == 'Goal' else 0)

    # Ajout de la colonne is_empty_net
    df['is_empty_net'] = df['empty_net'].apply(lambda x : 1 if x == True else 0)

    return df

def plot_efficiency_curve(df : pd.DataFrame, feature_name : str):

    if feature_name not in ['distance_to_net', 'shot_angle']:
        print(f"feature_name doit nécessairement être 'distance_to_net' ou 'shot_angle' ")
        return 
    
    efficiency_df = df.groupby([feature_name])['is_goal'].mean().to_frame().reset_index()
    efficiency_df = efficiency_df.rename(columns = {'is_goal': 'efficiency'})
    efficiency_df['efficiency'] = efficiency_df['efficiency'].apply(lambda x : 100*x)

    sns.lineplot(data = efficiency_df, x = f'{feature_name}', y = 'efficiency')
    plt.title(f"Efficacité en fonction de {feature_name}")
    plt.ylabel("Efficacité (en %)")
    plt.show()
    

def hist_for_goals(df : pd.DataFrame):
    # On récupère les évènements de type 'Goal' uniquement
    goal_df = df[df['play_type'] == 'Goal']

    # Affichage des histogrammes
    fig, axes = plt.subplots(2,1, sharex = True, figsize = (10,8))

    sns.histplot(goal_df[goal_df['is_empty_net'] == 1], x = 'distance_to_net', bins = 50, ax = axes[0])
    axes[0].set_title("Empty net")

    sns.histplot(goal_df[goal_df['is_empty_net'] == 0], x = 'distance_to_net', bins = 50, ax = axes[1])
    axes[1].set_title("Non empty net")
    
    plt.suptitle('Histogramme des buts en fonction de la distance de tir')

