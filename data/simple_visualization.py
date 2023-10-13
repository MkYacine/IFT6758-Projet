import pandas as pd
import numpy as np 

from clean import clean_json
from scrape import scrape_multiple_years
import os
import matplotlib.pyplot as plt
import ipywidgets as widgets


def get_shot_repartition_NHL(df : pd.DataFrame): 
    """This function returns the shot and goal count 
    for each type of shot in the whole league """
    shot_df = df[df['play_type'] == 'Shot']
    goal_df = df[df['play_type'] == 'Goal']

    shot_count = shot_df['shot_type'].value_counts()
    goal_count = goal_df['shot_type'].value_counts()

    return [shot_count, goal_count]


def plot_shot_repartition_NHL(df:pd.DataFrame, year: int):
    """This function plots the shot repartition in the whole NHL """

    shot_count, goal_count = get_shot_repartition_NHL(df)

    ## Compute the efficacy for each type of shot
    fig, ax = plt.subplots(figsize = (20,10))
    ax.grid()

    ax.bar(shot_count.index, shot_count.values, label ='Shot Count')
    ax.bar(goal_count.index, goal_count.values, color = 'green', label = 'Goal Count')
    
    ax.set_xlabel('Shot Type')
    ax.set_ylabel('Size')
    ax.set_title(f"Shot repartition in the NHL in {year}")

    ax.legend(fontsize='x-large')

    plt.show()

def pie_shot_repartition_NHL(df : pd.DataFrame, year : int):
    """This function plots a pie showing the shot repartition in the NHL"""

    # Une autre forme de visualisation (Meilleure pour voir la répartition des tirs)
    # Source : https://matplotlib.org/3.1.1/gallery/pie_and_polar_charts/pie_features.html#sphx-glr-gallery-pie-and-polar-charts-pie-features-py

    shot_count = get_shot_repartition_NHL(df)[0]

    fig, ax = plt.subplots(figsize=(12,7))

    ax.pie(shot_count.values, radius=1,
    labels = shot_count.index, autopct='%.0f%%')

    ax.set(aspect="equal", title=f"Shot repartition in the NHL in {year}")
    plt.legend(shot_count.index, loc = "upper right")
    plt.show()

def hist_shot_repartition_by_team(team_name : str, df : pd.DataFrame, year : int): 
    """"
    This function returns a histogram showing the count of each shot type
    for a specific team
    """
    team_list = df['attacking_team_name'].unique()
    if not team_name in team_list :
        print(f"The team doesn't exist")
        return None
    
    team_shot_count = df[(df['play_type'] == 'Shot') & (df['attacking_team_name'] == team_name)]
    team_goal_count = df[(df['play_type'] == 'Goal') & (df['attacking_team_name'] == team_name)]

    shot_count = team_shot_count['shot_type'].value_counts()
    goal_count = team_goal_count['shot_type'].value_counts()

    plt.figure(figsize = (20,10))
    plt.grid()
    plt.bar(shot_count.index, shot_count.values, alpha = 0.9, label ='Shot Count')
    plt.bar(goal_count.index, goal_count.values, color = 'green', label = 'Goal count')

    plt.ylim(0,2000)
    plt.xlabel('Shot Type')
    plt.ylabel('Size')
    plt.title(f"Shot repartition for {team_name} in {year}")

    plt.legend(fontsize='x-large')

    plt.show()


def find_net(coordinate_x : int):
    """This function returns the net where the attacking team should put the hockey puck"""
    if coordinate_x <= 0 :
        return 'left'
    else :
        return 'right'
    
def compute_distance_to_net(coordinate_x : float, coordinate_y : float):
    """This function determines the net where the attacking team shoots during the play
      and then computes the distance to the net"""
    net_side = find_net(coordinate_x)
    # Cas où le filet sur lequel l'équipe tire se trouve sur la droite
    if net_side == 'right' :
        dist = np.sqrt((coordinate_x - 89)**2 + coordinate_y**2)
    else : 
        # Cas où le filet sur lequel l'équipe tire se trouve sur la gauche
        dist = np.sqrt((coordinate_x + 89)**2 + coordinate_y**2)
    return dist

def df_add_distance_to_net(df : pd.DataFrame):
    """This function adds a column containing the distance to the net 
    to the DataFrame"""

    df['distance_to_net'] = df.apply(lambda row : compute_distance_to_net(row['x_coordinate'],
                                                row['y_coordinate']),axis = 1) 
    
    return df   

def regroup_by_range(df : pd.DataFrame, bins : list, bin_centers : list): 
    """This function is inspired of the 3rd homework.
    It groups the shots by interval of distance
    """
    df = df.copy()

    # We add a new column to the DataFrame containing the bin associated to the distance
    df['distance_interval'] = pd.cut(df['distance_to_net'], bins = bins, labels = bin_centers)
    hist = df.groupby(['distance_interval','play_type'], observed = False).size().reset_index(name = 'count')

    return hist

def get_efficiency_rate_by_distance(df : pd.DataFrame, bins : list, bin_centers : list) : 
    """This function returns the efficaciency rate for shots of a certain range"""

    counts = regroup_by_range(df,bins, bin_centers)
    
    shot_count = counts[counts['play_type'] == 'Shot']
    goal_count = counts[counts['play_type'] == 'Goal']

    efficiency_table = pd.DataFrame(goal_count['count'].values/(shot_count['count'].values + goal_count['count'].values))
    efficiency_table.fillna(0, inplace = True)

    return efficiency_table*100

def plot_efficiency_NHL(df: pd.DataFrame, bins : list, bin_centers : list, year : int):
    """This function plots the efficiency rate by distance in the whole league"""
    efficacy_rate = get_efficiency_rate_by_distance(df, bins, bin_centers)

    plt.grid()
    plt.ylim(0,100)
    plt.plot(bin_centers,efficacy_rate)
    plt.xlabel('Distance from the net (in ft)')
    plt.ylabel('Efficiency rate')
    plt.title(f"Efficiency rate in function of the distance from the net in {year}/{year+1}")
    plt.legend(['Efficiency curve'])

    plt.show()


def efficiency_rate_by_shot_type(shot_type : str, df : pd.DataFrame, bins : list, bin_centers : list):
    """This function returns the efficiency rate for a particular shot type in 
    the whole league """

    shot_type_list = df['shot_type'].unique()

    if not shot_type in shot_type_list : 
        print(f"Ce type de tir n'existe pas")
        return None
    
    df_bis = df[df['shot_type'] == shot_type]
    counts = regroup_by_range(df_bis, bins, bin_centers)

    goal_count = counts[counts['play_type'] == 'Goal']
    shot_count = counts[counts['play_type'] == 'Shot']
    
    total_attempts = goal_count['count'].values + shot_count['count'].values

    efficiency = goal_count['count']/total_attempts
    efficiency.fillna(0,inplace = True)

    return efficiency

def plot_efficiency_rate_by_shot_type(shot_type : str, df : pd.DataFrame, bins: list, bin_centers : list, year :int):
    """This function plots the efficiency_rate for a certain shot type in the whole league"""
    efficiency_table = efficiency_rate_by_shot_type(shot_type, df, bins, bin_centers)

    plt.grid()
    plt.ylim(0,105)
    plt.plot(bin_centers,100*efficiency_table)
    plt.xlabel('Distance from net(in ft)')
    plt.ylabel('Efficiency rate')
    plt.title(f"Efficiency rate in function of the distance from the net for {shot_type} in {year}/{year+1}")
    plt.legend(['Efficiency curve'])

    plt.show()

def plot_efficiency_rate_for_all_shots(df : pd.DataFrame, bins : list, bin_centers : list, year : int):
    shot_type_list = df['shot_type'].unique()[:-1]

    fig,ax = plt.subplots()
    plt.grid()
    for shot_type in shot_type_list : 
        efficiency = efficiency_rate_by_shot_type(shot_type, df, bins, bin_centers)
        ax.plot(bin_centers, efficiency, label = shot_type)

    plt.xlabel("Distance from the net(in ft)")
    plt.ylabel("Efficiency")
    plt.title(f"Efficiency in function of distance from the net for the different shot types in {year}/{year+1}")
    plt.legend(shot_type_list)
