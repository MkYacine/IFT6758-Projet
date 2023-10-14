import pandas as pd
import numpy as np
import os 
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import re

def adjust_coordinates(df : pd.DataFrame):
    """This function returns a DataFrame where the values of 'x_coordinate' are turned to positive
    and the sign of 'y_coordinate' is changed when 'x_coordinate' is negative"""

    attack_df = df.copy()

    attack_df.loc[df['x_coordinate'] < 0, 'x_coordinate'] = -attack_df['x_coordinate']
    attack_df.loc[df['x_coordinate'] < 0, 'y_coordinate'] = -attack_df['y_coordinate']

    return attack_df

def add_bins(df : pd.DataFrame, bin_size : int):
    """This function adds two columns containing the bins for 'x_coordinate' and 'y_coordinate' """
    attack_df = df.copy()

    attack_df['x_bin'] = pd.cut(attack_df['x_coordinate'], bins=range(0, 100+bin_size, bin_size))
    attack_df['y_bin'] = pd.cut(attack_df['y_coordinate'], bins=range(-45, 45+bin_size, bin_size))

    return attack_df

def get_year_from_gameID(gameID : int):
    """ This function returns the year from a gameID"""
    ID_str = str(gameID)
    
    return re.match(r'^(\d{4})', ID_str).group(1)


def add_year_column(df : pd.DataFrame):
    """This function adds a column containing the year """
    df['year'] = df['gameID'].apply(lambda x : get_year_from_gameID(x))   

    return df 

def get_shot_ratio_for_year(df : pd.DataFrame, year : int):
    """This function returns a DataFrame with the shot ratio for a specific team 
    in a specific year """

    year_df = df[df['year'] == year]

    games_per_team = year_df.groupby('attacking_team_name')['gameID'].nunique()
    shot_ratios = year_df.groupby(['attacking_team_name', 'x_bin', 'y_bin'], observed = False).size().reset_index(name='shots')
    shot_ratios['shots_per_hour'] = shot_ratios['shots'] / shot_ratios['attacking_team_name'].map(games_per_team)

    return shot_ratios

def get_visualization_df(df : pd.DataFrame, team_name : str, year : int):

    shot_ratios  = get_shot_ratio_for_year(df, year)
    league_avgs = shot_ratios.groupby(['x_bin', 'y_bin'], observed = False)['shots_per_hour'].mean().reset_index()

    df_vis = shot_ratios[shot_ratios['attacking_team_name'] == team_name]
   
    df_vis['x_center'] = df_vis['x_bin'].apply(lambda x: (x.left + x.right) / 2)
    df_vis['y_center'] = df_vis['y_bin'].apply(lambda x: (x.left + x.right) / 2)

    df_vis = df_vis.merge(league_avgs, on=['x_bin', 'y_bin'], how='left')
    # Create a new column for the difference between the team's shots_per_hour and the league average
    df_vis['shot_diff'] = df_vis['shots_per_hour_x'] - df_vis['shots_per_hour_y']

    return df_vis
