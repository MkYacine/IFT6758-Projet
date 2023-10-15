import pandas as pd
import numpy as np
import os 
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import re
import plotly.graph_objects as go
from PIL import Image
from scipy.ndimage import gaussian_filter
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

def adjust_coordinates(df : pd.DataFrame):
    """This function returns a DataFrame where the values of 'x_coordinate' are turned to positive
    and the sign of 'y_coordinate' is changed when 'x_coordinate' is negative"""

    df.loc[df['x_coordinate'] < 0, 'x_coordinate'] = -df['x_coordinate']
    df.loc[df['x_coordinate'] < 0, 'y_coordinate'] = -df['y_coordinate']
    return df

def add_bins(df : pd.DataFrame, bin_size : int):
    """This function adds two columns containing the bins for 'x_coordinate' and 'y_coordinate' """
    df['x_bin'] = pd.cut(df['x_coordinate'], bins=range(0, 100+bin_size, bin_size))
    df['y_bin'] = pd.cut(df['y_coordinate'], bins=range(-45, 45+bin_size, bin_size))

    return df

def get_shot_ratio(df : pd.DataFrame):
    """This function returns a DataFrame with the shot ratio for a specific team 
    in a specific year """
    
    games_per_team = df.groupby('attacking_team_name')['gameID'].nunique()
    shot_ratios = df.groupby(['attacking_team_name', 'x_bin', 'y_bin'], observed = False).size().reset_index(name='shots')
    shot_ratios['shots_per_hour'] = shot_ratios['shots'] / shot_ratios['attacking_team_name'].map(games_per_team)
    league_avgs = shot_ratios.groupby(['x_bin', 'y_bin'], observed = False)['shots_per_hour'].mean().reset_index()
    df = shot_ratios.merge(league_avgs, on=['x_bin', 'y_bin'], how='left')
    
    # Create a new column for the difference between the team's shots_per_hour and the league average
    df['shot_diff'] = df['shots_per_hour_x'] - df['shots_per_hour_y']

    # Create columns for the center of each bin (useful for plotting)
    df['x_center'] = df['x_bin'].apply(lambda x: (x.left + x.right) / 2)
    df['y_center'] = df['y_bin'].apply(lambda x: (x.left + x.right) / 2)
    return df

def prepare_data(df: pd.DataFrame, bin_size=5):
    df = adjust_coordinates(df)
    df =add_bins(df, bin_size)
    df = get_shot_ratio(df)
    return df

def get_figure(df: pd.DataFrame, year):
    fig = go.Figure()
    pyLogo = Image.open("img/attack_zone.png")
    # Ref: https://stackoverflow.com/questions/66150459/how-to-add-a-local-image-svg-png-to-plotly-layout
    fig.add_layout_image(
        dict(
            source=pyLogo,
            xref="x",
            yref="y",
            x=-45,
            y=100,
            sizex=90,
            sizey=100,
            sizing="stretch",
            opacity=1,
            layer="below"
        )
    )
    
    teams = df['attacking_team_name'].unique()
    
    # Ref: https://stackoverflow.com/questions/58867219/how-to-change-plots-of-several-datasets-with-plotly-button
    # Add contour plots for each team
    for team in teams:
    
        # Data interpolation
        team_data = df[df['attacking_team_name'] == team]
        [x,y] = np.round(np.meshgrid(np.linspace(-45,45,90), np.linspace(0,100,100)))
        grid = griddata((team_data['y_center'][::-1],team_data['x_center']),team_data['shot_diff'],(x,y),method='linear',fill_value=0)
        smoothed_data = gaussian_filter(grid,sigma = 3)
        
        data_min= smoothed_data.min()
        data_max= smoothed_data.max()
    
        if abs(data_min) > data_max:
            data_max = data_min * -1
        elif data_max > abs(data_min):
            data_min = data_max * -1
        
        visible = (team == teams[0])  # Only the first trace is visible
        
        fig.add_trace(
            go.Contour(
                z=smoothed_data,
                x=np.linspace(-45, 45, 90),
                y=np.linspace(0, 100, 100),
                colorscale='RdBu_r',
                zmin=data_min,
                zmax=data_max,
                opacity=0.5,
                visible=visible
            ))
    
    
    # Define dropdown menu
    updatemenu = []
    buttons = []
    
    for i, team in enumerate(teams):
        visible = [False] * len(teams)
        visible[i] = True
        
        button = dict(label=team,
                      method='update',
                      args=[{'visible': visible},
                            {'title': f"Shot Map for {team} in the {year}/{year+1} season"}])
        buttons.append(button)
    
    updatemenu.append(dict(buttons=buttons))
    fig.update_layout(updatemenus=updatemenu)
    
    
    fig.update_layout(
        width=1000,
        height=800,
    )
    
    fig.update_layout(
        xaxis_range=[-45, 45],
        yaxis_range=[0, 100]
    )

    return fig