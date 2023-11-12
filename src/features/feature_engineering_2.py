import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import os


def add_new_features(df: pd.DataFrame):
    # Rebond
    df['rebound'] = df['last_event_type']=='Shot'

    # Changement d'angle
    df['angle_difference'] = df['shot_angle'].diff().fillna(0)
    df['angle_change'] = np.where(df['rebound'], df['angle_difference'], 0)
    df.drop(['angle_difference'], axis = 1, inplace= True)

    # Vitesse
    df['speed'] = df['distance_from_last_event'] / df['time_since_last_event'].replace({0: np.nan})
    return df

def encode_categorical_features(df : pd.DataFrame, categorical_features: list, shot_type_classified : list):
    df = df.copy()

    # Encodage des 'shot_type'
    mapping_dict = {row[0]: row[1] for row in shot_type_classified}
    df['shot_type'] = df['shot_type'].replace(mapping_dict)

    # Encodage des autres caractéristiques
    label_encoder = LabelEncoder()

    for feature in categorical_features :
        df[feature] = label_encoder.fit_transform(df[feature]) 

    return df

def select_features(data: pd.DataFrame):
    df = data[['period', 'game_seconds', 'x_coordinate', 'y_coordinate', 'distance_to_net', 'shot_angle', 'shot_type', 'last_event_type', 'last_event_x', 'last_event_y', 'time_since_last_event', 'distance_from_last_event', 'rebound', 'angle_change', 'speed', 'powerplay_duration', 'home_team_players', 'away_team_players', 'is_goal']]
    X, y = df.drop(['is_goal'], axis=1), df['is_goal']


    # One hot encoding last_event_type
    encoder = OneHotEncoder(sparse=False, drop='if_binary')
    categorical_columns = ['last_event_type']
    data_encoded = encoder.fit_transform(X[categorical_columns])
    data_encoded = pd.DataFrame(data_encoded, columns=encoder.get_feature_names_out(categorical_columns))
    
    # Drop original categorical columns and concatenate encoded columns
    X = X.drop(columns=categorical_columns)
    X = pd.concat([X, data_encoded], axis=1)

    # Transforming rebound to int
    X['rebound'] = X['rebound'].astype(int)

    # Label encoding shot_type
    shot_type_classified = [['Wrap-around',0], ['Slap Shot', 1], ['Snap Shot', 2], ['Wrist Shot', 3], ['Backhand', 4], ['Deflected', 5], ['Tip-In',6]]
    X = encode_categorical_features(X, ['shot_type'], shot_type_classified)

    return X, y