import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE

def remove_nan_from_strength(df : pd.DataFrame):
    """This function removes the NaN values from the 'strength' column and replace them with 'Even', 
    'Short Handed' or 'Power Play' depending on the number of players on the field """

    for index, row in df.iterrows():
    # On cherche les lignes dans lesquelles il y a des NaN
        if (row['strength'] != 'Even') & (row['strength'] != 'Power Play') & (row['strength'] != 'Short Handed'):
            # 
            if row['attacking_team_name'] == row['home_team'] :
                if row['home_team_players'] < row['away_team_players']:
                    df.at[index, 'strength'] = 'Short Handed'
                else : 
                    if row['home_team_players'] == row['away_team_players']:
                        df.at[index, 'strength'] = 'Even'
                    else :
                        df.at[index, 'strength'] = 'Power Play'
            else :
                if row['away_team_players'] > row['home_team_players']:
                    df.at[index, 'strength'] = 'Power Play'
                else : 
                    if row['away_team_players'] == row['home_team_players']:
                        df.at[index, 'strength'] = 'Even'
                    else : 
                        df.at[index, 'strength'] = 'Short Handed'

    return df



def encode_categorical_features(df : pd.DataFrame, categorical_features: list, shot_type_classified : list, strength_classified : list):
    """This function encodes the categorical features of our DataFrame. The columns 'strength' and 'shot_type'
    are encoded with an ordinal encoding, while the others the others are encoded using LabelEncoder()
    from sklearn"""
    df = df.copy()

    # Encodage des 'shot_type'
    mapping_dict = {row[0]: row[1] for row in shot_type_classified}
    df['shot_type'] = df['shot_type'].replace(mapping_dict)

    # Encodage de 'strength'
    mapping_dict_1 = {row[0] : row[1] for row in strength_classified}
    df['strength'] = df['strength'].replace(mapping_dict_1)

    # Encodage des autres caractéristiques
    label_encoder = LabelEncoder()

    for feature in categorical_features :
        df[feature] = label_encoder.fit_transform(df[feature]) 

    return df


def get_features_KBest(X: pd.DataFrame, Y: pd.Series, nb_features: int):
    """This function select the 'nb_features' most relevant features using the K-Best method, which is 
     a filtering method """

    # https://stackoverflow.com/questions/39839112/the-easiest-way-for-getting-feature-names-after-running-selectkbest-in-scikit-le
    selector = SelectKBest(f_classif, k = nb_features)
    X_new = pd.DataFrame(selector.fit_transform(X, Y))

    names = X.columns.values[selector.get_support()]

    return X[names], names


def oversample_dataset(X_train : pd.DataFrame, Y_train : pd.Series):
    """This function returns an oversampled dataset on the minor class of the dataset"""

    oversample_SM = SMOTE(sampling_strategy = 'minority')
    X_train_over, Y_train_over = oversample_SM.fit_resample(X_train,Y_train) 

    return X_train_over, Y_train_over


