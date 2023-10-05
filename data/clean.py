import argparse
import os.path
from tqdm import tqdm
import numpy as np
import pandas as pd


def safe_get(dct, keys, default=np.NaN):
    """
    Safely get nested keys from a dictionary, return default if any key is not found.
    :param dct: dictionary to extract data from
    :param keys: dictionary keys to retrieve data
    :param default: return if any key is not found
    :return:
    """
    #Reference https://stackoverflow.com/questions/25833613/safe-method-to-get-value-of-nested-dictionary
    for key in keys:
        try:
            dct = dct[key]
        except (KeyError, TypeError):
            return default
    return dct


def clean_row(row):
    """
    Extract all shot and goal plays data from a row
    :param row: row to extract play data frame, should contain all data of a game, extracted using
    the community NHL API
    :return: dictionary containing data about all plays in one game
    """
    plays = []

    all_plays_list = row['liveData']['plays']['allPlays']
    shots_and_goals = [play for play in all_plays_list if (play['result']['event'] in ['Shot', 'Goal'])]

    for play in shots_and_goals:
        play_data = {
            'period': safe_get(play, ['about', 'period']),
            'period_type': safe_get(play, ['about', 'periodType']),
            'period_time': safe_get(play, ['about', 'periodTime']),
            'gameID': safe_get(row, ['gamePk']),
            'attacking_team_id': safe_get(play, ['team', 'id']),
            'attacking_team_name': safe_get(play, ['team', 'name']),
            'play_type': safe_get(play, ['result', 'event']),
            # Reference https://www.w3schools.com/python/ref_func_next.asp
            'shooter': next((player['player']['fullName'] for player in play.get('players', [])
                             if player['playerType'] in ['Scorer', 'Shooter']), np.NaN),
            'goalie': next((player['player']['fullName'] for player in play.get('players', [])
                            if player['playerType'] == 'Goalie'), np.NaN),
            'shot_type': safe_get(play, ['result', 'secondaryType']),
            'x_coordinate': safe_get(play, ['coordinates', 'x']),
            'y_coordinate': safe_get(play, ['coordinates', 'y']),
            'empty_net': safe_get(play, ['result', 'emptyNet']),
            'strength': safe_get(play, ['result', 'strength', 'name'])
        }
        plays.append(play_data)
    return plays


def clean_json(input_dir, output_dir):
    """
    Get clean data from raw NHL stats API
    :param input_dir: json file to clean data from, raw data
    :param output_dir: csv file to store clean data in
    :return: pandas dataframe containing the clean data
    """
    if os.path.exists(output_dir):
        return pd.read_csv(output_dir)
    if output_dir is None:
        output_dir = input_dir.replace('json', 'csv')

    df = pd.read_json(input_dir)
    tqdm.pandas()
    # Reference :https://stackoverflow.com/questions/18603270/progress-indicator-during-pandas-operations
    extracted_data = df.progress_apply(clean_row, axis=1)
    all_plays_list = [play for sublist in extracted_data for play in sublist]
    all_plays_df = pd.DataFrame(all_plays_list)

    all_plays_df.to_csv(output_dir, index=False)

    return all_plays_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Reference: https://stackoverflow.com/questions/4480075/argparse-optional-positional-arguments
    parser.add_argument('infile', type=str)
    parser.add_argument('outfile', type=str, nargs='?', default= None)
    args = parser.parse_args()

    clean_json(args.infile, args.outfile)
