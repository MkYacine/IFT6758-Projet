import os.path
from tqdm import tqdm
import numpy as np
import pandas as pd

import src.features.feature_engineering_1 as feng1
import src.visualization.simple_visualization as sv


def safe_get(dct, keys, default=np.NaN):
    """
    Safely get nested keys from a dictionary, return default if any key is not found.
    :param dct: dictionary to extract data from
    :param keys: dictionary keys to retrieve data
    :param default: return if any key is not found
    :return:
    """
    # Reference https://stackoverflow.com/questions/25833613/safe-method-to-get-value-of-nested-dictionary
    for key in keys:
        try:
            dct = dct[key]
        except (KeyError, TypeError):
            return default
    return dct


def clean_row(row):
    plays = []
    all_play_list = row["plays"]

    for play in all_play_list:
        if play["typeDescKey"] in [
            "blocked-shot",
            "shot-on-goal",
            "missed-shot",
            "goal",
        ]:
            # Teams information
            home_team_name = row["homeTeam"]["name"]["default"]
            home_team_id = row["homeTeam"]["id"]
            away_team_name = row["awayTeam"]["name"]["default"]
            away_team_id = row["awayTeam"]["id"]

            # Processing of Situation Code
            situation_Code = safe_get(play, ["situationCode"])
            away_team_gk = situation_Code[0]
            away_team_players = situation_Code[1]
            home_team_players = situation_Code[2]
            home_team_gk = situation_Code[3]

            # Checking if the net is empty or not
            event_Owner_id = safe_get(play, ["details", "eventOwnerTeamId"])
            if event_Owner_id == home_team_id:
                if home_team_gk == "1":
                    empty_net = 0
                else:
                    empty_net = 1
            else:
                if away_team_gk == "1":
                    empty_net = 0
                else:
                    empty_net = 1

            play_data = {
                "gameID": row["id"],
                "period": safe_get(play, ["period"]),
                "period_type": safe_get(play, ["periodDescriptor", "periodType"]),
                "period_time": safe_get(play, ["timeInPeriod"]),
                "home_team_name": home_team_name,
                "home_team_id": home_team_id,
                "away_team_name": away_team_name,
                "away_team_id": away_team_id,
                "shot_type": safe_get(play, ["details", "shotType"]),
                "event_type": safe_get(play, ["typeDescKey"]),
                "home_team_defending_side": safe_get(play, ["homeTeamDefendingSide"]),
                "x_coordinate": safe_get(play, ["details", "xCoord"]),
                "y_coordinate": safe_get(play, ["details", "yCoord"]),
                "event_owner_team_id": safe_get(play, ["details", "eventOwnerTeamId"]),
                "away_team_players": away_team_players,
                "home_team_players": home_team_players,
                "empty_net": empty_net,
            }
            plays.append(play_data)

    return plays


def add_new_features(df: pd.DataFrame):
    # Ajout de la colonne distance au DataFrame
    df["distance_to_net"] = df.apply(
        lambda x: sv.compute_distance_to_net(x["x_coordinate"], x["y_coordinate"]),
        axis=1,
    )
    df["distance_to_net"] = df["distance_to_net"].apply(lambda x: round(x, 1))

    # Ajout de la colonne 'shot_angle'
    df["shot_angle"] = df.apply(
        lambda x: feng1.compute_shot_angle(x["x_coordinate"], x["y_coordinate"]), axis=1
    )
    df["shot_angle"] = df["shot_angle"].apply(lambda x: round(x, 1))

    # Ajout de la colonne is_goal
    df["is_goal"] = df["event_type"].apply(lambda x: 1 if x == "goal" else 0)

    return df


def clean_json(input_dir, output_dir=""):
    if os.path.exists(output_dir):
        return pd.read_csv(output_dir)
    if output_dir == "":
        outpur_dir = input_dir.replace("json", "csv")

    df = pd.read_json(input_dir)
    tqdm.pandas()
    extracted_data = df.progress_apply(clean_row, axis=1)
    all_plays_list = [play for sublist in extracted_data for play in sublist]
    all_plays_df = pd.DataFrame(all_plays_list)

    all_plays_df = add_new_features(all_plays_df)

    all_plays_df.to_csv(output_dir, index=False)

    return all_plays_df


from concurrent.futures import ThreadPoolExecutor


def print_data(game_data):
    # rows = []

    # rows.extend([result for result in game_data if result is not None])

    rows = [
        game_data,
    ]

    df = pd.DataFrame(rows)

    extracted_data = df.apply(clean_row, axis=1)

    all_plays_list = [play for sublist in extracted_data for play in sublist]
    all_plays_df = pd.DataFrame(all_plays_list)

    all_plays_df = add_new_features(all_plays_df)

    return all_plays_df
