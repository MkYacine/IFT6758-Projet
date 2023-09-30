import requests
import os
import pandas as pd
from tqdm import tqdm
import argparse


def num_games_by_year(year: str):
    """
    Get the number of games in regular season by year
    :param year: Season start year
    :return: Number of games in the regular season
    """
    if year >= '2022':
        return 1353
    elif '2017' <= year <= '2020':
        return 1271
    elif '1917' <= year < '2017':
        return 1230
    else:
        print('Invalid year')
        return None  # for years that are not valid


def generate_ids(year: str):
    """
    For a given year, create a generator object that returns all game ids in regular season
    and playoffs
    :param year: Year to generate ids for
    :return: One game id at a time
    """
    num_games = num_games_by_year(year)

    # Yield game id for regular season
    for i in range(1, num_games + 1):
        game_num_str = str(i).zfill(4)  # Zero-pad the game number to make it a 4-digit string
        yield f'{year}02{game_num_str}'  # Construct the ID string and yield it

    # Yield game id for playoffs
    for round_num in range(1, 5):  # 4 Rounds in total
        num_matchups = 2 ** (
                4 - round_num)  # 8 matchups in the first round, 4 in the second, 2 in the third, and 1 in the final
        for matchup_num in range(1, num_matchups + 1):
            for game_num in range(1, 8):  # Each matchup has up to 7 games
                yield f'{year}030{round_num}{matchup_num}{game_num}'


def scrape_games_by_year(year: str, data_dir: str = './data/datasets'):
    """
    For a given year, checks if data exists in cache and returns it. If not, scrape data from the
    web, store it in .csv file in data_dir
    :param year: Year to scrape data for
    :param data_dir: Directory where datasets are to be stored as .csv files
    :return: Pandas dataframe containing all games raw data
    """
    if os.path.exists(f"{data_dir}/{year}"):
        return pd.read_csv(f"{data_dir}/{year}")
    else:
        print(f"Data doesn't exist in cache, scraping data for the year {year}")

    # Instantiate generator object to generate ids
    generator = generate_ids(year)

    total_games = num_games_by_year(year) + 15 * 7  # Games in reg season + games in playoff
    rows = []

    # Scrape games, use tqdm for progression bar
    for game_id in tqdm(generator, total=total_games, desc="Scraping",
                        unit="game"):
        response = requests.get(f"https://statsapi.web.nhl.com/api/v1/game/{game_id}/feed/live/")
        if response.status_code == 200:
            json_data = response.json()
            rows.append(json_data)
        else:
            print(f'Failed to get data for game {game_id}')

    # Store games in local cache
    df = pd.DataFrame(rows)
    df.to_csv(f"{data_dir}/{year}.csv", index=False)

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Web scraping script.')
    parser.add_argument('year', type=str, help='The year to scrape data for.')
    args = parser.parse_args()

    scrape_games_by_year(args.year)
