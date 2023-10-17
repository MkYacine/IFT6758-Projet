import requests
import os
import pandas as pd
from tqdm import tqdm
import argparse
from concurrent.futures import ThreadPoolExecutor


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


def fetch_game_data(game_id):
    """
    Fetch game data for a specific game_id.
    :param game_id: The game id to fetch data for
    :return: JSON data of the game or None if request fails
    """
    response = requests.get(f"https://statsapi.web.nhl.com/api/v1/game/{game_id}/feed/live/")
    if response.status_code == 200:
        return response.json()
    else:
        print(f'Failed to get data for game {game_id}')
        return None


def scrape_games_by_year(year: str, data_dir: str = './data/datasets/json_files'):
    """
    For a given year, checks if data exists in cache and returns it. If not, scrape data from the
    web, store it in .json file in data_dir
    :param year: Year to scrape data for
    :param data_dir: Directory where datasets are to be stored as .csv files
    :return: Pandas dataframe containing all games raw data
    """
    if os.path.exists(f"{data_dir}/{year}.json"):
        print(f"Data for the year {year} found in cache!")
        df = pd.read_json(f"{data_dir}/{year}.json")
        return df
    else:
        print(f"Data doesn't exist in cache, scraping data for the year {year}")

    # Instantiate generator object to generate ids
    generator = generate_ids(year)

    total_games = num_games_by_year(year) + 15 * 7  # Games in reg season + games in playoff
    rows = []

    # Scrape data using multi-threading
    # Reference: https://superfastpython.com/threadpoolexecutor-map/
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(tqdm(executor.map(fetch_game_data, generator), total=total_games, desc="Scraping", unit="game"))

    # Filter out None results
    rows.extend([result for result in results if result is not None])

    # Store games in local cache
    df = pd.DataFrame(rows)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    df.to_json(f"{data_dir}/{year}.json")
    return df


def scrape_multiple_years(years : list, output_file: str = './data/datasets/json_files'):
    """
    This function takes a list of years (can be an int or an str) as an argument 
    and returns a DataFrame containing the data for all those years
    """
    # Dans le cas où on veut récupérer les données d'une seule année
    if len(years) == 1 :
        df = scrape_games_by_year(years[0])
        output_path = os.path.join(output_file,f"{years[0]}.json")
        df.to_json(output_path, index = False)
        return df

    # On vérifie si les données pour cette range d'années sont déjà présentes
    if os.path.exists(f"{output_file}/{years[0]}_to_{years[-1]}.json"):
        return pd.read_json(f"{output_file}/{years[0]}_to_{years[-1]}.json")
    else:
        print(f"Data doesn't exist in cache, scraping data for the years {years[0]} to {years[-1]}")

    # DataFrame de sortie contenant les données pour les années contenues dans "years"
    df = pd.DataFrame()
    # DataFrame auxiliaire
    df_1 = pd.DataFrame()

    for year in years :
        df_1 = scrape_games_by_year(str(year), output_file)
        df = pd.concat([df,df_1], ignore_index = True)

    # Libération de la mémoire
    del df_1

    # Ecriture du fichier .csv pour toutes les années voulues
    df.to_json(f"{output_file}/{years[0]}_to_{years[-1]}.json", index = False)

    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Web scraping script.')
    parser.add_argument('year', type=str, help='The year to scrape data for.')
    args = parser.parse_args()

    scrape_games_by_year(args.year)
