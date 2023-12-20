import requests

from src.features.clean import clean_row


class GameClient:
    def __init__(self, ip: str = "0.0.0.0", port: int = 5000):
        self.base_url = f"http://{ip}:{port}"

        self.processed = {}

    def fetch_live_game_data(self, game_id):
        url = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play"
        response = requests.get(url)
        game_data = response.json()  # This will be a JSON object containing game data
        return game_data

    def make_prediction(self, features):
        url = f'{self.base_url}/predict'
        response = requests.post(url, json=features)
        return response.json()

    def process_events(self, events):
        for event in events:
            if event['gameID'] not in self.processed.keys():
                features = clean_row(event)
                preds = self.make_prediction(features)
                self.processed[event['gameID']] = preds