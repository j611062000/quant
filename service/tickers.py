import json


def get_tickers() -> list[str]:
    with open('config.json', 'r') as f:
        config = json.load(f)
    return config['intetested_tickers']