"""..."""
import os
import re
import json
import unicodedata
import pandas as pd
from pprint import pprint


import re
import unicodedata



def process_name(name: str) -> str:
    """Process album or artist names for consistency while preserving foreign characters."""
    # 0. Lowercase the name.
    name = name.lower()

    # 1. Normalize the string using NFKC (this still composes characters but doesn't force ASCII)
    name = unicodedata.normalize('NFKC', name)

    # 2. Replace dollar signs with the letter "s"
    name = name.replace('$', 's')

    # 3. Remove anything inside parentheses or square brackets (including the brackets)
    name = re.sub(r'\(.*?\)', '', name)
    name = re.sub(r'\[.*?\]', '', name)

    # 4. Replace a period between two word characters with an underscore
    name = re.sub(r'(?<=[A-Za-z])\.(?=[A-Za-z])', '_', name)

    # 4.5. Replace any remaining periods with a space
    name = re.sub(r'\.', ' ', name)

    # 5. Replace '&' with 'and'
    name = name.replace('&', 'and')

    # 6. Remove all symbols except word characters, whitespace, hyphens, and en-dashes
    name = re.sub(r"[^\w\s\-–]", "", name)

    # 7. Collapse whitespace around hyphens (ensuring no extra spaces around them)
    name = re.sub(r'\s*([-–])\s*', r'\1', name)

    # 8. Collapse multiple spaces into one and trim leading/trailing whitespace
    name = re.sub(r'\s+', ' ', name).strip()

    # 9. Replace all remaining spaces with hyphens
    name = name.replace(' ', '-')

    return name


def get_query_columns(album_df : pd.DataFrame) -> list[str]:
    """Adds the query columns to our DataFrame. This method assumes that the column names
        'ALBUM' and 'ARTIST'
       are part of the dataframe... otherwise it will throw an error of course. It then just
       returns a dictionary of the format `artist : album` such as:

        query_dict = {
            'bladee': 'red-light',
            'ecco2k': 'e',
            'burial': 'untrue',
        }
    """
    album_df['query_artist'] = album_df['ARTIST'].apply(process_name)
    album_df['query_album']  = album_df['ALBUM'].apply(process_name)
    return album_df[['query_artist', 'query_album']].to_dict(orient='records')


def get_missing_albums(csv_path : str, progress_path : str):
    album_df    = pd.read_csv(csv_path)
    album_list  = get_query_columns(album_df)
    result_list = []

    with open(progress_path, "r") as file:
        data        = json.load(file)
        album_cache = data.get("album_cache", {})

    for album_data in album_list:
        artist    = album_data['query_artist']
        album     = album_data['query_album']
        album_key = f"{artist}/{album}"

        if album_key not in album_cache:
            result_list.append({'query_artist' : artist, 'query_album' : album})

    pprint(result_list)
    return result_list



get_missing_albums(csv_path     ='/Users/nico/Desktop/CIIC/CAPSTONE/essentia_demo/grouped_output_750_v2_clean.csv',
                   progress_path='/Users/nico/Desktop/CIIC/CAPSTONE/essentia_demo/progress.json')
