"""..."""
import os
import re
import unicodedata
import pandas as pd
from pprint import pprint


def process_name(name: str) -> str:
    """Process album or artist names for consistency while preserving foreign characters."""
    # 0. Lowercase the name.
    name = name.lower()

    # 1. Normalize the string using NFKC (this still composes characters but doesn't force ASCII)
    name = unicodedata.normalize('NFKC', name)

    # 1.5 Second round of normalization.
    decomposed_name = unicodedata.normalize('NFD', name)
    filtered_name   = []
    for char in decomposed_name:
        if unicodedata.combining(char):
            continue
        filtered_name.append(char)
    name = unicodedata.normalize('NFC', "".join(filtered_name))

    # 2. Replace dollar signs with underscore "_"
    name = name.replace('$', '_')

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


def get_query_columns(csv_path : str) -> pd.DataFrame:
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
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No file found at '{csv_path}'")

    # Acquire the dataframe and rename it to have everything in lowercase.
    album_df                 = pd.read_csv(csv_path)
    album_df                 = album_df.rename(columns  = {'ALBUM'  : 'album',
                                                           'ARTIST' : 'artist'})
    album_df['query_artist'] = album_df['artist'].apply(process_name)
    album_df['query_album']  = album_df['album'].apply(process_name)
    return album_df
