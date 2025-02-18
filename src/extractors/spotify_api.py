"""
In this file we define an object to acquire track info from the Spotify API.
"""

import base64
import requests

class SpotifyAPI:
    """_summary_
    """

    @staticmethod
    def get_spotify_features(track_artist : str, track_name : str,
                             access_token : str) -> dict[str, str]:
        """Uses the Spotify API to acquire track features such as ID, Album Type and Popularity.

        By hitting the Spotify API Search Endpoint and filtering so that it only provides the top
        result for a track, we retrieve the response from the API and extract the previously
        mentioned features. If there is one drawback here, is that if the Spotify Search Algorithm
        fails us, then we will get the data for the wrong track... but that won't happen haha!

        Args:
            track_artist : The name of the artist for the track.
            track_name   : The name of the track to be searched for.

        Returns:
            A dictionary containg Track ID, Album Type and Popularity extracted from the results
            of the Spotify API Search Endpoint.

            NOTE : The idiot api groups EPs and Singles together!!! Albums and Mixtapes too!!!!
        """

        # First, make sure that we have a fresh access token, since these expire every hour.
        query = f"{track_artist} {track_name}"

        # Commence building our request query.
        endpoint = "https://api.spotify.com/v1/search"
        headers  = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type" : "application/json"
        }
        params   = {
            "q"     : query,        # The search query (artist + track name)
            "type"  : "track",      # We want to search for tracks only
            "limit" : 1,            # only one track
            "offset": 0             # the top search result
        }

        try:
            response = requests.get(endpoint, headers=headers, params=params, timeout=5)
            data = response.json()

            # Format the data and return it.
            spotify_features =  {
                                  "spotify_id" : data['tracks']['items'][0]['id'],
                                  "album_type" : data['tracks']['items'][0]['album']['album_type'],
                                  "popularity" : data['tracks']['items'][0]['popularity']
                                }
            return spotify_features

        except requests.exceptions.HTTPError as err:
            print(f"Failed to retrieve data: {response.status_code} {response.text}\nError:\n{err}")
            raise  # Raise so that the caller can handle it. see _get_metadata_and_spotify in track


def request_access_token(client_id : str, client_secret : str) -> str:
    """Obtain a Spotify access token via the Client Credentials Flow."""
    endpoint = "https://accounts.spotify.com/api/token"

    # Encode client_id:client_secret in Base64
    auth_str     = f"{client_id}:{client_secret}"
    b64_auth_str = base64.b64encode(auth_str.encode()).decode()

    headers = {
        "Authorization": f"Basic {b64_auth_str}",
        "Content-Type" : "application/x-www-form-urlencoded"
    }
    data = {
        "grant_type"   : "client_credentials"
    }

    try:
        response   = requests.post(endpoint, headers=headers, data=data, timeout=5)
        token_info = response.json()
        return token_info["access_token"]

    except requests.exceptions.HTTPError as err:
        print(f"Failed to get token: {response.status_code} {response.text}\nError:\n{err}")
