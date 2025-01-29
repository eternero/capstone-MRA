"""
In this file we define an object to acquire track info from the Spotify API.
"""

import base64
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class SpotifyAPI:
    """_summary_
    """

    def __init__(self, client_id : str, client_secret : str):
        self.access_token   = None

        self.client_id      = client_id
        self.client_secret  = client_secret

    def request_access_token(self):
        """Obtain a Spotify access token via the Client Credentials Flow."""
        endpoint = "https://accounts.spotify.com/api/token"

        # Encode client_id:client_secret in Base64
        auth_str = f"{self.client_id}:{self.client_secret}"
        b64_auth_str = base64.b64encode(auth_str.encode()).decode()

        headers = {
            "Authorization": f"Basic {b64_auth_str}",
            "Content-Type" : "application/x-www-form-urlencoded"
        }
        data = {
            "grant_type": "client_credentials"
        }

        try:
            response = requests.post(endpoint, headers=headers, data=data, timeout=5)
            token_info = response.json()
            self.access_token = token_info["access_token"]
            return self.access_token

        except requests.exceptions.HTTPError as err:
            print(f"Failed to get token: {response.status_code} {response.text}\nError:\n{err}")

    def get_track_popularity(self, track_artist : str, track_name : str) -> str:
        """Uses the Spotify API to acquire the popularity for a given track.

        Args:
            track_artist : _description_
            track_name   : _description_

        Returns:
            str: _description_
        """

        # First, make sure that we have a fresh access token, since these expire every hour.
        self.request_access_token()

        # Build our query with `track_artist` and `track_name`
        query = f"{track_artist} {track_name}"

        endpoint = "https://api.spotify.com/v1/search"
        headers  = {
            "Authorization": f"Bearer {self.access_token}",
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
            return data['tracks']['items'][0]['popularity']

        except requests.exceptions.HTTPError as err:
            print(f"Failed to retrieve data: {response.status_code} {response.text}\nError:\n{err}")
    