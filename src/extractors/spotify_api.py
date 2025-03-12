"""
In this file we define an object to acquire track info from the Spotify API.
"""

import json
import base64
import requests

class SpotifyAPI:
    """_summary_
    """

    @staticmethod
    def get_access_token(client_id : str, client_secret : str, auth_code : str) -> str:
        """Exchange authorization code for access token."""
        token_headers = {
            'Authorization': 'Basic ' + base64.b64encode(f'{client_id}:{client_secret}'.encode()).decode()
            }
        token_data = {
            'grant_type': 'authorization_code',
            'code': auth_code,
            'redirect_uri': 'https://127.0.0.1:5555/callback'
        }

        token_url      = 'https://accounts.spotify.com/api/token'
        token_response = requests.post(token_url, headers=token_headers, data=token_data, timeout=5)
        token_info     = token_response.json()

        return token_info['access_token']


    @staticmethod
    def get_auth_access_token(client_id : str, client_secret : str, scope : str) -> str:
        """Obtain a Spotify access token via the Authorization Code Flow.

        Args:
            client_id     : The Client ID generated after registering your application.
            client_secret : The Client SECRET generated after registering your application.
            scope         : A space-separated list of scopes. If no scopes are specified, 
                            authorization will be granted only to access publicly available 
                            information: that is, only information normally visible in the Spotify
                            desktop, web, and mobile players.

        Returns:
            str: The access token exchanged for an authorization  code.
        """

        # Direct user to authorize!
        auth_params = {
            'client_id'    : client_id,
            'response_type': 'code',
            'redirect_uri' : 'https://127.0.0.1:5555/callback',
            'scope'        : scope
        }

        auth_url     = 'https://accounts.spotify.com/authorize'
        auth_request = requests.get(auth_url, params=auth_params, timeout=5)
        print(f'Please go to the following URL to authorize: {auth_request.url}')

        # Time to get the access token. This one will have the privileges determined in the scope
        auth_code = input('Enter the authorization code from the URL: ')
        return      SpotifyAPI.get_access_token(client_id, client_secret, auth_code)

    @staticmethod
    def _create_playlist(user_id : str, access_token : str, playlist_name : str):
        """Create a new playlist."""
        playlist_data = {
            'name'       : playlist_name,
            'description': "I don't care about descriptions",
            'public'     : True
        }

        user_headers = {
            'Authorization': f'Bearer {access_token}',
            'Content-Type' : 'application/json'
        }

        playlist_url      = f"https://api.spotify.com/v1/users/{user_id}/playlists"
        playlist_response = requests.post(playlist_url, headers=user_headers,
                                          json=playlist_data, timeout=5)
        return playlist_response.json()


    @staticmethod
    def _add_tracks_to_playlist(playlist_id : str, access_token : str, uri_list : list[str]):

        # A maximum of 100 items can be added in one request.
        curr_uris      = uri_list[:100]
        remaining_uris = uri_list[100:]

        # Commence preparing the request!
        add_tracks_data = {
            'uris': curr_uris
        }
        user_headers = {
            'Authorization': f'Bearer {access_token}',
            'Content-Type': 'application/json'
        }

        add_tracks_url = f"https://api.spotify.com/v1/playlists/{playlist_id}/tracks"
        requests.post(add_tracks_url, headers=user_headers, json=add_tracks_data, timeout=5)

        if remaining_uris:
            SpotifyAPI._add_tracks_to_playlist(playlist_id, access_token, remaining_uris)



    @staticmethod
    def populate_playlist(playlist_name : str, access_token : str, uri_list : list[str]) -> None:
        """Add a set of tracks to a new playlist."""

        # Use the access token to acquire the Spotify User ID which is needed for playlist creation.
        user_headers = {
       'Authorization': f'Bearer {access_token}'
        }
        user_profile_url = 'https://api.spotify.com/v1/me'
        user_response    = requests.get(user_profile_url, headers=user_headers, timeout=5)
        user_info        = user_response.json()
        user_id          = user_info['id']

        # Create a new playlist
        playlist_info = SpotifyAPI._create_playlist(user_id, access_token, playlist_name)
        playlist_id = playlist_info['id']

        # Populate the playlist
        SpotifyAPI._add_tracks_to_playlist(playlist_id, access_token, uri_list)


    @staticmethod
    def get_spotify_features(track_artist: str, track_name: str,
                             track_album : str, access_token: str) -> dict:
        """
        Retrieves track features such as ID, URI, popularity, and album type from the Spotify API.

        Args:
            track_artist : The name of the artist for the track.
            track_name   : The name of the track to be searched for.
            track_album  : The name of the album the track is part of.
            access_token : The OAuth token for accessing the Spotify API.

        Returns:
            A dictionary containing the track's Spotify ID, URI, popularity, and album type.

        Raises:
            requests.exceptions.RequestException: An error occurred while making the HTTP request.
            KeyError: The expected data was not found in the API response.
        """

        # Construct the search query with field filters
        # query = f'track:"{track_name}" artist:"{track_artist}" album:"{track_album}"'    # Old query w/ filters.
        query = f'{track_artist} {track_name} {track_album}'

        # Define the endpoint and headers
        endpoint = "https://api.spotify.com/v1/search"
        headers  = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
        params   = {
            "q"      : query,       # The search query with filters
            "type"   : "track",     # Search for tracks only
            "limit"  : 1,           # Retrieve only one track (the top result)
            "offset" : 0            # Start from the top search result
        }

        try:
            # Make the request to the Spotify API
            response = requests.get(endpoint, headers=headers, params=params, timeout=5)
            response.raise_for_status()  # Raise an exception for HTTP errors
            data = response.json()

            # Extract the track information from the response
            track_item = data['tracks']['items'][0]
            spotify_features =  {
                                  "spotify_id" : track_item['id'],
                                  "uri"        : track_item['uri'],
                                  "sp_name"    : track_item['name'],
                                  "popularity" : track_item['popularity'],
                                  "album_id"   : track_item['album']['id'],
                                  "album_type" : track_item['album']['album_type'],
                                }
            return spotify_features

        except requests.exceptions.RequestException as err:
            print(f"HTTP Request failed: {err}")
            raise

        except (KeyError, IndexError) as err:
            print(f"Error processing data: {err}")
            raise

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
