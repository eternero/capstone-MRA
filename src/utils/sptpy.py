"""Using SpotiPy to simplify operations regarding the Spotify API that require authentication."""

import spotipy
from spotipy.oauth2 import SpotifyOAuth

def create_playlist_with_tracks(client_id : str, client_secret : str,
                                redirect_uri : str, playlist_name : str, uri_list : list[str]):
    """Creates a new Spotify Playlist given a list of Track URIs.

    Args:
        client_id     : The Client ID generated after registering your application.
        client_secret : The Client SECRET generated after registering your application.
        redirect_uri  : The callback URI(s) which have been created in your Spotify API Dashboard.
        playlist_name : The name of the playlist.
        uri_list      : The list of Spotify Track URIs. These represent the tracks to be added.
    """

    # Set up authentication
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=client_id,
                                                   client_secret=client_secret,
                                                   redirect_uri=redirect_uri,
                                                   scope='playlist-modify-public'))

    # Retrieve current user's profile information
    user_id = sp.me()['id']

    # Create a new playlist
    playlist = sp.user_playlist_create(user=user_id, name=playlist_name, public=True)
    playlist_id = playlist['id']

    # Add tracks to the playlist in batches of 100 (Spotify's limit per request)
    for i in range(0, len(uri_list), 100):
        sp.playlist_add_items(playlist_id, uri_list[i:i + 100])

    print(f'Playlist "{playlist_name}" created successfully with {len(uri_list)} tracks.')
