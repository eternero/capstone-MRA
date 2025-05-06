import pandas as pd
import numpy as np
from pathlib import Path
import json
from track_repository import TrackRepository

def extract_features(row):
    """..."""
    features = {}
    
    numeric_features = [
        'electronic_effnet', 'instrumental_effnet', 'acoustic_effnet',
        'female_effnet', 'bpm', 'happy_effnet', 'danceable_effnet',
        'aggressive_effnet', 'relaxed_effnet', 'party_effnet', 'sad_effnet'
    ]
    
    for feature in numeric_features:
        if feature in row:
            features[feature] = row[feature]
    
    if 'mfcc_mean' in row:
        features['mfcc_mean'] = row['mfcc_mean']
    
    if 'discogs_effnet_embeddings' in row:
        features['discogs_effnet_embeddings'] = row['discogs_effnet_embeddings']
    
    return features

def store_tracks_in_database(csv_path="src/audio/output_features.csv", db_path="src/backend/data/music_data.db"):
    """
    Process tracks from the CSV file and store them in the database.
    """
    # Check if CSV file exists
    csv_file = Path(csv_path)
    if not csv_file.exists():
        print(f"Error: CSV file not found at {csv_file}")
        return False
    
    # Load the CSV data
    print(f"Loading track data from {csv_file}...")
    track_dataframe = pd.read_csv(csv_file)
    
    # Process each track in the dataframe
    print(f"Saving {len(track_dataframe)} tracks to database...")
    
    repo = TrackRepository(db_path)

    tracks_saved = 0
    for idx, row in track_dataframe.iterrows():
        # Extract the main track information
        clean_artist = row.get("clean_artist", "")
        clean_album = row.get("clean_album", "")
        clean_track = row.get("title", "")
        uri = row.get("uri", None)
        
        # Convert popularity to integer if exists
        popularity = None
        if "popularity" in row and row["popularity"] is not None:
            try:
                popularity = int(row["popularity"])
            except (ValueError, TypeError):
                popularity = None
        
        # Extract specific features needed for the recommendation system
        features = extract_features(row)
        
        # Save track and features with one call
        try:
            repo.save_track_with_features(
                clean_artist=clean_artist,
                clean_album=clean_album,
                clean_track=clean_track,
                features=features,
                spotify_link=uri,
                popularity=popularity
            )
            tracks_saved += 1
            
            # Print progress every 20 tracks or at the end
            if tracks_saved % 20 == 0 or tracks_saved == len(track_dataframe):
                print(f"Saved {tracks_saved}/{len(track_dataframe)} tracks...")
                
        except Exception as e:
            print(f"Error saving track {clean_artist} - {clean_track}: {e}")
    
    print(f"Database operations completed. Saved {tracks_saved} tracks.")
    print(f"Database location: {Path(db_path).absolute()}")
    return True

if __name__ == "__main__":
    store_tracks_in_database()