import pandas as pd
import numpy as np
from pathlib import Path
import json
from track_repository import TrackRepository

def prepare_features_dict(row, feature_columns):
    """Process a row from the dataframe and extract features."""
    features_dict = {}
    
    for col in feature_columns:
        value = row.get(col)
        # Handle different data types for JSON :)
        if isinstance(value, np.ndarray):
            features_dict[col] = value.tolist()
        elif isinstance(value, (int, float, str, bool, list, dict)) or value is None:
            features_dict[col] = value
        else:
            # Convert other types to string
            features_dict[col] = str(value)
    
    return features_dict

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
    
    # Initialize the database repository
    repo = TrackRepository(db_path)
    
    # Define which columns should be excluded from the features dictionary since they don't matter
    excluded_cols = [
        "filename", "artist", "title", "album", "length", "release_year","clean_artist", "clean_album",
        "segment_num", "segment_start", "uri", "sp_name, popularity"
    ]

    
    # Get feature columns (everything except excluded columns)
    feature_cols = [col for col in track_dataframe.columns if col not in excluded_cols]
    
    # Process each track in the dataframe
    print(f"Saving {len(track_dataframe)} tracks to database...")
    
    tracks_saved = 0
    for _, row in track_dataframe.iterrows():
        # Extract basic track information
        clean_artist = row.get("clean_artist", "")
        clean_album  = row.get("clean_album", "")
        clean_track  = row.get("title", "")
        uri = row.get("uri", None)
        
        # Convert popularity to integer if exists, just in case spotify returns it as a string or smth
        # They already fucked up the API
        popularity = None
        if "popularity" in row and row["popularity"] is not None:
            try:
                popularity = int(row["popularity"])
            except (ValueError, TypeError):
                popularity = None
        
        # Idk how to handle this yet, so just set it to 0 for now
        embedding = np.zeros(128, dtype=np.float32)
        
        # Prepare features dictionary
        features_dict = prepare_features_dict(row, feature_cols)
        
        try:
            repo.save_track_with_features(
                clean_artist=clean_artist,
                clean_album=clean_album,
                clean_track=clean_track,
                embedding=embedding,
                audio_features=features_dict,
                spotify_link=uri,
                popularity=popularity
            )
            tracks_saved += 1
            
            # Print progress every 20 tracks or at the end
            if tracks_saved % 20 == 0 or tracks_saved == len(track_dataframe):
                print(f"Saved {tracks_saved}/{len(track_dataframe)} tracks...")
                
        except Exception as e:
            print(f"Error saving track {clean_artist} - {clean_track}: {e}")
    
    print(f"Database inserts completed. Saved {tracks_saved} tracks.")
    print(f"Database location: {Path(db_path).absolute()}")
    return True

if __name__ == "__main__":
    store_tracks_in_database()