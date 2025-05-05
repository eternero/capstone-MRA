import sqlite3
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

class TrackRepository:
    """Repository class for handling track database operations."""
    
    def __init__(self, db_path="data/music_data.db"):
        """Initialize the repository with database path."""
        self.db_path = db_path
        self._ensure_db_exists()
        
    def _ensure_db_exists(self):
        """Create database and tables if they don't exist."""
        Path(self.db_path).parent.mkdir(exist_ok=True)
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Create tracks table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS tracks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            clean_artist TEXT NOT NULL,
            clean_album TEXT NOT NULL,
            clean_track TEXT NOT NULL,
            spotify_link TEXT,
            popularity INTEGER,
            UNIQUE(clean_artist, clean_album, clean_track)
        )
        ''')
        
        # Create features table
        # TODO: Verify if its better o use BLOB or TEXT for the embedding
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS features (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            track_id INTEGER NOT NULL,
            clean_artist TEXT NOT NULL,
            clean_album TEXT NOT NULL,
            clean_track TEXT NOT NULL,
            embedding BLOB,
            audio_features TEXT,
            FOREIGN KEY (track_id) REFERENCES tracks (id),
            UNIQUE(track_id)
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def _get_connection(self):
        """Create and return a database connection."""
        return sqlite3.connect(self.db_path)
    
    def insert_track_table(self, clean_artist: str, clean_album: str, clean_track: str, 
                           spotify_link: Optional[str] = None, popularity: Optional[int] = None) -> int:
        """Save track information to database."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Check if track already exists, just in case
        cursor.execute(
            """
            SELECT id 
            FROM tracks 
            WHERE clean_artist = ? AND clean_album = ? AND clean_track = ?
            """,
            (clean_artist, clean_album, clean_track)
        )
        result = cursor.fetchone()
        
        if result:
            return result[0]  # Return existing track ID if it already exists (just in case)
        
        # Insert new track
        cursor.execute(
            """
            INSERT INTO tracks 
            (clean_artist, clean_album, clean_track, spotify_link, popularity) 
            VALUES (?, ?, ?, ?, ?)
            """,
            (clean_artist, clean_album, clean_track, spotify_link, popularity)
        )
        track_id = cursor.lastrowid

        if track_id is None:
            raise Exception("Database instert failed, no row ID returned.")
        
        conn.commit()
        conn.close()
        return track_id
    
    def insert_features_table(self, track_id: int, clean_artist: str, clean_album: str, clean_track: str, 
                              embedding: np.ndarray, audio_features: Dict[str, Any]) -> int:
        """Save audio features for a track."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Convert numpy array to bytes for storage
        embedding_bytes = embedding.tobytes()
        
        # Convert audio features dict to JSON string
        audio_features_json = json.dumps(audio_features)
        
        # Check if features for this track already exist
        cursor.execute("SELECT id FROM features WHERE track_id = ?", (track_id,))
        result = cursor.fetchone()
        
        if result:
            # Update existing features
            cursor.execute(
                """
                UPDATE features 
                SET clean_artist = ?, clean_album = ?, clean_track = ?, embedding = ?, audio_features = ?
                WHERE track_id = ?
                """,
                (clean_artist, clean_album, clean_track, embedding_bytes, audio_features_json, track_id)
            )
            feature_id = result[0]
        else:
            # Insert new features
            cursor.execute(
                """
                INSERT INTO features 
                (track_id, clean_artist, clean_album, clean_track, embedding, audio_features) 
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (track_id, clean_artist, clean_album, clean_track, embedding_bytes, audio_features_json)
            )
            feature_id = cursor.lastrowid
        
        if feature_id is None:
            raise Exception("Database insert failed, no row ID returned.")
        
        conn.commit()
        conn.close()
        return feature_id
    
    def get_track(self, track_id: int = None, clean_artist: str = None, 
                  clean_album: str = None, clean_track: str = None) -> Optional[Dict]:
        """Retrieve a track by ID or metadata."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if track_id:
            cursor.execute("SELECT * FROM tracks WHERE id = ?", (track_id,))
        elif clean_artist and clean_album and clean_track:
            cursor.execute(
                "SELECT * FROM tracks WHERE clean_artist = ? AND clean_album = ? AND clean_track = ?",
                (clean_artist, clean_album, clean_track)
            )
        else:
            conn.close()
            raise ValueError("Must provide either track_id or (clean_artist, clean_album, clean_track)")
        
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            return None
            
        return {
            "id": result[0],
            "clean_artist": result[1],
            "clean_album": result[2],
            "clean_track": result[3],
            "spotify_link": result[4],
            "popularity": result[5]
        }
        
    def get_features(self, track_id: int) -> Optional[Dict]:
        """Retrieve features for a track by ID."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM features WHERE track_id = ?", (track_id,))
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            return None
        
        # Convert bytes back to numpy array
        embedding_bytes = result[5]
        embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
        
        # Parse JSON string back to dict
        audio_features = json.loads(result[6])
        
        return {
            "id": result[0],
            "track_id": result[1],
            "clean_artist": result[2],
            "clean_album": result[3],
            "clean_track": result[4],
            "embedding": embedding,
            "audio_features": audio_features
        }
    
    def save_track_with_features(self, clean_artist: str, clean_album: str, clean_track: str,
                               embedding: np.ndarray, audio_features: Dict[str, Any],
                               spotify_link: Optional[str] = None, popularity: Optional[int] = None) -> int:
        """
        To save a track and its features in the database, there are separated methods for 
        saving the track and the features. This method combines them for convenience.
        """
        # First save the track
        track_id = self.insert_track_table(
            clean_artist=clean_artist, 
            clean_album=clean_album, 
            clean_track=clean_track,
            spotify_link=spotify_link, 
            popularity=popularity
        )
        
        # Then save the features
        self.insert_features_table(
            track_id=track_id,
            clean_artist=clean_artist,
            clean_album=clean_album,
            clean_track=clean_track,
            embedding=embedding,
            audio_features=audio_features
        )
        
        return track_id