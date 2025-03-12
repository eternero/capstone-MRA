"""
...
"""
import os
from typing import Any, Dict, Optional
from mutagen.id3 import ID3
from mutagen.mp3 import MP3
from mutagen.flac import FLAC


class MetadataExtractor:
    """
    Extracts metadata from audio files (FLAC and MP3) using Mutagen.
    
    This version is stateless. The `extract` method takes the file path as input,
    making it clear that extraction is solely a function of the input file.
    """

    @staticmethod
    def _process_field(field: Any) -> Any:
        """
        Processes a metadata field that may be a list or a single value. Real shit this doesn't even
        fucking work because I get multiple artists all the fucking time because of commmas and fts.
        
        Args:
            field: The metadata field value.
        
        Returns:
            If the field is a list, returns its first element; otherwise, returns the field as is.
        """
        if isinstance(field, list):
            return field[0]
        return field

    @staticmethod
    def extract(track_path: str) -> Optional[Dict[str, Any]]:
        """
        Extracts metadata from the audio file specified by `track_path`.

        Args:
            track_path: The file system path to the audio track.

        Returns:
            A dictionary containing metadata (filename, artist, title, album, album artist),
            or None if the file format is unsupported.
        """
        extension = os.path.splitext(track_path)[1].lower()
        if extension == ".flac":
            return MetadataExtractor._extract_flac(track_path)
        elif extension == ".mp3":
            return MetadataExtractor._extract_mp3(track_path)
        else:
            print("Invalid file type, please provide a FLAC or MP3 only.")
            return None

    @staticmethod
    def _extract_flac(track_path: str) -> Dict[str, Any]:
        """
        Extracts metadata from a FLAC file.

        Args:
            track_path: The path to the FLAC file.

        Returns:
            A dictionary containing metadata for the FLAC file.
        """
        audio = FLAC(track_path)
        return {
            "FILENAME": os.path.basename(track_path),
            "ARTIST"  : MetadataExtractor._process_field(audio.get("artist") or audio.get("artists")),
            "TITLE"   : MetadataExtractor._process_field(audio.get("title") or audio.get("song_name")),
            "ALBUM"   : MetadataExtractor._process_field(audio.get("album") or audio.get("album_name")),
            "ALBUM_ARTIST": MetadataExtractor._process_field(audio.get("album_artist") or audio.get("albumartist")),
            "RELEASE_YEAR": MetadataExtractor._process_field(audio.get("date") or audio.get("year"))
        }

    @staticmethod
    def _extract_mp3(track_path: str) -> Dict[str, Any]:
        """
        Extracts metadata from an MP3 file.

        Args:
            track_path: The path to the MP3 file.

        Returns:
            A dictionary containing metadata for the MP3 file.
        """
        audio = MP3(track_path, ID3=ID3)
        tags = audio.tags
        return {
            "FILENAME": os.path.basename(track_path),
            "ARTIST"  : tags.get("TPE1").text[0] if "TPE1" in tags and tags.get("TPE1").text else None,
            "TITLE"   : tags.get("TIT2").text[0] if "TIT2" in tags and tags.get("TIT2").text else None,
            "ALBUM"   : tags.get("TALB").text[0] if "TALB" in tags and tags.get("TALB").text else None,
            "ALBUM_ARTIST": tags.get("TPE2").text[0] if "TPE2" in tags and tags.get("TPE2").text else None,
            "RELEASE_YEAR": tags.get("TDRC").text[0] if "TDRC" in tags and tags.get("TDRC").text else None
        }
