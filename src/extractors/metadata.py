"""
...
"""
import os
import re
import math
from typing import Any, Dict, Optional
from mutagen.id3 import ID3
from mutagen.mp3 import MP3
from mutagen.flac import FLAC
from src.utils.clean_csv import process_name


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

        NOTE : Added shit to deal with feats and parentheses. Not commas yet, might be dangerous!

        Args:
            field: The metadata field value.

        Returns:
            If the field is a list, returns its first element; otherwise, returns the field as is.
        """
        if not field:
            return None

        field = field[0] if isinstance(field, list) else field
        field = re.sub(r'\(.*?\)', '', field)

        # Remove the word "feat" (case-insensitive) and its common variations (e.g., "ft.")
        field = re.sub(r'\b(?:feat|ft)\b.*', '', field, flags=re.IGNORECASE)

        # Replace multiple whitespace with a single space and trim leading/trailing spaces
        field = re.sub(r'\s+', ' ', field).strip()
        return field

    @staticmethod
    def extract_year(date_str):
        # If it's already a valid year
        if re.fullmatch(r"\d{4}", date_str):
            return date_str

        # Match formats like YYYY-XX-XX
        match1 = re.match(r"^(\d{4})-\d{2}-\d{2}$", date_str)
        if match1:
            return match1.group(1)

        # Match formats like XX-XX-YYYY
        match2 = re.match(r"^\d{2}-\d{2}-(\d{4})$", date_str)
        if match2:
            return match2.group(1)

        # If none of the formats match, return None (or alternatively raise an error)
        return None


    @staticmethod
    def _clean_metadata(metadata : dict[str]) -> dict[str]:
        """Cleans the acquired metadata. The main functionality of this method is that (1) it
        overwrites the `artist` with the `album_artist` and (2) it cleans the `artist` and
        `album` fields so that they can then be used for JOINs when turned to tables.

        NOTE : Could add cleaning for release year. Some releases are in YYYY-MM-DD while others
               are in YYYY. Making them all YYYY is ideal.

        Args:
            metadata : The track metadata...

        Returns:
            The cleaned up metadata...
        """

        # First overwrite the `artist` field and once we're done, delete the `album_artist` field.`
        metadata["artist"] = (metadata["album_artist"] if metadata["album_artist"]
                                                       else metadata["artist"])
        del metadata["album_artist"]

        # Now, create the two new clean fields for `artist` and `album`
        metadata["clean_artist"] = process_name(metadata["artist"])
        metadata["clean_album"]  = process_name(metadata["album"])

        return metadata


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
        audio    = FLAC(track_path)
        metadata = {
            "filename": os.path.basename(track_path),
            "artist"       : MetadataExtractor._process_field(audio.get("artist") or audio.get("artists")),
            "title"        : MetadataExtractor._process_field(audio.get("title") or audio.get("song_name")),
            "album"        : MetadataExtractor._process_field(audio.get("album") or audio.get("album_name")),
            "length"       : math.floor(audio.info.length),
            "album_artist" : MetadataExtractor._process_field(audio.get("album_artist") or audio.get("albumartist")),
            "release_year" : MetadataExtractor._process_field(audio.get("date") or audio.get("year"))

        }

        metadata['release_year'] = MetadataExtractor.extract_year(metadata['release_year'])
        return MetadataExtractor._clean_metadata(metadata)


    @staticmethod
    def _extract_mp3(track_path: str) -> Dict[str, Any]:
        """
        Extracts metadata from an MP3 file.

        Args:
            track_path: The path to the MP3 file.

        Returns:
            A dictionary containing metadata for the MP3 file.
        """
        audio    = MP3(track_path, ID3=ID3)
        tags     = audio.tags
        metadata = {
            "filename"     : os.path.basename(track_path),
            "artist"       : tags.get("TPE1").text[0] if "TPE1" in tags and tags.get("TPE1").text else None,
            "title"        : tags.get("TIT2").text[0] if "TIT2" in tags and tags.get("TIT2").text else None,
            "album"        : tags.get("TALB").text[0] if "TALB" in tags and tags.get("TALB").text else None,
            "album_artist" : tags.get("TPE2").text[0] if "TPE2" in tags and tags.get("TPE2").text else None,
            "release_year" : tags.get("TDRC").text[0] if "TDRC" in tags and tags.get("TDRC").text else None
        }

        return MetadataExtractor._clean_metadata(metadata)
