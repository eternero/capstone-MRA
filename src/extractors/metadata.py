"""
...
"""
import os
from mutagen.id3 import ID3
from mutagen.mp3 import MP3
from mutagen.flac import FLAC


class MetadataExtractor:
    """_summary_
    """

    def __init__(self, track_path):
        self.track_path = track_path

    def process_field(self, field):
        """Handle list or single value metadata fields. Helper for `get_track_metadata()`."""
        if isinstance(field, list):
            return field[0]  # Extract the first element
        return field  # Return as is if not a list

    def process_flac(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        audio = dict(FLAC(self.track_path))
        return {
            'FILENAME' : os.path.basename(self.track_path),
            'ARTIST': self.process_field(audio.get('artist') or 
                                        audio.get('artists')),
            'TITLE': self.process_field(audio.get('title')   or 
                                        audio.get('song_name')),
            'ALBUM': self.process_field(audio.get('album')   or
                                        audio.get('album_name')),
            'ALBUM_ARTIST': self.process_field(audio.get('album_artist') or 
                                                audio.get('albumartist')),
        }

    def process_mp3(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        audio = MP3(self.track_path, ID3=ID3)
        return {
            "FILENAME": os.path.basename(self.track_path),
            "ARTIST": audio.tags.get("TPE1").text[0] if "TPE1" in audio.tags else None,
            "TITLE": audio.tags.get("TIT2").text[0] if "TIT2" in audio.tags else None,
            "ALBUM": audio.tags.get("TALB").text[0] if "TALB" in audio.tags else None,
            "ALBUM_ARTIST": audio.tags.get("TPE2").text[0] if "TPE2" in audio.tags else None
        }


path = "src/audio/01 BANK OF AMERIKKKA.mp3"
m_ex = MetadataExtractor(path)

from pprint import pprint
metadata = m_ex.process_mp3()
pprint(metadata)