"""
In this file I'll be defining a class that assists us at converting FLAC files to MP3 files. This 
is so that we can have two datasets, the initial one, which consists of FLAC files and a converted
dataset with MP3 tracks.

We'll check if there are any major differences with the audio pipeline between FLAC and MP3 files, 
and if there isn't then we might just stick with MP3 files to save us some processing power and 
storage as well.
"""

import os
import glob
import subprocess

class FlacConvert:
    """
    Object in charge of converting FLAC files to MP3s
    """

    def __init__(self, source_dir):
        self.source_dir = source_dir

        # This is the target directory where .mp3 files will be saved
        self.target_dir = "audio_dataset_mp3"

    def convert_files(self):
        """
        Method to convert out FLAC files into MP3s
        """
        # Create the target directory if it doesn't exist
        if not os.path.exists(self.target_dir):
            os.makedirs(self.target_dir)

        # Iterate over all .flac files in the source directory
        flac_files = glob.glob(os.path.join(self.source_dir, "*.flac"))
        for flac_file in flac_files:
            # Extract the base filename without extension
            base_name = os.path.splitext(os.path.basename(flac_file))[0]

            # Define the output filename (MP3)
            mp3_file = os.path.join(self.target_dir, base_name + ".mp3")

            # Option 1: Use constant bit rate (CBR) at 320 kbps for highest quality
            command = [
                "ffmpeg",
                "-i", flac_file,
                "-b:a", "320k",
                mp3_file
            ]

            # Option 2: Alternatively, for highest quality VBR,
            # uncomment the next lines and comment out the above command.
            #
            # command = [
            #     "ffmpeg",
            #     "-i", flac_file,
            #     "-qscale:a", "0",
            #     mp3_file
            # ]

            print(f"Converting {flac_file} to {mp3_file}...")

            try:
                subprocess.run(command, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error converting {flac_file}: {e}")

        print("Conversion complete!")


SOURCE_DIR = "src/audio_dataset_flac"
flac_converter = FlacConvert(SOURCE_DIR)

flac_converter.convert_files()
