"""
These are all utility methods used to alter files.
Whether it be turning M4As to FLACs, FLACs to MP3 or
flattening a directory. There's not much really.
"""
import os
import glob
import shutil
import subprocess


class FlacConvert:
    """
    Object in charge of converting FLAC files to MP3s

    Args:
        source_dir : The directory which contains all of the FLAC files which will be converted.
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


def flatten_directory(root_dir : str):
    """Flattens a directory so that all files are stored on the top level.
    Subdirectories will be removed so be careful when running!

    Args:
        root_dir (str): The root directory to be flattened.
    """
    for dirpath, _, filenames in os.walk(root_dir, topdown=False):
        for file in filenames:
            src_path = os.path.join(dirpath, file)
            dest_path = os.path.join(root_dir, file)

            # Handle duplicates
            if os.path.exists(dest_path):
                base, ext = os.path.splitext(file)
                counter = 1
                while os.path.exists(dest_path):
                    dest_path = os.path.join(root_dir, f"{base}_{counter}{ext}")
                    counter += 1

            shutil.move(src_path, dest_path)

    # Clean up empty directories
    for dirpath, dirnames, _ in os.walk(root_dir, topdown=False):
        for dirname in dirnames:
            full_path = os.path.join(dirpath, dirname)
            if not os.listdir(full_path):
                os.rmdir(full_path)

    print("Flattening complete!")


def convert_m4a_to_flac(input_dir : str, output_dir : str):
    """Copies a directory of M4As and creates a new directory of FLACs.

    Args:
        input_dir  : The original directory which contains the M4A files.
        output_dir : The path to which you want these files to be converted and saved.
    """

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through all .m4a files in the input directory
    for file in os.listdir(input_dir):
        if file.endswith(".m4a"):
            input_path = os.path.join(input_dir, file)
            output_path = os.path.join(output_dir, os.path.splitext(file)[0] + ".flac")

            # Run ffmpeg to convert the file
            subprocess.run(["ffmpeg", "-i", input_path, "-c:a", "flac", output_path], check=True)

    print("Conversion complete!")
