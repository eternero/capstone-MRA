import os
import subprocess

# Define source (input) and destination (output) directories
input_dir  = ""
output_dir = ""

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
