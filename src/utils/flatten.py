"""Self Explanatory."""
import os
import shutil

def flatten_directory(root_dir : str):
    """Flattens a directory so that all files are stored on the top level. 
    Subdirectories will be removed so be careful when running!

    Args:
        root_dir (str): _description_
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

flatten_directory('/Users/nico/Desktop/complete_jorge')
