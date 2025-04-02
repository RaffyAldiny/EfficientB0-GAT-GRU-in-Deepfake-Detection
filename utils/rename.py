import os

def rename_folders(base_dir):
    # Starting ID
    start_id = 63
    suffix = "_0000"
    
    # Get a sorted list of folders in the specified directory
    folders = sorted(os.listdir(base_dir))
    
    for index, folder in enumerate(folders):
        old_path = os.path.join(base_dir, folder)
        
        # Check if it's a directory
        if os.path.isdir(old_path):
            # Generate the new folder name
            new_name = f"id{start_id + index}{suffix}"
            new_path = os.path.join(base_dir, new_name)
            
            # Rename the folder
            os.rename(old_path, new_path)
            print(f"Renamed: {folder} -> {new_name}")

# Base directory path
base_dir = "data/preprocessed/YouTube-real"
rename_folders(base_dir)
