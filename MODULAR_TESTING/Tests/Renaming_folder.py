import os

def rename_folders(directory):
    # Get all folder names
    folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
    
    # Sort to ensure a consistent order
    folders.sort()

    # First, rename folders to temporary names to avoid conflicts
    temp_names = {}
    for index, folder in enumerate(folders):
        temp_name = f"temp_{index}"
        temp_path = os.path.join(directory, temp_name)
        old_path = os.path.join(directory, folder)
        os.rename(old_path, temp_path)
        temp_names[temp_name] = folder  # Store original names
    
    # Rename from temporary names to final numbers
    for index, temp_name in enumerate(temp_names.keys()):
        final_path = os.path.join(directory, str(index))
        temp_path = os.path.join(directory, temp_name)
        os.rename(temp_path, final_path)
        print(f'Renamed "{temp_names[temp_name]}" -> "{index}"')

# Example usage
directory_path =  r"C:\Users\Bennett\Documents\WORKING_THESIS\Dataset Sample\Standing"
rename_folders(directory_path)
