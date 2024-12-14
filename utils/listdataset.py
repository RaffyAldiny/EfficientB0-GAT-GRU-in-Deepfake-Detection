import os

# Define paths for your directories
input_file_path = 'data/List_of_testing_videos.txt'
celebrity_real_path = 'data/preprocessed/Celeb-real'
celebrity_synthesis_path = 'data/preprocessed/Celeb-synthesis'

# Create the necessary directories if they don't exist
os.makedirs(celebrity_real_path, exist_ok=True)
os.makedirs(celebrity_synthesis_path, exist_ok=True)

def collect_folders():
    """
    Collects all folder names from Celeb-real and Celeb-synthesis directories
    and writes them to List_of_testing_videos.txt with appropriate labels.
    """
    try:
        # Collect all folder names in Celeb-real and Celeb-synthesis
        real_folders = [
            os.path.join("Celeb-real", folder).replace("\\", "/")
            for folder in os.listdir(celebrity_real_path)
            if os.path.isdir(os.path.join(celebrity_real_path, folder))
        ]
        synthesis_folders = [
            os.path.join("Celeb-synthesis", folder).replace("\\", "/")
            for folder in os.listdir(celebrity_synthesis_path)
            if os.path.isdir(os.path.join(celebrity_synthesis_path, folder))
        ]

        # Write to List_of_testing_videos.txt
        with open(input_file_path, 'w') as output_file:
            for real_folder in real_folders:
                output_file.write(f"1 {real_folder}\n")
            for synthesis_folder in synthesis_folders:
                output_file.write(f"0 {synthesis_folder}\n")

        print(f"Successfully updated {input_file_path} with folder paths.")

    except Exception as e:
        print(f"An error occurred: {e}")

# Main function to run the script
def main():
    collect_folders()

# Run the script
if __name__ == "__main__":
    main()