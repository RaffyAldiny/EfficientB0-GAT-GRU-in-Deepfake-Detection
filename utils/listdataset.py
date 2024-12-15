import os

# Define paths for your directories
input_file_path = 'data/List_of_testing_videos.txt'
celebrity_real_path = 'data/preprocessed/Celeb-real'
celebrity_synthesis_path = 'data/preprocessed/Celeb-synthesis'
youtube_real_path = 'data/preprocessed/Youtube-real'

# Create the necessary directories if they don't exist
os.makedirs(celebrity_real_path, exist_ok=True)
os.makedirs(celebrity_synthesis_path, exist_ok=True)
os.makedirs(youtube_real_path, exist_ok=True)

def collect_folders():
    """
    Collects all folder names from Celeb-real, Celeb-synthesis, and Youtube-real directories
    and writes them to List_of_testing_videos.txt with appropriate labels.
    """
    try:
        # Initialize lists for folder paths
        real_folders = []
        synthesis_folders = []
        youtube_folders = []

        # Collect all folder names in Celeb-real
        if os.path.exists(celebrity_real_path):
            real_folders = [
                os.path.join("Celeb-real", folder).replace("\\", "/")
                for folder in os.listdir(celebrity_real_path)
                if os.path.isdir(os.path.join(celebrity_real_path, folder))
            ]

        # Collect all folder names in Celeb-synthesis
        if os.path.exists(celebrity_synthesis_path):
            synthesis_folders = [
                os.path.join("Celeb-synthesis", folder).replace("\\", "/")
                for folder in os.listdir(celebrity_synthesis_path)
                if os.path.isdir(os.path.join(celebrity_synthesis_path, folder))
            ]

        # Collect all folder names in Youtube-real
        if os.path.exists(youtube_real_path):
            youtube_folders = [
                os.path.join("Youtube-real", folder).replace("\\", "/")
                for folder in os.listdir(youtube_real_path)
                if os.path.isdir(os.path.join(youtube_real_path, folder))
            ]

        # Write to List_of_testing_videos.txt
        with open(input_file_path, 'w') as output_file:
            for real_folder in real_folders:
                output_file.write(f"1 {real_folder}\n")
            for synthesis_folder in synthesis_folders:
                output_file.write(f"0 {synthesis_folder}\n")
            for youtube_folder in youtube_folders:
                output_file.write(f"1 {youtube_folder}\n")

        print(f"Successfully updated {input_file_path} with folder paths.")

    except Exception as e:
        print(f"An error occurred: {e}")

# Main function to run the script
def main():
    collect_folders()

# Run the script
if __name__ == "__main__":
    main()
