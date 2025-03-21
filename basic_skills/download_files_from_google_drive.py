from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os

# Authenticate and create the PyDrive client
gauth = GoogleAuth()
gauth.LocalWebserverAuth()  # Open a web browser for authentication
drive = GoogleDrive(gauth)

# Folder ID from the Google Drive link
folder_id = "1HYUZBMvTiLvQPQiBeCT0nKyQ2P3oTeOh"

# List all files in the folder
file_list = drive.ListFile({'q': f"'{folder_id}' in parents and trashed=false"}).GetList()

# Create a local folder to save the downloaded files
os.makedirs("downloaded_files", exist_ok=True)

# Download each file in the folder
for file in file_list:
    print(f"Downloading {file['title']}...")
    file.GetContentFile(os.path.join("downloaded_files", file['title']))
    print(f"Downloaded {file['title']} successfully.")

print("All files downloaded!")