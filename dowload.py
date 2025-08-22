import requests
import os

def download_file(url, dest_path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded weights to {dest_path}")
    else:
        print(f"Failed to download file: {response.status_code}")

# Example: Replace with your actual URL
url = "https://drive.usercontent.google.com/download?id=1V5WYBB_Of9VGi8LMeUse8Rj8FRQpIIls&export=download&authuser=0&confirm=t&uuid=e2e2ba73-d771-4397-885c-17975dee6f89&at=AN8xHooogbQIAb6-BXL6cgVIWB0H%3A1755839609200"
dest_path = "vgg19_trained.pth"
download_file(url, dest_path)