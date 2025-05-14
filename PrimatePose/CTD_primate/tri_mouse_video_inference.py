
import requests
import shutil
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile
import os

import deeplabcut
import deeplabcut.pose_estimation_pytorch as dlc_torch
import deeplabcut.utils.auxiliaryfunctions as auxiliaryfunctions
import matplotlib.pyplot as plt
import numpy as np


download_path = Path.cwd()
video_name = "videocompressed1.mp4"
video_path = str(download_path / "videos" /video_name)
print(f"Video will be saved in {video_path}")

print(f"Downloading the tri-mouse video into {download_path}")

url_video_record = "https://zenodo.org/api/records/7883589"
response = requests.get(url_video_record)
if response.status_code == 200:
    file = response.json()["files"][0]
    title = file["key"]
    print(f"Downloading {title}...")
    with requests.get(file['links']['self'], stream=True) as r:
        with ZipFile(BytesIO(r.content)) as zf:
            zf.extractall(path=download_path)
else:
    raise ValueError(f"The URL {url_video_record} could not be reached.")

# Check that the video was downloaded
src_video_path = download_path / "demo-me-2021-07-14" / "videos" / video_name
if not src_video_path.exists():
    raise ValueError("Failed to download the video")

# Move the video to the final path
shutil.move(src_video_path, video_path)
if not Path(video_path).exists():
    raise ValueError("Failed to move the video")


