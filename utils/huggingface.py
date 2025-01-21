from datasets import Dataset, Features, Value, Image
from pathlib import Path
from huggingface_hub import create_repo
import os
import os.path as osp
import sys

BASE_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(BASE_DIR)

from huggingface_hub import login

# Login with your Hugging Face token
login(token="")

# Path to your images folder
# image_folder = osp.join("dataset", "val2014")

# # Create a list of image file paths
# image_files = list(Path(image_folder).glob("*.*"))  # Adjust pattern for specific extensions if needed

# # Create a dataset with image paths
# data = {
#     "image": [str(image) for image in image_files],  # Store image paths
#     "label": [None] * len(image_files)  # Optional: Add labels if applicable
# }

# # Define the features for the dataset
# features = Features({
#     "image": Image(),  # Automatically loads image data
#     "label": Value("string")  # Adjust type based on your data
# })

# # Create the dataset
# dataset = Dataset.from_dict(data, features=features)

# # Push to Hugging Face Hub
# repo_name = "Kindred510/vqa_image"  # Choose a name for your dataset
# create_repo(repo_name, repo_type="dataset", exist_ok=True)  # Create the repo

# # Upload the dataset to the hub
# dataset.push_to_hub(repo_name)
# print(f"Dataset uploaded to https://huggingface.co/datasets/{repo_name}")

from huggingface_hub import HfApi
import time
api = HfApi()


# Define your dataset repository and the folder you're uploading
repo_id = "Kindred510/vqa_image"  # Your Hugging Face dataset repository
subfolder = "val2014"  # Desired subfolder in the repo

# Path to your local folder of images
local_folder = "dataset/val2014"

# Loop through all image files in the folder
for filename in os.listdir(local_folder):
    local_filepath = os.path.join(local_folder, filename)
    
    # Only process files (not directories)
    if os.path.isfile(local_filepath):
        # Construct the remote path where the file will be uploaded
        remote_filepath = os.path.join(subfolder, filename)
        
        # Upload the file to Hugging Face
        api.upload_file(
            path_or_fileobj=local_filepath,  # Local path to the file
            path_in_repo=remote_filepath,    # Destination in the Hugging Face repo (subfolder)
            repo_id=repo_id,                 # Dataset repository
            repo_type="dataset"              # Type of repo (dataset)
        )
        print(f"Uploaded {filename} to {repo_id}/{subfolder}")

