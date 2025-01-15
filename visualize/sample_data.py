import random
import json
import os
import os.path as osp
import sys

from PIL import Image
import matplotlib.pyplot as plt

BASE_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(BASE_DIR)

import random
import json
from PIL import Image
import matplotlib.pyplot as plt
import os

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def display_entry(ax, entry):
    image_path = entry["image_path"]
    image_id = entry["id"]
    question = entry["question"]
    answer = entry["answer"]

    if os.path.exists(image_path):
        img = Image.open(image_path)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"ID: {image_id}\nQuestion: {question}\nAnswer: {answer}", fontsize=10)
    else:
        ax.set_title(f"Image not found", fontsize=10)

def visualize_random_entry(data):
    random_entry = random.choice(data)
    fig, ax = plt.subplots()
    display_entry(ax, random_entry)
    plt.show()

def visualize_entry_by_id(data, image_id):
    entry = next((item for item in data if item["id"] == image_id), None)
    if entry:
        fig, ax = plt.subplots()
        display_entry(ax, entry)
        plt.show()
    else:
        print(f"No entry found with ID: {image_id}")

def visualize_random_n_entries(data, n):
    random_entries = random.sample(data, min(n, len(data)))
    fig, axes = plt.subplots(1, n, figsize=(15, 5))  # Display images in a row
    for i, entry in enumerate(random_entries):
        display_entry(axes[i], entry)
    plt.show()

def visualize_n_elements_in_one_frame(data, n):
    random_entries = random.sample(data, min(n, len(data)))
    rows = (n // 3) + (n % 3 > 0) 
    fig, axes = plt.subplots(rows, 3, figsize=(15, 5 * rows))
    axes = axes.flatten()

    for i, entry in enumerate(random_entries):
        display_entry(axes[i], entry)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    dataset_path = osp.join("dataset", "generated", "val_dataset.json")
    data = load_json(dataset_path)

    visualize_random_entry(data)
    # specific_id = 458752000
    # visualize_entry_by_id(data, specific_id)
    # visualize_random_n_entries(data, 3)
    # visualize_n_elements_in_one_frame(data, 6)
