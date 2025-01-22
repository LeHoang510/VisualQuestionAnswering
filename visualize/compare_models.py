import os
import os.path as osp
import sys
import json

BASE_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(BASE_DIR)

import random
import matplotlib.pyplot as plt
from PIL import Image
import torch

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def choose_random_element(dataset):
    return random.choice(dataset)

def find_element_by_id(dataset, element_id):
    return next((el for el in dataset if el["id"] == element_id), None)

def plot_predictions(element, models, model_names):
    image = Image.open(element["image_path"])
    
    element_id = element["id"]
    question = element["question"]
    label = element["answer"]  
    
    #TODO => get predictions from models
    predictions = ["no", "yes", "no"]
    
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    ax[0].imshow(image)
    ax[0].axis("off")
    ax[0].set_title("Image")
    ax[0].text(0.5, -0.2, f"ID: {element_id}\nQuestion: {question}\nLabel: {label}",
               ha="center", va="center", transform=ax[0].transAxes, fontsize=10)
    
    ax[1].axis("off")
    ax[1].set_title("Model Predictions")
    text = "\n".join([f"{name}: {pred}" for name, pred in zip(model_names, predictions)])
    ax[1].text(0.5, 0.5, text, ha="center", va="center", fontsize=12)
    
    plt.tight_layout()
    plt.show()

def plot_multiple_predictions(elements, models, model_names):
    #TODO 
    n = len(elements)  # Number of images to plot
    
    # Create a figure with n rows and 2 columns
    fig, axs = plt.subplots(n, 2, figsize=(10, 5 * n))
    
    # If n = 1, axs is not a list, so wrap it
    if n == 1:
        axs = [axs]
    
    for i, (element, ax_row) in enumerate(zip(elements, axs)):
        image_ax, text_ax = ax_row  # Split the row into two axes: image and text
        
        # Load image
        image = Image.open(element["image_path"])
        
        # Extract details
        element_id = element["id"]
        question = element["question"]
        label = element.get("answer", "N/A")  # Use "N/A" if the answer is not available
        
        # Generate predictions
        predictions = []
        for model in models:
            model.eval()
            with torch.no_grad():
                pred = model.predict(image, question)
            predictions.append(pred)
        
        # Plot image and details
        image_ax.imshow(image)
        image_ax.axis("off")
        image_ax.set_title(f"Image {i + 1}")
        image_ax.text(0.5, -0.2, f"ID: {element_id}\nQuestion: {question}\nLabel: {label}",
                      ha="center", va="center", transform=image_ax.transAxes, fontsize=10)
        
        # Plot predictions
        text_ax.axis("off")
        text_ax.set_title("Model Predictions")
        text = "\n".join([f"{name}: {pred}" for name, pred in zip(model_names, predictions)])
        text_ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=12)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    val_path = osp.join("dataset", "generated_yes_no", "val_dataset.json")
    val_dataset = load_json(val_path)

    test_path = osp.join("dataset", "generated_yes_no", "test_dataset.json")
    test_dataset = load_json(test_path)

    #TODO => load model
    model1 = torch.load("model1.pth")  
    model2 = torch.load("model2.pth")
    model3 = torch.load("model3.pth")

    models = [model1, model2, model3]
    model_names = ["Basic", "Advance", "VLLM"]
    
    random_element = choose_random_element(val_dataset)
    plot_predictions(random_element, models, model_names)
    
    element_id = 393225001
    element = find_element_by_id(val_dataset, element_id)
    if element:
        plot_predictions(element, models, model_names)
    else:
        print(f"Element with ID {element_id} not found.")

    
    num_images = 3
    
    random_elements = random.sample(val_dataset, num_images)
    plot_multiple_predictions(random_elements, models, model_names)