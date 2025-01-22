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
from model.VQAModelBasic import VQAModelBasic
from model.VQAModelAdvance import VQAModelAdvance
from model.VQAVLLM import VQAVLLM

from model.utils import build_vocab, mapping_classes
from model.vqa_dataset import VQADatasetBasic, VQADatasetAdvance, VQATransform
from torch.utils.data import DataLoader

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def choose_random_element(dataset):
    return random.choice(dataset)

def find_element_by_id(dataset, element_id):
    return next((el for el in dataset if el["id"] == element_id), None)

def load_model(device):
    train_path = osp.join("dataset", "generated_yes_no", "train_dataset.json")
    train_dataset = load_json(train_path)

    vocab = build_vocab(train_dataset)
    mapping = mapping_classes(train_dataset)

    basic_model = VQAModelBasic(
        n_classes=len(mapping[0]),
        vocab=vocab
    ).to(device)

    advance_model = VQAModelAdvance(
        n_classes=len(mapping[0])
    ).to(device)

    vllm_model = VQAVLLM(device)

    return basic_model, advance_model, vllm_model

def predict(models, input, device):
    train_path = osp.join("dataset", "generated_yes_no", "train_dataset.json")
    train_dataset = load_json(train_path)

    vocab = build_vocab(train_dataset)
    mapping = mapping_classes(train_dataset)

    basic_dataset = VQADatasetBasic(data=[input],
                                    vocab=vocab,
                                    mapping=mapping,
                                    transform=VQATransform().get_transform("basic","train"))
    
    advance_dataset = VQADatasetAdvance(data=[input],
                                        mapping=mapping,
                                        device=device,
                                        transform=VQATransform().get_transform("advance"))
    
    basic_loader = DataLoader(basic_dataset,
                            batch_size=1,
                            shuffle=False)
    
    advance_loader = DataLoader(advance_dataset,
                            batch_size=1,
                            shuffle=False)
    
    basic_model, advance_model, vllm_model = models

    for img, question, label in basic_loader:
        basic_model.eval()
        with torch.no_grad():
            img = img.to(device)
            question = question.to(device)
            basic_output = basic_model(img, question)
            _, predicted = torch.max(basic_output.data, 1)
    
    print("Basic Model Prediction:", predicted)
    
    for img, question, label in advance_loader:
        advance_model.eval()
        with torch.no_grad():
            img = img.to(device)
            question = question.to(device)
            advance_output = advance_model(img, question)
            _, predicted = torch.max(advance_output.data, 1)

    print("Advance Model Prediction:", predicted)

    vllm_output = vllm_model.predict(input["image_path"], input["question"]).lower()
    print("VLLM Model Prediction:", vllm_output)
    # Check the output
    return basic_output, advance_output, vllm_output


def plot_predictions(element, models, model_names, device):
    image = Image.open(element["image_path"])
    
    element_id = element["id"]
    question = element["question"]
    label = element["answer"]  
    
    predictions = predict(models, element, device)
    
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

def plot_multiple_predictions(elements, models, model_names, device):
    n = len(elements) 
    
    fig, axs = plt.subplots(n, 2, figsize=(10, 5 * n))
    
    if n == 1:
        axs = [axs]
    
    for i, (element, ax_row) in enumerate(zip(elements, axs)):
        image_ax, text_ax = ax_row  
        
        image = Image.open(element["image_path"])
        
        element_id = element["id"]
        question = element["question"]
        label = element.get("answer", "N/A")
        
        predictions = predict(models, element, device)
        
        image_ax.imshow(image)
        image_ax.axis("off")
        image_ax.set_title(f"Image {i + 1}")
        image_ax.text(0.5, -0.2, f"ID: {element_id}\nQuestion: {question}\nLabel: {label}",
                      ha="center", va="center", transform=image_ax.transAxes, fontsize=10)
        
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = load_model(device)
    model_names = ["Basic", "Advance", "VLLM"]
    
    random_element = choose_random_element(val_dataset)
    plot_predictions(random_element, models, model_names, device)
    
    element_id = 393225001
    element = find_element_by_id(val_dataset, element_id)
    if element:
        plot_predictions(element, models, model_names, device)
    else:
        print(f"Element with ID {element_id} not found.")

    
    num_images = 3
    
    random_elements = random.sample(val_dataset, num_images)
    plot_multiple_predictions(random_elements, models, model_names, device)