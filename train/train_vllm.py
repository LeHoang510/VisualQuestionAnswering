import os.path as osp
import json
import sys

from torch.utils.data import DataLoader
import torch
import torch.nn as nn

BASE_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(BASE_DIR)

from utils.utils import set_seed, load_yaml
from model.utils import *
from model.VQAVLLM import VQAVLLM
from tqdm import tqdm

def evaluate(model, dataset):
    correct = 0
    total = 0
    losses = []

    with torch.no_grad():
        for sample in tqdm(dataset, desc="Validation"):
            img = sample["image_path"]
            question = sample["question"]
            label = sample["answer"]

            output = model.predict(img, question).lower()
            
            output = 1.0 if output == "yes" else 0.0
            label = 1 if label.lower() == "yes" else 0

            loss = -(label * torch.log(torch.tensor(output)) + (1 - label) * torch.log(torch.tensor(1 - output)))
            losses.append(loss.item())
            total += 1
            correct += 1 if output == label else 0
            
    loss = sum(losses) / len(losses)
    acc = correct / total

    return loss, acc

def train():
    set_seed()
    with open(osp.join("dataset", "generated_yes_no", "train_dataset.json"), "r") as f:
        dataset = json.load(f)
    print(f"Dataset length: {len(dataset)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    print("Device:", device)

    torch.cuda.empty_cache()
    model = VQAVLLM(device)

    val_loss, val_acc = evaluate(model, dataset)

    print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_acc}")


if __name__ == "__main__":
    train()