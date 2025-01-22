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

def evaluate(model, dataset, label2idx):
    correct = 0
    total = 0
    losses = []

    with torch.no_grad():
        for sample in tqdm(dataset, desc="Validation"):
            img = sample["image_path"]
            question = sample["question"]
            label = sample["answer"].lower()

            output = model.predict(img, question).lower()
            
            output = label2idx[output]
            label = label2idx[label]

            loss = -(label * torch.log(torch.tensor(output)) + (1 - label) * torch.log(torch.tensor(1 - output)))
            losses.append(loss.item())
            total += 1
            correct += 1 if output == label else 0
            
    loss = sum(losses) / len(losses)
    acc = correct / total

    return loss, acc

def train():
    set_seed()
    with open(osp.join("dataset", "generated_yes_no", "val_dataset.json"), "r") as f:
        dataset = json.load(f)
    print(f"Dataset length: {len(dataset)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    print("Device:", device)

    mapping = mapping_classes(dataset)

    torch.cuda.empty_cache()
    model = VQAVLLM(device)

    val_loss, val_acc = evaluate(model, dataset, mapping[1])

    print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_acc}")


if __name__ == "__main__":
    train()