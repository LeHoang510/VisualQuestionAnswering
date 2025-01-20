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
from model.vqa_dataset import VQADatasetAdvance, VQATransform
from model.VQAVLLM import VQAVLLM
from tqdm import tqdm

def evaluate(model, dataloader, criterion):
    model.eval()
    correct = 0
    total = 0
    losses = []
    with torch.no_grad():
        for img, question, label in tqdm(dataloader, desc="Validation"):
            outputs = model(img, question)
            print(outputs)
            print(label)
            exit(0)
            loss = criterion(outputs, label)
            losses.append(loss.item())
            _, predicted = torch.max(outputs.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
    
    loss = sum(losses) / len(losses)
    acc = correct / total

    return loss, acc

def train():
    set_seed()

    config = load_yaml(osp.join("utils", "train_config.yaml"))

    val_batch_size = config["val"]["batch_size"]
    val_shuffle = config["val"]["shuffle"]

    with open(osp.join("dataset", "generated_yes_no", "train_dataset.json"), "r") as f:
        dataset = json.load(f)
    print(f"Dataset length: {len(dataset)}")

    mapping = mapping_classes(dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    print("Device:", device)

    val_dataset = VQADatasetAdvance(data=dataset,
                                    mapping=mapping,
                                    device=device,
                                    transform=VQATransform().get_transform("advance"))
    
    val_loader = DataLoader(val_dataset,
                            batch_size=val_batch_size,
                            shuffle=val_shuffle)
    
    model = VQAVLLM(device)

    criterion = nn.CrossEntropyLoss()
    val_loss, val_acc = evaluate(model, val_loader, criterion)

    print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_acc}")


if __name__ == "__main__":
    train()