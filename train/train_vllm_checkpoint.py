import os
import json
import torch
from tqdm import tqdm
import os.path as osp
import sys

BASE_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(BASE_DIR)

from utils.utils import set_seed
from model.utils import mapping_classes
from model.VQAVLLM import VQAVLLM


def save_checkpoint(checkpoint_path, progress):
    with open(checkpoint_path, "w") as f:
        json.dump(progress, f)
    print(f"Checkpoint saved to {checkpoint_path}")

def load_checkpoint(checkpoint_path):
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            progress = json.load(f)
        print(f"Checkpoint loaded from {checkpoint_path}")
        return progress
    return {"processed_indices": [], "losses": [], "correct": 0, "total": 0}

def evaluate(model, dataset, label2idx, checkpoint_path):
    progress = load_checkpoint(checkpoint_path)
    processed_indices = set(progress["processed_indices"])
    losses = progress["losses"]
    correct = progress["correct"]
    total = progress["total"]

    with torch.no_grad():
        for idx, sample in enumerate(tqdm(dataset, desc="Validation")):
            if idx in processed_indices:
                continue 

            img = sample["image_path"]
            question = sample["question"]
            label = sample["answer"].lower()

            try:
                output = model.predict(img, question).lower()
                output = label2idx[output]
                label = label2idx[label]

                loss = -(label * torch.log(torch.tensor(output)) + (1 - label) * torch.log(torch.tensor(1 - output)))
                losses.append(loss.item())
                total += 1
                correct += 1 if output == label else 0

                progress["processed_indices"].append(idx)
                progress["losses"] = losses
                progress["correct"] = correct
                progress["total"] = total

                save_checkpoint(checkpoint_path, progress)

            except Exception as e:
                print(f"Error processing sample {idx}: {e}")

    loss = sum(losses) / len(losses) if losses else float("inf")
    acc = correct / total if total > 0 else 0

    return loss, acc

def train():
    set_seed()
    checkpoint_path = os.path.join("output", "vllm_checkpoint.json")

    with open(os.path.join("dataset", "generated_yes_no", "val_dataset.json"), "r") as f:
        dataset = json.load(f)
    print(f"Dataset length: {len(dataset)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    mapping = mapping_classes(dataset)

    torch.cuda.empty_cache()
    model = VQAVLLM(device)

    val_loss, val_acc = evaluate(model, dataset, mapping[1], checkpoint_path)

    print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_acc}")

    # Clean up checkpoint after successful completion
    if osp.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"Checkpoint file {checkpoint_path} removed after completion.")

if __name__ == "__main__":
    train()
