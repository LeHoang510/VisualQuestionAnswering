import os.path as osp
import os
import json
import sys

from torch.utils.data import DataLoader
import torch
import torch.nn as nn

BASE_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(BASE_DIR)

from utils.utils import set_seed, load_yaml, Logger
from model.utils import *
from model.vqa_dataset import VQADatasetAdvance, VQATransform
from model.VQAModelAdvance import VQAModelAdvance
from tqdm import tqdm

def save_checkpoint(epoch, model, optimizer, scheduler, train_losses, val_losses, checkpoint_path):
    os.makedirs(osp.dirname(checkpoint_path), exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "train_losses": train_losses,
        "val_losses": val_losses,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch+1}")


def load_checkpoint(checkpoint_path, model, optimizer, scheduler):
    """Load training state from a checkpoint."""
    if osp.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        train_losses = checkpoint["train_losses"]
        val_losses = checkpoint["val_losses"]
        print(f"Checkpoint loaded. Resuming from epoch {start_epoch}.")
        return start_epoch, train_losses, val_losses
    else:
        print("No checkpoint found. Starting from scratch.")
        return 0, [], []

def evaluate(model, dataloader, criterion):
    model.eval()
    correct = 0
    total = 0
    losses = []
    with torch.no_grad():
        for img, question, label in tqdm(dataloader, desc="Validation"):
            outputs = model(img, question)
            loss = criterion(outputs, label)
            losses.append(loss.item())
            _, predicted = torch.max(outputs.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
    
    loss = sum(losses) / len(losses)
    acc = correct / total

    return loss, acc

def fit(model, 
        train_loader, 
        val_loader, 
        criterion, 
        optimizer, 
        scheduler, 
        epochs,
        logger,
        validation=False,
        start_epoch=0,
        checkpoint_path=osp.join("output", "advance_checkpoint.pth"),
        train_losses=None,
        val_losses=None):
    
    if train_losses is None:
        train_losses = []
    if val_losses is None:
        val_losses = []

    for epoch in range(start_epoch, epochs):
        batch_train_losses = []
        model.train()

        for i, (imgs, questions, labels) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            outputs = model(imgs, questions)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_train_losses.append(loss.item())
        
        train_loss = sum(batch_train_losses) / len(batch_train_losses)
        train_losses.append(train_loss)

        val_loss, val_acc = None, None
        if validation:
            val_loss, val_acc = evaluate(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
            )
            val_losses.append(val_loss)
        
        logger.write_dict(epoch+1, epochs, train_loss, val_loss, val_acc)
        scheduler.step()

        save_checkpoint(epoch, model, optimizer, scheduler, train_losses, val_losses, checkpoint_path)
    
    return train_losses, val_losses

def train():
    set_seed()

    config = load_yaml(osp.join("utils", "train_config.yaml"))
    checkpoint_path = osp.join("output", "advance_checkpoint.pth")

    train_batch_size = config["train"]["batch_size"]
    train_shuffle = config["train"]["shuffle"]
    train_workers = config["train"]["workers"]

    val_batch_size = config["val"]["batch_size"]
    val_shuffle = config["val"]["shuffle"]
    val_workers = config["val"]["workers"]

    lr = config["lr"]
    epochs = config["num_epochs"]

    with open(osp.join("dataset", "generated_yes_no", "train_dataset.json"), "r") as f:
        train_dataset = json.load(f)
    print(f"Dataset length: {len(train_dataset)}")

    with open(osp.join("dataset", "generated_yes_no", "val_dataset.json"), "r") as f:
        val_dataset = json.load(f)
    print(f"Dataset length: {len(val_dataset)}")

    mapping = mapping_classes(train_dataset)
    print(mapping)
    exit(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    print("Device:", device)

    train_dataset = VQADatasetAdvance(data=train_dataset,
                                      mapping=mapping,
                                      device=device,
                                      transform=VQATransform().get_transform("advance"))

    val_dataset = VQADatasetAdvance(data=val_dataset,
                                    mapping=mapping,
                                    device=device,
                                    transform=VQATransform().get_transform("advance"))
    
    train_loader = DataLoader(train_dataset,
                              batch_size=train_batch_size,
                              shuffle=train_shuffle)
    
    val_loader = DataLoader(val_dataset,
                            batch_size=val_batch_size,
                            shuffle=val_shuffle)
    
    model = VQAModelAdvance(n_classes=len(mapping[0])).to(device)
    model.freeze()

    scheduler_step_size = epochs * 0.8
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=0.1)

    start_epoch, train_losses, val_losses = load_checkpoint(checkpoint_path, model, optimizer, scheduler)

    logger = Logger(model, scheduler, log_dir=osp.join("output", "advance_logs"))

    train_losses, val_losses = fit(model=model,
                                   train_loader=train_loader,
                                   val_loader=val_loader,
                                   criterion=criterion,
                                   optimizer=optimizer,
                                   scheduler=scheduler,
                                   epochs=epochs,
                                   logger=logger,
                                   checkpoint_path=checkpoint_path,
                                   start_epoch=start_epoch,
                                   train_losses=train_losses,
                                   val_losses=val_losses)
    
    val_loss, val_acc = evaluate(model, val_loader, criterion)

    print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_acc}")

    logger.close()
    torch.save(model.state_dict(), osp.join("output", "advance_model.pth"))
    print("Model saved")

    if osp.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"Checkpoint file {checkpoint_path} removed after completion.")

if __name__ == "__main__":
    train()