import os.path as osp
import json
import sys

from torch.utils.data import DataLoader
import torch
import torch.nn as nn

BASE_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(BASE_DIR)

from utils.utils import set_seed, load_yaml, Logger
from model.utils import *
from model.vqa_dataset import VQADatasetBasic, VQATransform
from model.VQAModelBasic import VQAModelBasic
from tqdm import tqdm

def evaluate(model, dataloader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    losses = []
    with torch.no_grad():
        for img, question, label in tqdm(dataloader, desc="Validation"):
            img = img.to(device)
            question = question.to(device)
            label = label.to(device)
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
        device, 
        epochs,
        logger,
        validation=False):
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        batch_train_losses = []
        model.train()

        for i, (imgs, questions, labels) in enumerate(tqdm(train_loader)):
            imgs = imgs.to(device)
            questions = questions.to(device)
            labels = labels.to(device)

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
                device=device,
            )
            val_losses.append(val_loss)
        
        logger.write_dict(epoch, epochs, train_loss, val_loss, val_acc)
        scheduler.step()
    
    return train_losses


def train():
    set_seed()

    config = load_yaml(osp.join("utils", "train_config.yaml"))

    train_batch_size = config["train"]["batch_size"]
    train_shuffle = config["train"]["shuffle"]
    train_workers = config["train"]["workers"]

    val_batch_size = config["val"]["batch_size"]
    val_shuffle = config["val"]["shuffle"]
    val_workers = config["val"]["workers"]

    lr = config["lr"]
    epochs = config["num_epochs"]

    with open(osp.join("dataset", "generated_yes_no", "train_dataset.json"), "r") as f:
        dataset = json.load(f)
    print(f"Dataset length: {len(dataset)}")

    vocab = build_vocab(dataset)
    mapping = mapping_classes(dataset)

    train_dataset = VQADatasetBasic(data=dataset, 
                               vocab=vocab,
                               mapping=mapping,
                               transform=VQATransform().get_transform("basic","train"))
    
    val_dataset = VQADatasetBasic(data=dataset,
                             vocab=vocab,
                             mapping=mapping,
                             transform=VQATransform().get_transform("basic","val"))
    
    train_loader = DataLoader(train_dataset, 
                              batch_size=train_batch_size, 
                              num_workers=train_workers,
                              shuffle=train_shuffle)
    
    val_loader = DataLoader(val_dataset,
                            batch_size=val_batch_size,
                            num_workers=val_workers,
                            shuffle=val_shuffle)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    print("Device:", device)

    model = VQAModelBasic(
        n_classes=len(mapping[0]),
        vocab=vocab,
        img_model_name="resnet18",
        embedding_dim=128,
        n_layers=2,
        hidden_dim=256,
        dropout=0.2,
    ).to(device)
    
    scheduler_step_size = epochs * 0.8
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=0.1)

    logger = Logger(model, scheduler)

    train_losses = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=epochs,
        logger=logger
    )

    val_loss, val_acc = evaluate(
        model=model,
        dataloader=val_loader,
        criterion=criterion,
        device=device,
    )

    print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_acc}")

    logger.close()
    torch.save(model.state_dict(), osp.join("output", "basic_model.pth"))
    print("Model saved")


if __name__ == "__main__":
    train()