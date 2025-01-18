import os.path as osp
import json
import sys

from torch.utils.data import DataLoader

BASE_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(BASE_DIR)

from utils.utils import set_seed, load_yaml
from model.utils import *
from model.vqa_dataset import VQADataset, VQATransform

if __name__ == "__main__":
    set_seed()

    config = load_yaml(osp.join("utils", "dataset_config.yaml"))

    train_batch_size = config["train"]["batch_size"]
    train_shuffle = config["train"]["shuffle"]
    train_workers = config["train"]["workers"]

    val_batch_size = config["val"]["batch_size"]
    val_shuffle = config["val"]["shuffle"]
    val_workers = config["val"]["workers"]

    with open(osp.join("dataset", "generated_yes_no", "train_dataset.json"), "r") as f:
        dataset = json.load(f)
    print(f"Dataset length: {len(dataset)}")

    vocab = build_vocab(dataset)
    mapping = mapping_classes(dataset)

    train_dataset = VQADataset(data=dataset, 
                               vocab=vocab,
                               mapping=mapping,
                               transform=VQATransform().get_transform("train"))
    
    val_dataset = VQADataset(data=dataset,
                             vocab=vocab,
                             mapping=mapping,
                             transform=VQATransform().get_transform("val"))
    
    train_loader = DataLoader(train_dataset, 
                              batch_size=train_batch_size, 
                              num_workers=train_workers,
                              shuffle=train_shuffle)
    
    val_loader = DataLoader(val_dataset,
                            batch_size=val_batch_size,
                            num_workers=val_workers,
                            shuffle=val_shuffle)

