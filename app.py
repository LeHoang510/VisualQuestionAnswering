import os.path as osp
import json

from model.utils import *
from model.vqa_dataset import VQADataset, VQATransform

if __name__ == "__main__":
    with open(osp.join("dataset", "generated_yes_no", "train_dataset.json"), "r") as f:
        dataset = json.load(f)
    print(f"Dataset length: {len(dataset)}")

    # with open(osp.join("dataset", "generated", "val_dataset.json"), "r") as f:
    #     dataset = json.load(f)
    # print(f"Dataset length: {len(dataset)}")