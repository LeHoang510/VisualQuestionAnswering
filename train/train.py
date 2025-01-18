import os.path as osp
import json
import sys

BASE_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(BASE_DIR)

from data_preprocess.utils import *
from data_preprocess.vqa_dataset import VQADataset, VQATransform

if __name__ == "__main__":
    with open(osp.join("dataset", "generated_yes_no", "train_dataset.json"), "r") as f:
        dataset = json.load(f)
    print(f"Dataset length: {len(dataset)}")

    vocab = build_vocab(dataset)
    mapping = mapping_classes(dataset)

    print(mapping)
    print(vocab)

    # train_dataset = VQADataset(data=dataset, 
    #                            vocab=vocab,
    #                            mapping=mapping,
    #                            transform=VQATransform().get_transform("train"))

