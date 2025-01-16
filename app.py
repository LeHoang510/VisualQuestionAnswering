import os.path as osp
import json

from data_preprocess.utils import *

if __name__ == "main":
    with open(osp.join("dataset", "yes_no_dataset.json"), "r") as f:
        dataset = json.load(f)
    print(f"Dataset length: {len(dataset)}")

