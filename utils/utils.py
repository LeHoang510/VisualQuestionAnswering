import random
import os.path as osp

import numpy as np
import torch

import yaml
from torch.utils.tensorboard import SummaryWriter

def set_seed(seed: int = 5555):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministc = True
	torch.backends.cudnn.benchmark = False

def load_yaml(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

class Logger:
    def __init__(self, model, scheduler, log_dir=osp.join("output", "logs")):
        self.model = model
        self.scheduler = scheduler
        self.writer = None
        self.log_dir = log_dir

    def write_dict(self, epoch, train_loss, val_loss, val_acc):
        if self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir)
        
        if val_loss is not None and val_acc is not None:
            self.writer.add_scalar("Loss/Train", train_loss, epoch)
            self.writer.add_scalar("Loss/Val", val_loss, epoch)
            self.writer.add_scalar("Accuracy/Val", val_acc, epoch)
            print(f"Epoch {epoch+1}/{epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}, Val Acc: {val_acc}")
        else:
            self.writer.add_scalar("Loss/Train", train_loss, epoch)
            print(f"Epoch {epoch+1}/{epoch}, Train Loss: {train_loss}")
    
    def close(self):
        self.writer.close()
