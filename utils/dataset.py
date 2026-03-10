import glob
import os
import random

from torch.utils.data import Dataset

from options.config import *


class ImageSet(Dataset):
    def __init__(self, root, model, transforms=None):
        super(ImageSet, self).__init__()

        self.root_A = os.path.join(root, model, "A/*")
        self.root_B = os.path.join(root, model, "B/*")
        self.transform = transforms

        self.list_A = glob.glob(self.root_A)
        self.list_B = glob.glob(self.root_B)

    def __getitem__(self, index):
        img_path_A = self.list_A[index % len(self.list_A)]
        img_path_B = random.choice(self.list_B)

        img_A = Image.open(img_path_A).convert("RGB")
        img_B = Image.open(img_path_B).convert("RGB")

        item_A = self.transform(img_A)
        item_B = self.transform(img_B)

        return {"A": item_A, "B": item_B}

    def __len__(self):
        return max(len(self.list_A), len(self.list_B))