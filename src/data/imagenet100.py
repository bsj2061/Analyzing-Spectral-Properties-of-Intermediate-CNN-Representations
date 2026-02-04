import os
import numpy as np
import torchvision
from torch.utils.data import ConcatDataset
from .base import BaseDataset


class ImageNet100(BaseDataset):
    def __init__(self, root, split="train", train_dirs = ['train.X1', 'train.X2', 'train.X3', 'train.X4'], val_dirs = ['val.X'],transform=None):
        """
        Args:
            root (str): ImageNet100 root path
            split (str): 'train' or 'val'
            transform: torchvision transform
        """
        self.root = root
        self.split = split
        self.train_dirs = train_dirs
        self.val_dirs = val_dirs
        self.transform = transform

        self.class_to_idx = self._build_class_index()
        self.dataset = self._build_concat_dataset()

    def _build_class_index(self):
        all_classes = set()
        dirs = self.train_dirs + self.val_dirs
        
        for d in dirs:
            path = os.path.join(self.root, d)
            all_classes.update(
                sub for sub in os.listdir(path)
                if os.path.isdir(os.path.join(path, sub))
            )
        return {cls: i for i, cls in enumerate(sorted(all_classes))}

    def _build_concat_dataset(self):
        if self.split == "train":
            dir_list = self.train_dirs
        elif self.split == "val":
            dir_list = self.val_dirs
        else:
            raise ValueError(f"Unknown split: {self.split}")

        datasets = []
        for d in dir_list:
            path = os.path.join(self.root, d)
            ds = torchvision.datasets.ImageFolder(
                root=path,
                transform=self.transform
            )

            ds.class_to_idx = self.class_to_idx
            ds.samples = [
                (s[0], self.class_to_idx[os.path.basename(os.path.dirname(s[0]))])
                for s in ds.samples
            ]
            ds.targets = [s[1] for s in ds.samples]

            datasets.append(ds)

        return ConcatDataset(datasets)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]