from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from torchvision import datasets, transforms
from PIL import Image
import glob


class CustomDataset(Dataset):
    def __init__(self, path, n_classes=10, transform=False, isFullPath=False):
        self.transform = transform
        if not isFullPath:
            path = path + '/*.png'

        self.filelist = glob.glob(path)
        self.labels = np.zeros(len(self.filelist))  # load the labels (copy from the notebook)

        for class_i in range(10):
            files_that_are_of_this_class = ['class' + str(class_i) in x for x in self.filelist]
            self.labels[files_that_are_of_this_class] = class_i

        self.labels = torch.LongTensor(self.labels)

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        img = Image.open(self.filelist[idx])

        transforms.ToTensor()(img)

        x = transforms.ToTensor()(img).view(-1)

        # x =  ....transform to tesnor and  flatten it to a vector of 69*69 = 4761 with .view(-1)

        y = self.labels[idx]

        return x, y

