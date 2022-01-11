from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from torchvision import datasets, transforms
from PIL import Image
import glob

# class name from file it can be animal or class or CAPTCHA
from config import class_name, number_of_classes


class CustomDataset(Dataset):
    def __init__(self, path, transform=False, isFullPath=False):
        self.transform = transform
        if not isFullPath:
            path = path + '/*.png'

        self.filelist = glob.glob(path)
        self.labels = np.zeros(len(self.filelist))  # load the labels (copy from the notebook)

        for class_i in range(number_of_classes):
            files_that_are_of_this_class = [class_name + str(class_i) in file_name for file_name in self.filelist]
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
        # print ("tensor", x.resize_(14700), y)
        return x.resize_(4761), y

