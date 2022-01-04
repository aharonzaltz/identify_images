import glob

import torch
from torch.utils.data import Dataset, DataLoader
from dataloader import CustomDataset
import matplotlib

from test_items import compute_accuracy_item

matplotlib.use('TkAgg')
from model import Net


path_to_training_data = 'Dataset/train/'
path_to_validation_data = 'Dataset/validation/'

training_ds = CustomDataset(path_to_training_data)
validation_ds = CustomDataset(path_to_validation_data)

training_dataloader = DataLoader(training_ds, batch_size=300, shuffle=True)
valid_dataloader = DataLoader(validation_ds, batch_size=300)


net = Net()

net.load_state_dict(torch.load('trained_model.pt'))
net.eval()

filelist = glob.glob(path_to_training_data + '/*.png')
totalFiles = len(filelist)
correctAnswers = 0
for path in filelist:
    indexOfClassNum = [index for index, x in enumerate(path) if x.isdigit()][0]
    classNum = int(path[indexOfClassNum])
    item = CustomDataset(path, 10, False, True)
    image_item = DataLoader(item, batch_size=1, shuffle=True)
    answer = compute_accuracy_item(image_item, net)[0].item()
    # print (answer, classNum, answer == classNum)
    if answer == classNum:
        correctAnswers+=1
print (correctAnswers, totalFiles, float(correctAnswers)/totalFiles)

# print ( compute_accuracy_and_loss(training_dataloader,net))
# print ( compute_accuracy_and_loss(valid_dataloader,net))
