import glob
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


item = CustomDataset(path_to_validation_data + 'class9_4170.png', 10, False, True)
image_item = DataLoader(item, batch_size=300, shuffle=True)
print(compute_accuracy_item(image_item, net))

# print ( compute_accuracy_and_loss(training_dataloader,net))
# print ( compute_accuracy_and_loss(valid_dataloader,net))
