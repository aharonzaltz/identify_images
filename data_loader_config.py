from config import path_to_training_data, path_to_validation_data, batch_size
from dataloader import CustomDataset
from torch.utils.data import DataLoader

training_ds = CustomDataset(path_to_training_data)
validation_ds = CustomDataset(path_to_validation_data)

training_dataloader = DataLoader(training_ds, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(validation_ds, batch_size=batch_size)
