from PIL import Image, ImageFile
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
from torch.utils.data import Dataset, DataLoader

from config import batch_size, net_file_name
from dataloader import CustomDataset
from model import Net
from test_items import compute_accuracy_item

ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_item(url):
    img = Image.open(url)
    transforms.ToTensor()(img)
    x = transforms.ToTensor()(img).view(-1)
    return x.resize_(4761), 1

def test_animal(url):
    net = Net()

    net.load_state_dict(torch.load('trained_animals_model.pt'))
    net.eval()
    item = CustomDataset(url, False, True)
    image_item = DataLoader(item, batch_size=batch_size, shuffle=True)
    answer = compute_accuracy_item(image_item, net)[0].item()
    return answer