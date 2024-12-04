from torchvision import transforms, datasets
from torch.utils.data import Dataset

import medmnist
from medmnist import INFO

import os
from torch.nn import functional as F
import torch.nn as nn

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

class LeNet5(nn.Module):
   def __init__(self):
       super().__init__()
       self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
       self.conv2 = nn.Conv2d(6, 16, 5)
       self.fc1 = nn.Linear(16*5*5, 120)
       self.fc2 = nn.Linear(120, 84)
       self.fc3 = nn.Linear(84, 10)
   def forward(self, x):
       x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
       x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
       x = x.view(-1, self.num_flat_features(x))
       x = F.relu(self.fc1(x))
       x = F.relu(self.fc2(x))
       x = self.fc3(x)
       return x
   def num_flat_features(self, x):
       size = x.size()[1:]
       num_features = 1
       for s in size:
           num_features *= s
       return num_features

def load_data(args):

    trans = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[.5], std=[.5])])

    if args.dataset in ["CIFAR10", "MNIST", "FaMNIST","SVHN","ImageNet"]:
        data_path = '../dataset/'
        if not os.path.exists(data_path):
            os.makedirs(data_path)

        if args.dataset == "CIFAR10":
            dataset_train = datasets.CIFAR10(data_path, train=True, download=True,  transform=trans)
            dataset_test = datasets.CIFAR10(data_path, train=False, download=True, transform=trans)
        elif args.dataset == "MNIST":
            dataset_train = datasets.MNIST(data_path, train=True, download=True,  transform=trans)
            dataset_test = datasets.MNIST(data_path, train=False, download=True, transform=trans)
        elif args.dataset == "FaMNIST":
            dataset_train = datasets.FashionMNIST(data_path, train=True, download=True,  transform=trans)
            dataset_test = datasets.FashionMNIST(data_path, train=False, download=True, transform=trans)
        elif args.dataset == "SVHN":
            dataset_train = datasets.SVHN(data_path, split='train', download=True,  transform=trans)
            dataset_test = datasets.SVHN(data_path, split='test', download=True, transform=trans)
        elif args.dataset == "ImageNet":
            data_path_val = "/home/syjiang/Datasets/ImageNet12/"
            dataset_train = datasets.ImageNet(data_path_val, split= "train", transform=trans)
            dataset_test = datasets.ImageNet(data_path_val, split= "val", transform=trans)

        dataset_val = []
    else:
        info = INFO[args.dataset]
        DataClass = getattr(medmnist, info['python_class'])
        dataset_train = DataClass(split='train', transform=trans, download=True, as_rgb=True)
        dataset_test = DataClass(split='test', transform=trans, download=True, as_rgb=True)
        dataset_val = DataClass(split='val', transform=trans, download=True, as_rgb=True)
    
    return dataset_train, dataset_test, dataset_val


