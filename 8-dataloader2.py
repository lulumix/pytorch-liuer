import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

train_dataset = datasets.MNIST(root='./dataset/mnist2', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./dataset/mnist2', train=False, transform=transforms.ToTensor(), download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)
