#Basic Module
import os
import sys

#Pytorch
import torch
import torchvision
import torchvision.transforms as transforms


def CIFAR10(train_transform, test_transform, batch_size = 100):
    train_dataset = torchvision.datasets.CIFAR10(root = "../Data/", train = True, transform = train_transform, download = True)
    test_dataset = torchvision.datasets.CIFAR10(root = "../Data/", train = False, transform = test_transform)
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
    test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False)
    return train_loader, test_loader

def CIFAR100(train_transform, test_transform, batch_size = 100):
    train_dataset = torchvision.datasets.CIFAR100(root = "../Data/", train = True, transform = train_transform, download = True)
    test_dataset = torchvision.datasets.CIFAR100(root = "../Data/", train = False, transform = test_transform)
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
    test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False)
    return train_loader, test_loader

def ImageNet(args, train_transform, test_transform):
    train_dataset = torchvision.datasets.ImageFolder(root = args.training_path, transform = train_transform)
    test_dataset = torchvision.datasets.ImageFolder(root = args.validation_path, transform = test_transform)
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 4, pin_memory = True)
    test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 4, pin_memory = True)
    return train_loader, test_loader
