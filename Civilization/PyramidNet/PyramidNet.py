#!/home/atheist8E/anaconda3/bin/python


#Basic Modules
import os
import sys
import argparse
import numpy as np
from datetime import datetime
from tensorboardX import SummaryWriter

#Pytorch
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.utils as utils

#Custom Modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Shrine.architecture import *
from Shrine.gpu import *
from Shrine.dataset import *
from Shrine.report import *


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_path", type = str, default = "/dev/disk/datasets/ImageNet/Subset/train")
    parser.add_argument("--validation_path", type = str, default = "/dev/disk/datasets/ImageNet/Subset/val")
    parser.add_argument("--gpu", type = int, default = 0)
    parser.add_argument("--batch_size", type = int, default = 128)
    parser.add_argument("--epoch", type = int, default = 100)
    parser.add_argument("--learning_rate", type = float, default = 0.0001)
    return parser.parse_args()


if __name__ == "__main__":
    args = set_args()
    select_gpu(str(args.gpu))
    writer = SummaryWriter(logdir = "../Alexandria", comment="_MNasNet_batch_size_{}_epoch_{}_time_{}".format(args.batch_size,args.epoch,datetime.now()))
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225])

    ])
    test_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225])
    ])
    train_loader, test_loader = ImageNet(args, train_transform, test_transform)
    model = MNasNet(args, writer).to(torch.device("cuda"))
    criterion = nn.CrossEntropyLoss()
    #optimizer = torch.optim.SGD(model.parameters(), lr = args.learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate, weight_decay = 0.01)
    #optimizer = torch.optim.RMSprop(model.parameters(), lr = args.learning_rate, weight_decay = 0.1, momentum = 0.1)
    model.fit(train_loader, test_loader, criterion, optimizer, num_epochs = args.epoch)
    writer = model.report()
    writer.close()
