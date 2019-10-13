#!/home/atheist8E/anaconda3/bin/python


#Basic Modules
import os
import sys
import argparse
import numpy as np
from datetime import datetime

#Pytorch
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.utils as utils
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR

#Custom Modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from Shrine.architecture import *
from Shrine.dataset import *
from Shrine.report import *


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",              type = str,     default = "IcarusNet_v5")
    parser.add_argument("--optimizer",          type = str,     default = "SGD")
    parser.add_argument("--learning_rate",      type = float,   default = 0.1)
    parser.add_argument("--teacher",            type = str,     default = "ResNet_101")
    parser.add_argument("--student",            type = str,     default = "ResNet_18")
    parser.add_argument("--attention",          type = str,     default = "GE")
    parser.add_argument("--attention_weight_0", type = float,   default = 1.0)
    parser.add_argument("--attention_weight_1", type = float,   default = 1.0)
    parser.add_argument("--attention_weight_2", type = float,   default = 1.0)
    parser.add_argument("--attention_weight_3", type = float,   default = 1.0)
    parser.add_argument("--training_path",      type = str,     default = "/share/Datasets/ImageNet/Subset/train")
    parser.add_argument("--validation_path",    type = str,     default = "/share/Datasets/ImageNet/Subset/val")
    parser.add_argument("--gpu",                type = int,     default = 2)
    parser.add_argument("--batch_size",         type = int,     default = 32)
    parser.add_argument("--epoch",              type = int,     default = 30)
    parser.add_argument('--interval',           type = int,     default = 30)
    parser.add_argument('--gamma',              type = float,   default = 0.1)
    return parser.parse_args()


if __name__ == "__main__":
    args = set_args()
    filename = save_env_on_filename(args)
    writer = SummaryWriter(log_dir = "/home/atheist8E/Earth/Civilization/Alexandria/{}_time_{}".format(filename, datetime.now().strftime("%Y_%m_%d_%H_%M_%S")))
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
    model = IcarusNet_v5(args, writer).cuda(args.gpu)
    student_criterion = nn.CrossEntropyLoss()
    attention_criterion = nn.MSELoss()
    student_optimizer = torch.optim.SGD(model.parameters(), lr = args.learning_rate, weight_decay = 0.0001, momentum = 0.1)
    scheduler = StepLR(student_optimizer, step_size=30, gamma=0.1)
    model.fit(train_loader, test_loader, student_criterion, attention_criterion, student_optimizer, args.epoch, scheduler)
    writer = model.report()
    writer.close()
