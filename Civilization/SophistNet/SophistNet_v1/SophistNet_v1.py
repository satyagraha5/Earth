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

#Custom Modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from Shrine.architecture import *
from Shrine.dataset import *
from Shrine.report import *


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_path", type = str, default = "/share/Datasets/ImageNet/Subset/train")
    parser.add_argument("--validation_path", type = str, default = "/share/Datasets/ImageNet/Subset/val")
    parser.add_argument("--gpu", type = int, default = 2)
    parser.add_argument("--batch_size", type = int, default = 32)
    parser.add_argument("--epoch", type = int, default = 30)
    parser.add_argument("--student_learning_rate", type = float, default = 0.01)
    parser.add_argument("--attention_learning_rate_1", type = float, default = 0.5)
    parser.add_argument("--attention_learning_rate_2", type = float, default = 0.5)
    parser.add_argument("--attention_learning_rate_3", type = float, default = 0.5)
    parser.add_argument("--attention_learning_rate_4", type = float, default = 0.5)
    return parser.parse_args()


if __name__ == "__main__":
    args = set_args()
    writer = SummaryWriter(log_dir = "/home/atheist8E/Earth/Civilization/Alexandria/SophistNet_v1_batch_size_{}_epoch_{}_learning_rate_{}_time_{}".format(args.batch_size,args.epoch,args.student_learning_rate,datetime.now().strftime("%Y_%m_%d_%H_%M_%S")))
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

    model = SophistNet_v1(args, writer).cuda(args.gpu)

    student_criterion = nn.CrossEntropyLoss()
    attention_criterion = nn.MSELoss()

    student_optimizer = torch.optim.SGD(model.conv1.parameters(), lr = args.student_learning_rate, weight_decay = 0.0001, momentum = 0.1)
    for module in model.student[1:]:
        student_optimizer.add_param_group({'params': module.parameters()})
    attention_optimizer_1 = torch.optim.SGD(model.attention_1.parameters(), lr = args.attention_learning_rate_1, weight_decay = 0.0001, momentum = 0.1)
    attention_optimizer_2 = torch.optim.SGD(model.attention_2.parameters(), lr = args.attention_learning_rate_2, weight_decay = 0.0001, momentum = 0.1)
    attention_optimizer_3 = torch.optim.SGD(model.attention_3.parameters(), lr = args.attention_learning_rate_3, weight_decay = 0.0001, momentum = 0.1)
    attention_optimizer_4 = torch.optim.SGD(model.attention_4.parameters(), lr = args.attention_learning_rate_4, weight_decay = 0.0001, momentum = 0.1)
    attention_optimizers = [attention_optimizer_1, attention_optimizer_2, attention_optimizer_3, attention_optimizer_4]
    model.fit(train_loader, test_loader, student_criterion, attention_criterion, student_optimizer, attention_optimizers, num_epochs = args.epoch)
    writer = model.report()
    writer.close()
