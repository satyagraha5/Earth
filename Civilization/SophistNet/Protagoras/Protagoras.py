#!/home/atheist8E/anaconda3/bin/python


#Basic Modules
import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tensorboardX import SummaryWriter

#Pytorch
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.utils as utils
import torchvision.models as pretrained_models

#Custom Modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from Shrine.architecture import *
from Shrine.dataset import *
from Shrine.report import *


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_path", type = str, default = "/share/Datasets/ImageNet/Subset/train")
    parser.add_argument("--validation_path", type = str, default = "/share/Datasets/ImageNet/Subset/val")
    parser.add_argument("--batch_size", type = int, default = 8)
    parser.add_argument("--gpu", type = int, default = 2)
    return parser.parse_args()

def score(outputs, labels):
    max_vals, max_indices = outputs.max(1)
    accuracy = (max_indices == labels).float().sum()/max_indices.size()[0]
    return accuracy

def clear():
    for path, dirs, files in os.walk("./"):
        for filename in files:
            if filename.endswith(".jpg"):
                os.remove(os.path.join(path, filename))


if __name__ == "__main__":
    clear()
    args = set_args()
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    with open("imagenet_class_index.json", "rb") as f:
        class_idx = json.load(f)
    train_loader, test_loader = ImageNet(args, train_transform, test_transform)
    teacher_model = pretrained_models.resnet101(pretrained = True).cuda(args.gpu)
    student_model = pretrained_models.resnet18(pretrained = True).cuda(args.gpu)
    criterion = nn.CrossEntropyLoss()
    teacher_sub_layers = list(teacher_model.children())[:5]
    teacher_sub_model = nn.Sequential(*teacher_sub_layers).cuda(args.gpu)
    student_sub_layers = list(student_model.children())[:5]
    student_sub_model = nn.Sequential(*student_sub_layers).cuda(args.gpu)
    
    for i, (images, labels) in enumerate(train_loader, start = 1):
        with torch.no_grad():
            teacher_model.eval()
            student_model.eval()
            images = images.cuda(args.gpu)
            labels = labels.cuda(args.gpu)

            teacher_full_outputs = teacher_model.forward(images)
            teacher_sub_outputs = teacher_sub_model.forward(images)

            student_full_outputs = student_model.forward(images)
            student_sub_outputs = student_sub_model.forward(images)

            for i_th_batch in range(args.batch_size):
                image = transforms.ToPILImage()(images[i_th_batch].cpu())
                answer = labels[i_th_batch]
                answer_name = class_idx[str(answer.item())][1]

                #Teacher
                max_vals, max_indices = teacher_full_outputs.max(1)
                teacher_full_guess = max_indices[i_th_batch]
                teacher_full_guess_name = class_idx[str(teacher_full_guess.item())][1]
                teacher_full_guess_prob = max_vals[i_th_batch]

                #Student
                max_vals, max_indices = student_full_outputs.max(1)
                student_full_guess = max_indices[i_th_batch]
                student_full_guess_name = class_idx[str(student_full_guess.item())][1]
                student_full_guess_prob = max_vals[i_th_batch]

                image.save("{}/teacher_guess_{}_student_guess_{}_answer_{}.jpg".format(i_th_batch,teacher_full_guess_name, student_full_guess_name, answer_name))
                
                teacher_feature_map = teacher_sub_model.forward(images)
                student_feature_map = student_sub_model.forward(images)
                
                fig = plt.figure()
                for i_th_channel in range(64):
                    feature_map = teacher_feature_map[i_th_batch][i_th_channel]
                    ax = plt.subplot(8, 8, i_th_channel + 1)
                    ax.axis("off")
                    ax.imshow(feature_map.cpu(), cmap = "gray")
                fig.savefig("{}/teacher_feature_map.jpg".format(i_th_batch))

                fig = plt.figure()
                for i_th_channel in range(64):
                    feature_map = student_feature_map[i_th_batch][i_th_channel]
                    ax = plt.subplot(8, 8, i_th_channel + 1)
                    ax.axis("off")
                    ax.imshow(feature_map.cpu(), cmap = "gray")
                fig.savefig("{}/student_feature_map.jpg".format(i_th_batch))

                fig = plt.figure()
                for i_th_channel in range(192):
                    if i_th_channel % 3 == 0:
                        feature_map = teacher_feature_map[i_th_batch][i_th_channel]
                    elif i_th_channel % 3 == 1:
                        feature_map += teacher_feature_map[i_th_batch][i_th_channel]
                    elif i_th_channel % 3 == 2:
                        feature_map += teacher_feature_map[i_th_batch][i_th_channel]
                        feature_map /= 3
                        ax = plt.subplot(8, 8, i_th_channel // 3 + 1)
                        ax.axis("off")
                        ax.imshow(feature_map.cpu(), cmap = "gray")
                fig.savefig("{}/teacher_feature_map_3.jpg".format(i_th_batch))
        break