#Basic Modules
from tqdm import tqdm as tqdm

#Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as utils
import torchvision.models as pretrained_models

#Custom Modules
from .report import *
from .block import *
from .lr import *

class MyNeuralNetwork(nn.Module):
    def __init__(self, args, writer):
        super().__init__()
        self.args = args
        self.writer = writer

    def fit(self, train_loader, test_loader, criterion, optimizer, num_epochs):
        for epoch in tqdm(range(1, num_epochs + 1)):
            print()
            test_loader_iterator = iter(test_loader)
            for i, (train_images, train_labels) in enumerate(train_loader, start = 1):
                #Train
                with torch.enable_grad():
                    self.train()
                    train_images = train_images.cuda(self.args.gpu)
                    train_labels = train_labels.cuda(self.args.gpu)
                    train_outputs = self.forward(train_images)
                    train_loss = criterion(train_outputs, train_labels)
                    optimizer.zero_grad()
                    train_loss.backward()
                    optimizer.step()
                    (self.writer).add_scalar("Train Loss", train_loss.item(), len(train_loader.dataset) * epoch + i)
                    train_top_1_accuracy, train_top_5_accuracy = self.score(train_outputs, train_labels)
                    (self.writer).add_scalar("Train Top 1 Accuracy", train_top_1_accuracy, len(train_loader.dataset) * epoch + i)
                    (self.writer).add_scalar("Train Top 5 Accuracy", train_top_5_accuracy, len(train_loader.dataset) * epoch + i)
                #Validation
                five_percent = int(len(train_loader.dataset) / self.args.batch_size * 0.05)
                if i % five_percent == 0:
                    with torch.no_grad():
                        self.eval()
                        val_images, val_labels = next(iter(test_loader))
                        val_images = val_images.cuda(self.args.gpu)
                        val_labels = val_labels.cuda(self.args.gpu)
                        val_outputs = self.forward(val_images)
                        val_loss = criterion(val_outputs, val_labels)
                        (self.writer).add_scalar("Validation Loss", val_loss.item(), len(train_loader.dataset) * epoch + i)
                        val_top_1_accuracy, val_top_5_accuracy = self.score(val_outputs, val_labels)
                        (self.writer).add_scalar("Validation Top 1 Accuracy", val_top_1_accuracy, len(train_loader.dataset) * epoch + i)
                        (self.writer).add_scalar("Validation Top 5 Accuracy", val_top_5_accuracy, len(train_loader.dataset) * epoch + i)
                        print("Epoch: {} [{}/{} ({:.0f}%)]\n\tTrain Loss: {:.2f} Train Top 1 Accuracy: {:.2f} Train Top 5 Accuracy: {:.2f}\n\tValidation Loss: {:.2f} Validation Top 1 Accuracy: {:.2f} Validation Top 5 Accuracy: {:.2f}".format(epoch,
                            i * self.args.batch_size, len(train_loader.dataset),
                            100. * i / len(train_loader),
                            train_loss.item(), train_top_1_accuracy, train_top_5_accuracy,
                            val_loss.item(), val_top_1_accuracy, val_top_5_accuracy))

    def score(self, outputs, labels):
        top_vals, top_indices = outputs.topk(5)
        top_1_accuracy = 0.0
        top_5_accuracy = 0.0
        for i_th_batch in range(self.args.batch_size):
            top_1_accuracy += (top_indices[i_th_batch][0] == labels[i_th_batch]).float()
            for top_k in range(5):
                if (top_indices[i_th_batch][top_k] == labels[i_th_batch]):
                    top_5_accuracy += 1.0
                    break
        top_1_accuracy /= self.args.batch_size
        top_5_accuracy /= self.args.batch_size
        return top_1_accuracy, top_5_accuracy

    def report(self):
        return self.writer

    def dummy_test(self):
        dummy_input = torch.zeros(1, 3, 224,224).to(torch.device("cuda"))
        dummy_output = self.forward(dummy_input)
        print(dummy_output.shape)

class SophistNet_v1(MyNeuralNetwork):
    #Consider Attention Module as Independent Network
    def __init__(self, args, writer):
        super().__init__(args, writer)

        self.teacher = pretrained_models.resnet101(pretrained = True)
        for param in self.teacher.parameters():
            param.requires_grad = False

        #ResNet 18 + SE
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.pool1 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

        self.resblock1_a = ResBlock(in_channels = 64, out_channels = 64, stride = 1)
        self.resblock1_b = ResBlock(in_channels = 64, out_channels = 64, stride = 1)
        self.attention_1 = SE(in_channels = 64, reduction_ratio = 4)

        self.resblock2_a = ResBlock(in_channels = 64, out_channels = 128, stride = 2)
        self.resblock2_b = ResBlock(in_channels = 128, out_channels = 128, stride = 1)
        self.attention_2 = SE(in_channels = 128, reduction_ratio = 4)

        self.resblock3_a = ResBlock(in_channels = 128, out_channels = 256, stride = 2)
        self.resblock3_b = ResBlock(in_channels = 256, out_channels = 256, stride = 1)
        self.attention_3 = SE(in_channels = 256, reduction_ratio = 4)

        self.resblock4_a = ResBlock(in_channels = 256, out_channels = 512, stride = 2)
        self.resblock4_b = ResBlock(in_channels = 512, out_channels = 512, stride = 1)
        self.attention_4 = SE(in_channels = 512, reduction_ratio = 4)

        self.pool2 = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_features = 512, out_features = 1000, bias = True)

        self.student = [
            self.conv1, self.pool1,
            self.resblock1_a, self.resblock1_b,
            self.resblock2_a, self.resblock2_b,
            self.resblock3_a, self.resblock3_b,
            self.resblock4_a, self.resblock4_b,
            self.pool2, self.fc
        ]

        self.attention = [
            self.attention_1,
            self.attention_2,
            self.attention_3,
            self.attention_4,
        ]

        self.attention_feature_maps = list(range(4))
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.pool1(out)

        out = self.resblock1_a(out)
        out = self.resblock1_b(out)
        out = self.attention_1(out)
        self.attention_feature_maps[0] = out

        out = self.resblock2_a(out)
        out = self.resblock2_b(out)
        out = self.attention_2(out)
        self.attention_feature_maps[1] = out

        out = self.resblock3_a(out)
        out = self.resblock3_b(out)
        out = self.attention_3(out)
        self.attention_feature_maps[2] = out

        out = self.resblock4_a(out)
        out = self.resblock4_b(out)
        out = self.attention_4(out)
        self.attention_feature_maps[3] = out

        out = self.pool2(out)
        out = out.view(-1, 512)
        out = self.fc(out)
        return out

    def fit(self, train_loader, test_loader, student_criterion, attention_criterion, student_optimizer, attention_optimizers, num_epochs):
        for epoch in tqdm(range(1, num_epochs + 1)):
            print()
            test_loader_iterator = iter(test_loader)
            for i, (train_images, train_labels) in enumerate(train_loader, start = 1):
                #Train
                self.train()
                train_images = train_images.cuda(self.args.gpu)
                train_labels = train_labels.cuda(self.args.gpu)
                

                #Update Student
                self.set_attention_no_grad()
                student_train_outputs = self.forward(train_images)
                student_train_loss = student_criterion(student_train_outputs, train_labels)
                student_optimizer.zero_grad()
                student_train_loss.backward()
                student_optimizer.step()
                (self.writer).add_scalar("Student Loss", student_train_loss.item(), len(train_loader.dataset) * epoch + i)
                (self.writer).add_scalar("Student Train Accuracy", self.score(student_train_outputs, train_labels), len(train_loader.dataset) * epoch + i)

                #Update Attention
                teacher_feature_maps = self.get_teacher_feature_map(train_images)
                projected_teacher_feature_maps = list(range(4))
                attention_train_loss = list(range(4))

                for location in range(4):
                    self.set_attention_grad(location)
                    projected_teacher_feature_maps[location] = self.feature_map_projection(teacher_feature_maps[location], self.attention_feature_maps[location])
                    attention_train_loss[location] = attention_criterion(projected_teacher_feature_maps[location], self.attention_feature_maps[location])
                    attention_optimizers[location].zero_grad()
                    attention_train_loss[location].backward()
                    attention_optimizers[location].step()
                    (self.writer).add_scalar("Attention Loss {}".format(location), attention_train_loss[location].item(), len(train_loader.dataset) * epoch + i)

                #Validation
                five_percent = int(len(train_loader.dataset) / self.args.batch_size * 0.001)
                if i % five_percent == 0:
                    with torch.no_grad():
                        self.eval()
                        val_images, val_labels = next(iter(test_loader))
                        val_images = val_images.cuda(self.args.gpu)
                        val_labels = val_labels.cuda(self.args.gpu)
                        val_outputs = self.forward(val_images)
                        val_loss = student_criterion(val_outputs, val_labels)
                        (self.writer).add_scalar("Validation Loss", val_loss.item(), len(train_loader.dataset) * epoch + i)
                        (self.writer).add_scalar("Validation Accuracy", self.score(val_outputs, val_labels), len(train_loader.dataset) * epoch + i)
                        print("Epoch: {} [{}/{} ({:.0f}%)] \nStudent Train Loss: {:.6f} Train Accuracy: {:.6f} \nAttention Loss 0: {:.6f} Attention Loss 1: {:.6f} Attention Loss 2: {:.6f} Attention Loss 3: {:.6f} \nValidation Loss: {:.6f} Validation Accuracy: {:.6f}""".format(epoch, i * self.args.batch_size, len(train_loader.dataset), 100. * i / len(train_loader),
                            student_train_loss.item(), self.score(student_train_outputs, train_labels),
                            attention_train_loss[0].item(), attention_train_loss[1].item(), attention_train_loss[2].item(), attention_train_loss[3].item(), 
                            val_loss.item(), self.score(val_outputs, val_labels)))
    
    def get_teacher_feature_map(self, images):
        teacher_feature_maps = list()
        for location in (5, 6, 7, 8):
            teacher_sub_layers = list(self.teacher.children())[:location]
            teacher_sub_model = nn.Sequential(*teacher_sub_layers).cuda(self.args.gpu)
            teacher_feature_maps.append(teacher_sub_model.forward(images))
        return teacher_feature_maps

    def set_attention_grad(self, location):
        for param in self.parameters():
            param.requires_grad = False
        target_attention = self.attention[location]
        for param in target_attention.parameters():
            param.requires_grad = True

    def set_attention_no_grad(self):
        for module in self.student:
            for param in module.parameters():
                param.requires_grad = True
        for param in self.attention_1.parameters():
            param.requires_grad = False
        for param in self.attention_2.parameters():
            param.requires_grad = False
        for param in self.attention_3.parameters():
            param.requires_grad = False
        for param in self.attention_4.parameters():
            param.requires_grad = False

    def feature_map_projection(self, teacher_feature_map, student_feature_map):
        num_channels = teacher_feature_map.size()[1]
        projected_feature_map = torch.zeros_like(student_feature_map)
        for i_th_batch in range(self.args.batch_size):
            for i_th_channel in range(num_channels):
                if i_th_channel % 4 == 0:
                    feature_map = teacher_feature_map[i_th_batch][i_th_channel]
                elif i_th_channel % 4 == 1:
                    feature_map += teacher_feature_map[i_th_batch][i_th_channel]
                elif i_th_channel % 4 == 2:
                    feature_map += teacher_feature_map[i_th_batch][i_th_channel]
                elif i_th_channel % 4 == 3:
                    feature_map += teacher_feature_map[i_th_batch][i_th_channel]
                    feature_map /= 4
                    projected_feature_map[i_th_batch][i_th_channel//4] = feature_map
        return projected_feature_map

class SophistNet_v2(MyNeuralNetwork):
    #loss = loss + sum(loss)
    def __init__(self, args, writer):
        super().__init__(args, writer)

        self.teacher = pretrained_models.resnet101(pretrained = True)
        for param in self.teacher.parameters():
            param.requires_grad = False

        #ResNet 18 + SE
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.pool1 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

        self.resblock1_a = ResBlock(in_channels = 64, out_channels = 64, stride = 1)
        self.resblock1_b = ResBlock(in_channels = 64, out_channels = 64, stride = 1)
        self.attention_1 = SE(in_channels = 64, reduction_ratio = 4)

        self.resblock2_a = ResBlock(in_channels = 64, out_channels = 128, stride = 2)
        self.resblock2_b = ResBlock(in_channels = 128, out_channels = 128, stride = 1)
        self.attention_2 = SE(in_channels = 128, reduction_ratio = 4)

        self.resblock3_a = ResBlock(in_channels = 128, out_channels = 256, stride = 2)
        self.resblock3_b = ResBlock(in_channels = 256, out_channels = 256, stride = 1)
        self.attention_3 = SE(in_channels = 256, reduction_ratio = 4)

        self.resblock4_a = ResBlock(in_channels = 256, out_channels = 512, stride = 2)
        self.resblock4_b = ResBlock(in_channels = 512, out_channels = 512, stride = 1)
        self.attention_4 = SE(in_channels = 512, reduction_ratio = 4)

        self.pool2 = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_features = 512, out_features = 1000, bias = True)

        self.attention_feature_maps = list(range(4))

    def forward(self, x):
        out = self.conv1(x)
        out = self.pool1(out)

        out = self.resblock1_a(out)
        out = self.resblock1_b(out)
        out = self.attention_1(out)
        self.attention_feature_maps[0] = out

        out = self.resblock2_a(out)
        out = self.resblock2_b(out)
        out = self.attention_2(out)
        self.attention_feature_maps[1] = out

        out = self.resblock3_a(out)
        out = self.resblock3_b(out)
        out = self.attention_3(out)
        self.attention_feature_maps[2] = out

        out = self.resblock4_a(out)
        out = self.resblock4_b(out)
        out = self.attention_4(out)
        self.attention_feature_maps[3] = out

        out = self.pool2(out)
        out = out.view(-1, 512)
        out = self.fc(out)
        return out

    def fit(self, train_loader, test_loader, student_criterion, attention_criterion, student_optimizer, num_epochs):
        for epoch in tqdm(range(1, num_epochs + 1)):
            print()
            test_loader_iterator = iter(test_loader)
            for i, (train_images, train_labels) in enumerate(train_loader, start = 1):
                #Train
                self.train()
                train_images = train_images.cuda(self.args.gpu)
                train_labels = train_labels.cuda(self.args.gpu)
                
                #Student
                student_train_outputs = self.forward(train_images)
                student_train_loss = student_criterion(student_train_outputs, train_labels)
                total_train_loss = student_train_loss
                (self.writer).add_scalar("Student Train Loss", student_train_loss.item(), len(train_loader.dataset) * epoch + i)
                (self.writer).add_scalar("Train Accuracy", self.score(student_train_outputs, train_labels), len(train_loader.dataset) * epoch + i)

                #Attention
                teacher_feature_maps = self.get_teacher_feature_map(train_images)
                attention_train_loss = list(range(4))
                projected_teacher_feature_maps = list(range(4))
                for location in range(4):
                    projected_teacher_feature_maps[location] = self.feature_map_projection(teacher_feature_maps[location], self.attention_feature_maps[location])
                    attention_train_loss[location] = attention_criterion(projected_teacher_feature_maps[location], self.attention_feature_maps[location])
                    total_train_loss += attention_train_loss[location]
                    (self.writer).add_scalar("Attention Train Loss {}".format(location), attention_train_loss[location].item(), len(train_loader.dataset) * epoch + i)

                #Combine
                (self.writer).add_scalar("Total Train Loss", total_train_loss.item(), len(train_loader.dataset) * epoch + i)
                student_optimizer.zero_grad()
                total_train_loss.backward()
                student_optimizer.step()
                
                #Validation
                report_point = int(len(train_loader.dataset) / self.args.batch_size * 0.001)
                if i % report_point == 0:
                    with torch.no_grad():
                        self.eval()
                        val_images, val_labels = next(iter(test_loader))
                        val_images = val_images.cuda(self.args.gpu)
                        val_labels = val_labels.cuda(self.args.gpu)
                        student_val_outputs = self.forward(val_images)
                        student_val_loss = student_criterion(student_val_outputs, val_labels)
                        total_val_loss = student_val_loss
                        (self.writer).add_scalar("Student Validation Loss", student_val_loss.item(), len(train_loader.dataset) * epoch + i)
                        (self.writer).add_scalar("Validation Accuracy", self.score(student_val_outputs, val_labels), len(train_loader.dataset) * epoch + i)

                        teacher_feature_maps = self.get_teacher_feature_map(val_images)
                        attention_val_loss = list(range(4))
                        projected_teacher_feature_maps = list(range(4))
                        for location in range(4):
                            projected_teacher_feature_maps[location] = self.feature_map_projection(teacher_feature_maps[location], self.attention_feature_maps[location])
                            attention_val_loss[location] = attention_criterion(projected_teacher_feature_maps[location], self.attention_feature_maps[location])
                            total_val_loss += attention_val_loss[location]
                            (self.writer).add_scalar("Attention Validation Loss {}".format(location), attention_val_loss[location].item(), len(train_loader.dataset) * epoch + i)
                        (self.writer).add_scalar("Total Validation Loss", total_val_loss.item(), len(train_loader.dataset) * epoch + i)

                        print("Epoch: {} [{}/{}({:.0f}%)] Total Train Loss: {:.2f} Student/Attention Train Loss: {:.2f}/{:.2f},{:.2f},{:.2f},{:.2f} Train Accuracy: {:.2f} Total Val Loss: {:.2f} Student/Attention Val Loss: {:.2f}/{:.2f},{:.2f},{:.2f},{:.2f} Val Accuracy: {:.2f}".format(epoch,
                            i * self.args.batch_size, len(train_loader.dataset), 100. * i / len(train_loader),
                            total_train_loss.item(), student_train_loss.item(), attention_train_loss[0].item(), attention_train_loss[1].item(), attention_train_loss[2].item(), attention_train_loss[3].item(), self.score(student_train_outputs, train_labels), 
                            total_val_loss.item(), student_val_loss.item(), attention_val_loss[0].item(), attention_val_loss[1].item(), attention_val_loss[2].item(), attention_val_loss[3].item(), self.score(student_val_outputs, val_labels)))
    
    def get_teacher_feature_map(self, images):
        teacher_feature_maps = list()
        for location in (5, 6, 7, 8):
            teacher_sub_layers = list(self.teacher.children())[:location]
            teacher_sub_model = nn.Sequential(*teacher_sub_layers).cuda(self.args.gpu)
            teacher_feature_maps.append(teacher_sub_model.forward(images))
        return teacher_feature_maps

    def feature_map_projection(self, teacher_feature_map, student_feature_map):
        num_channels = teacher_feature_map.size()[1]
        projected_feature_map = torch.zeros_like(student_feature_map)
        for i_th_batch in range(self.args.batch_size):
            for i_th_channel in range(num_channels):
                try:
                    if i_th_channel % 4 == 0:
                        feature_map = teacher_feature_map[i_th_batch][i_th_channel]
                    elif i_th_channel % 4 == 1:
                        feature_map += teacher_feature_map[i_th_batch][i_th_channel]
                    elif i_th_channel % 4 == 2:
                        feature_map += teacher_feature_map[i_th_batch][i_th_channel]
                    elif i_th_channel % 4 == 3:
                        feature_map += teacher_feature_map[i_th_batch][i_th_channel]
                        feature_map /= 4
                        projected_feature_map[i_th_batch][i_th_channel//4] = feature_map
                except IndexError as e:
                    print("IndexError: {}\n".format(e))
                    print("Index: i_th_batch - {} i_th_channel - {}\n".format(i_th_batch, i_th_channel))
        return projected_feature_map

class SophistNet_v3(MyNeuralNetwork):
    #loss = loss + lambda * sum(loss)
    def __init__(self, args, writer):
        super().__init__(args, writer)

        self.teacher = pretrained_models.resnet101(pretrained = True)
        for param in self.teacher.parameters():
            param.requires_grad = False

        #ResNet 18 + SE
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.pool1 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

        self.resblock1_a = ResBlock(in_channels = 64, out_channels = 64, stride = 1)
        self.resblock1_b = ResBlock(in_channels = 64, out_channels = 64, stride = 1)
        self.attention_1 = SE(in_channels = 64, reduction_ratio = 4)

        self.resblock2_a = ResBlock(in_channels = 64, out_channels = 128, stride = 2)
        self.resblock2_b = ResBlock(in_channels = 128, out_channels = 128, stride = 1)
        self.attention_2 = SE(in_channels = 128, reduction_ratio = 4)

        self.resblock3_a = ResBlock(in_channels = 128, out_channels = 256, stride = 2)
        self.resblock3_b = ResBlock(in_channels = 256, out_channels = 256, stride = 1)
        self.attention_3 = SE(in_channels = 256, reduction_ratio = 4)

        self.resblock4_a = ResBlock(in_channels = 256, out_channels = 512, stride = 2)
        self.resblock4_b = ResBlock(in_channels = 512, out_channels = 512, stride = 1)
        self.attention_4 = SE(in_channels = 512, reduction_ratio = 4)

        self.pool2 = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_features = 512, out_features = 1000, bias = True)

        self.attention_feature_maps = list(range(4))

    def forward(self, x):
        out = self.conv1(x)
        out = self.pool1(out)

        out = self.resblock1_a(out)
        out = self.resblock1_b(out)
        out = self.attention_1(out)
        self.attention_feature_maps[0] = out

        out = self.resblock2_a(out)
        out = self.resblock2_b(out)
        out = self.attention_2(out)
        self.attention_feature_maps[1] = out

        out = self.resblock3_a(out)
        out = self.resblock3_b(out)
        out = self.attention_3(out)
        self.attention_feature_maps[2] = out

        out = self.resblock4_a(out)
        out = self.resblock4_b(out)
        out = self.attention_4(out)
        self.attention_feature_maps[3] = out

        out = self.pool2(out)
        out = out.view(-1, 512)
        out = self.fc(out)
        return out

    def fit(self, train_loader, test_loader, student_criterion, attention_criterion, student_optimizer, num_epochs):
        for epoch in tqdm(range(1, num_epochs + 1)):
            print()
            test_loader_iterator = iter(test_loader)
            for i, (train_images, train_labels) in enumerate(train_loader, start = 1):
                #Train
                self.train()
                train_images = train_images.cuda(self.args.gpu)
                train_labels = train_labels.cuda(self.args.gpu)
                
                #Student
                student_train_outputs = self.forward(train_images)
                student_train_loss = student_criterion(student_train_outputs, train_labels)
                total_train_loss = student_train_loss
                (self.writer).add_scalar("Student Train Loss", student_train_loss.item(), len(train_loader.dataset) * epoch + i)
                (self.writer).add_scalar("Train Accuracy", self.score(student_train_outputs, train_labels), len(train_loader.dataset) * epoch + i)

                #Attention
                teacher_feature_maps = self.get_teacher_feature_map(train_images)
                attention_train_loss = list(range(4))
                projected_teacher_feature_maps = list(range(4))
                for location in range(4):
                    projected_teacher_feature_maps[location] = self.feature_map_projection(teacher_feature_maps[location], self.attention_feature_maps[location])
                    attention_train_loss[location] = attention_criterion(projected_teacher_feature_maps[location], self.attention_feature_maps[location])
                    total_train_loss += self.args.attention_weight * attention_train_loss[location]
                    (self.writer).add_scalar("Attention Train Loss {}".format(location), attention_train_loss[location].item(), len(train_loader.dataset) * epoch + i)

                #Combine
                (self.writer).add_scalar("Total Train Loss", total_train_loss.item(), len(train_loader.dataset) * epoch + i)
                student_optimizer.zero_grad()
                total_train_loss.backward()
                student_optimizer.step()
                
                #Validation
                five_percent = int(len(train_loader.dataset) / self.args.batch_size * 0.001)
                if i % five_percent == 0:
                    with torch.no_grad():
                        self.eval()
                        val_images, val_labels = next(iter(test_loader))
                        val_images = val_images.cuda(self.args.gpu)
                        val_labels = val_labels.cuda(self.args.gpu)
                        student_val_outputs = self.forward(val_images)
                        student_val_loss = student_criterion(student_val_outputs, val_labels)
                        total_val_loss = student_val_loss
                        (self.writer).add_scalar("Student Validation Loss", student_val_loss.item(), len(train_loader.dataset) * epoch + i)
                        (self.writer).add_scalar("Validation Accuracy", self.score(student_val_outputs, val_labels), len(train_loader.dataset) * epoch + i)

                        teacher_feature_maps = self.get_teacher_feature_map(val_images)
                        attention_val_loss = list(range(4))
                        projected_teacher_feature_maps = list(range(4))
                        for location in range(4):
                            projected_teacher_feature_maps[location] = self.feature_map_projection(teacher_feature_maps[location], self.attention_feature_maps[location])
                            attention_val_loss[location] = attention_criterion(projected_teacher_feature_maps[location], self.attention_feature_maps[location])
                            total_val_loss += self.args.attention_weight * attention_val_loss[location]
                            (self.writer).add_scalar("Attention Validation Loss {}".format(location), attention_val_loss[location].item(), len(train_loader.dataset) * epoch + i)
                        (self.writer).add_scalar("Total Validation Loss", total_val_loss.item(), len(train_loader.dataset) * epoch + i)

                        print("Epoch: {} [{}/{}({:.0f}%)] Total Train Loss: {:.2f} Student/Attention Train Loss: {:.2f}/{:.2f},{:.2f},{:.2f},{:.2f} Train Accuracy: {:.2f} Total Val Loss: {:.2f} Student/Attention Val Loss: {:.2f}/{:.2f},{:.2f},{:.2f},{:.2f} Val Accuracy: {:.2f}".format(epoch,
                            i * self.args.batch_size, len(train_loader.dataset), 100. * i / len(train_loader),
                            total_train_loss.item(), student_train_loss.item(), attention_train_loss[0].item(), attention_train_loss[1].item(), attention_train_loss[2].item(), attention_train_loss[3].item(), self.score(student_train_outputs, train_labels), 
                            total_val_loss.item(), student_val_loss.item(), attention_val_loss[0].item(), attention_val_loss[1].item(), attention_val_loss[2].item(), attention_val_loss[3].item(), self.score(student_val_outputs, val_labels)))
    
    def get_teacher_feature_map(self, images):
        teacher_feature_maps = list()
        for location in (5, 6, 7, 8):
            teacher_sub_layers = list(self.teacher.children())[:location]
            teacher_sub_model = nn.Sequential(*teacher_sub_layers).cuda(self.args.gpu)
            teacher_feature_maps.append(teacher_sub_model.forward(images))
        return teacher_feature_maps

    def feature_map_projection(self, teacher_feature_map, student_feature_map):
        num_channels = teacher_feature_map.size()[1]
        projected_feature_map = torch.zeros_like(student_feature_map)
        for i_th_batch in range(self.args.batch_size):
            for i_th_channel in range(num_channels):
                try:
                    if i_th_channel % 4 == 0:
                        feature_map = teacher_feature_map[i_th_batch][i_th_channel]
                    elif i_th_channel % 4 == 1:
                        feature_map += teacher_feature_map[i_th_batch][i_th_channel]
                    elif i_th_channel % 4 == 2:
                        feature_map += teacher_feature_map[i_th_batch][i_th_channel]
                    elif i_th_channel % 4 == 3:
                        feature_map += teacher_feature_map[i_th_batch][i_th_channel]
                        feature_map /= 4
                        projected_feature_map[i_th_batch][i_th_channel//4] = feature_map
                except IndexError as e:
                    print("IndexError: {}\n".format(e))
                    print("Index: i_th_batch - {} i_th_channel - {}\n".format(i_th_batch, i_th_channel))

        return projected_feature_map

class SophistNet_v4(MyNeuralNetwork):
    #loss = loss + sum(lambda * loss)
    def __init__(self, args, writer):
        super().__init__(args, writer)

        self.teacher = pretrained_models.resnet101(pretrained = True)
        for param in self.teacher.parameters():
            param.requires_grad = False

        #ResNet 18 + SE
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.pool1 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

        self.resblock1_a = ResBlock(in_channels = 64, out_channels = 64, stride = 1)
        self.resblock1_b = ResBlock(in_channels = 64, out_channels = 64, stride = 1)
        self.attention_1 = SE(in_channels = 64, reduction_ratio = 4)

        self.resblock2_a = ResBlock(in_channels = 64, out_channels = 128, stride = 2)
        self.resblock2_b = ResBlock(in_channels = 128, out_channels = 128, stride = 1)
        self.attention_2 = SE(in_channels = 128, reduction_ratio = 4)

        self.resblock3_a = ResBlock(in_channels = 128, out_channels = 256, stride = 2)
        self.resblock3_b = ResBlock(in_channels = 256, out_channels = 256, stride = 1)
        self.attention_3 = SE(in_channels = 256, reduction_ratio = 4)

        self.resblock4_a = ResBlock(in_channels = 256, out_channels = 512, stride = 2)
        self.resblock4_b = ResBlock(in_channels = 512, out_channels = 512, stride = 1)
        self.attention_4 = SE(in_channels = 512, reduction_ratio = 4)

        self.pool2 = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_features = 512, out_features = 1000, bias = True)

        self.attention_feature_maps = list(range(4))

    def forward(self, x):
        out = self.conv1(x)
        out = self.pool1(out)

        out = self.resblock1_a(out)
        out = self.resblock1_b(out)
        out = self.attention_1(out)
        self.attention_feature_maps[0] = out

        out = self.resblock2_a(out)
        out = self.resblock2_b(out)
        out = self.attention_2(out)
        self.attention_feature_maps[1] = out

        out = self.resblock3_a(out)
        out = self.resblock3_b(out)
        out = self.attention_3(out)
        self.attention_feature_maps[2] = out

        out = self.resblock4_a(out)
        out = self.resblock4_b(out)
        out = self.attention_4(out)
        self.attention_feature_maps[3] = out

        out = self.pool2(out)
        out = out.view(-1, 512)
        out = self.fc(out)
        return out

    def fit(self, train_loader, test_loader, student_criterion, attention_criterion, student_optimizer, num_epochs):
        for epoch in tqdm(range(1, num_epochs + 1)):
            print()
            test_loader_iterator = iter(test_loader)
            for i, (train_images, train_labels) in enumerate(train_loader, start = 1):
                #Train
                self.train()
                train_images = train_images.cuda(self.args.gpu)
                train_labels = train_labels.cuda(self.args.gpu)
                
                #Student
                student_train_outputs = self.forward(train_images)
                student_train_loss = student_criterion(student_train_outputs, train_labels)
                total_train_loss = student_train_loss
                (self.writer).add_scalar("Student Train Loss", student_train_loss.item(), len(train_loader.dataset) * epoch + i)
                (self.writer).add_scalar("Train Accuracy", self.score(student_train_outputs, train_labels), len(train_loader.dataset) * epoch + i)

                #Attention
                teacher_feature_maps = self.get_teacher_feature_map(train_images)
                attention_train_loss = list(range(4))
                projected_teacher_feature_maps = list(range(4))
                for location in range(4):
                    projected_teacher_feature_maps[location] = self.feature_map_projection(teacher_feature_maps[location], self.attention_feature_maps[location])
                    attention_train_loss[location] = attention_criterion(projected_teacher_feature_maps[location], self.attention_feature_maps[location])
                    total_train_loss += self.args.attention_weights[location] * attention_train_loss[location]
                    (self.writer).add_scalar("Attention Train Loss {}".format(location), attention_train_loss[location].item(), len(train_loader.dataset) * epoch + i)

                #Combine
                (self.writer).add_scalar("Total Train Loss", total_train_loss.item(), len(train_loader.dataset) * epoch + i)
                student_optimizer.zero_grad()
                total_train_loss.backward()
                student_optimizer.step()
                
                #Validation
                five_percent = int(len(train_loader.dataset) / self.args.batch_size * 0.001)
                if i % five_percent == 0:
                    with torch.no_grad():
                        self.eval()
                        val_images, val_labels = next(iter(test_loader))
                        val_images = val_images.cuda(self.args.gpu)
                        val_labels = val_labels.cuda(self.args.gpu)
                        student_val_outputs = self.forward(val_images)
                        student_val_loss = student_criterion(student_val_outputs, val_labels)
                        total_val_loss = student_val_loss
                        (self.writer).add_scalar("Student Validation Loss", student_val_loss.item(), len(train_loader.dataset) * epoch + i)
                        (self.writer).add_scalar("Validation Accuracy", self.score(student_val_outputs, val_labels), len(train_loader.dataset) * epoch + i)

                        teacher_feature_maps = self.get_teacher_feature_map(val_images)
                        attention_val_loss = list(range(4))
                        projected_teacher_feature_maps = list(range(4))
                        for location in range(4):
                            projected_teacher_feature_maps[location] = self.feature_map_projection(teacher_feature_maps[location], self.attention_feature_maps[location])
                            attention_val_loss[location] = attention_criterion(projected_teacher_feature_maps[location], self.attention_feature_maps[location])
                            total_val_loss += self.args.attention_weights[location] * attention_val_loss[location]
                            (self.writer).add_scalar("Attention Validation Loss {}".format(location), attention_val_loss[location].item(), len(train_loader.dataset) * epoch + i)
                        (self.writer).add_scalar("Total Validation Loss", total_val_loss.item(), len(train_loader.dataset) * epoch + i)

                        print("Epoch: {} [{}/{}({:.0f}%)] Total Train Loss: {:.2f} Student/Attention Train Loss: {:.2f}/{:.2f},{:.2f},{:.2f},{:.2f} Train Accuracy: {:.2f} Total Val Loss: {:.2f} Student/Attention Val Loss: {:.2f}/{:.2f},{:.2f},{:.2f},{:.2f} Val Accuracy: {:.2f}".format(epoch,
                            i * self.args.batch_size, len(train_loader.dataset), 100. * i / len(train_loader),
                            total_train_loss.item(), student_train_loss.item(), attention_train_loss[0].item(), attention_train_loss[1].item(), attention_train_loss[2].item(), attention_train_loss[3].item(), self.score(student_train_outputs, train_labels), 
                            total_val_loss.item(), student_val_loss.item(), attention_val_loss[0].item(), attention_val_loss[1].item(), attention_val_loss[2].item(), attention_val_loss[3].item(), self.score(student_val_outputs, val_labels)))
    
    def get_teacher_feature_map(self, images):
        teacher_feature_maps = list()
        for location in (5, 6, 7, 8):
            teacher_sub_layers = list(self.teacher.children())[:location]
            teacher_sub_model = nn.Sequential(*teacher_sub_layers).cuda(self.args.gpu)
            teacher_feature_maps.append(teacher_sub_model.forward(images))
        return teacher_feature_maps

    def feature_map_projection(self, teacher_feature_map, student_feature_map):
        num_channels = teacher_feature_map.size()[1]
        projected_feature_map = torch.zeros_like(student_feature_map)
        for i_th_batch in range(self.args.batch_size):
            for i_th_channel in range(num_channels):
                try:
                    if i_th_channel % 4 == 0:
                        feature_map = teacher_feature_map[i_th_batch][i_th_channel]
                    elif i_th_channel % 4 == 1:
                        feature_map += teacher_feature_map[i_th_batch][i_th_channel]
                    elif i_th_channel % 4 == 2:
                        feature_map += teacher_feature_map[i_th_batch][i_th_channel]
                    elif i_th_channel % 4 == 3:
                        feature_map += teacher_feature_map[i_th_batch][i_th_channel]
                        feature_map /= 4
                        projected_feature_map[i_th_batch][i_th_channel//4] = feature_map
                except IndexError as e:
                    print("IndexError: {}\n".format(e))
                    print("Index: i_th_batch - {} i_th_channel - {}\n".format(i_th_batch, i_th_channel))

        return projected_feature_map

class IcarusNet(MyNeuralNetwork):
    def __init__(self, args, writer):
        super().__init__(args, writer)
        self.args = args
        self.writer = writer
        self.attention_feature_maps = list(range(4))
        self.attention_weights = list(range(4))
        self.attention_weights[0] = args.attention_weight_0
        self.attention_weights[1] = args.attention_weight_1
        self.attention_weights[2] = args.attention_weight_2
        self.attention_weights[3] = args.attention_weight_3

    def get_teacher_feature_map(self, images):
        teacher_feature_maps = list()
        for location in (5, 6, 7, 8):
            teacher_sub_layers = list(self.teacher.children())[:location]
            teacher_sub_model = nn.Sequential(*teacher_sub_layers).cuda(self.args.gpu)
            teacher_feature_maps.append(teacher_sub_model.forward(images))
        return teacher_feature_maps

    def feature_map_projection(self, teacher_feature_map, student_feature_map):
        num_channels = teacher_feature_map.size()[1]
        projected_feature_map = torch.zeros_like(student_feature_map)
        for i_th_batch in range(self.args.batch_size):
            for i_th_channel in range(num_channels):
                if i_th_channel % 4 == 0:
                    feature_map = teacher_feature_map[i_th_batch][i_th_channel]
                elif i_th_channel % 4 == 1:
                    feature_map += teacher_feature_map[i_th_batch][i_th_channel]
                elif i_th_channel % 4 == 2:
                    feature_map += teacher_feature_map[i_th_batch][i_th_channel]
                elif i_th_channel % 4 == 3:
                    feature_map += teacher_feature_map[i_th_batch][i_th_channel]
                    feature_map /= 4
                    projected_feature_map[i_th_batch][i_th_channel//4] = feature_map
        return projected_feature_map

    def total_loss(self, student_loss, attention_loss_list):
        total_loss = student_loss
        for i, attention_loss in enumerate(attention_loss_list):
            total_loss += self.attention_weights[i] * attention_loss
        return total_loss
    
    def fit(self, train_loader, test_loader, student_criterion, attention_criterion, student_optimizer, num_epochs, scheduler):
        for epoch in tqdm(range(1, num_epochs + 1)):
            print()
            test_loader_iterator = iter(test_loader)
            #Train
            for i, (train_images, train_labels) in enumerate(train_loader, start = 1):
                b, _, _, _ = train_images.shape
                if b != self.args.batch_size:
                    print("b: {}".format(b))
                    continue
                self.train()
                train_images = train_images.cuda(self.args.gpu)
                train_labels = train_labels.cuda(self.args.gpu)
                    
                #Student
                student_train_outputs = self.forward(train_images)
                student_train_loss = student_criterion(student_train_outputs, train_labels)
                (self.writer).add_scalar("Student Train Loss", student_train_loss.item(), len(train_loader.dataset) * epoch + i)
                train_top_1_accuracy, train_top_5_accuracy = self.score(student_train_outputs, train_labels)
                (self.writer).add_scalar("Train Top 1 Accuracy", train_top_1_accuracy, len(train_loader.dataset) * epoch + i)
                (self.writer).add_scalar("Train Top 5 Accuracy", train_top_5_accuracy, len(train_loader.dataset) * epoch + i)

                #Attention
                teacher_feature_maps = self.get_teacher_feature_map(train_images)
                attention_train_loss = list(range(4))
                projected_teacher_feature_maps = list(range(4))
                for location in range(4):
                    projected_teacher_feature_maps[location] = self.feature_map_projection(teacher_feature_maps[location], self.attention_feature_maps[location])
                    attention_train_loss[location] = attention_criterion(projected_teacher_feature_maps[location], self.attention_feature_maps[location])
                    (self.writer).add_scalar("Attention Train Loss {}".format(location), attention_train_loss[location].item(), len(train_loader.dataset) * epoch + i)
                    
                #Combine & Backward
                total_train_loss = self.total_loss(student_train_loss, attention_train_loss)
                (self.writer).add_scalar("Total Train Loss", total_train_loss.item(), len(train_loader.dataset) * epoch + i)
                student_optimizer.zero_grad()
                total_train_loss.backward()
                student_optimizer.step()
                    
                #Validation
                report_point = int(len(train_loader.dataset) / self.args.batch_size * 0.001)
                if i % report_point == 0:
                    with torch.no_grad():
                        self.eval()
                        val_images, val_labels = next(iter(test_loader))
                        val_images = val_images.cuda(self.args.gpu)
                        val_labels = val_labels.cuda(self.args.gpu)
                        student_val_outputs = self.forward(val_images)
                        student_val_loss = student_criterion(student_val_outputs, val_labels)
                        total_val_loss = student_val_loss
                        val_top_1_accuracy, val_top_5_accuracy = self.score(student_val_outputs, val_labels)
                        (self.writer).add_scalar("Student Validation Loss", student_val_loss.item(), len(train_loader.dataset) * epoch + i)
                        (self.writer).add_scalar("Validation Top 1 Accuracy", val_top_1_accuracy, len(train_loader.dataset) * epoch + i)
                        (self.writer).add_scalar("Validation Top 5 Accuracy", val_top_5_accuracy, len(train_loader.dataset) * epoch + i)

                        teacher_feature_maps = self.get_teacher_feature_map(val_images)
                        attention_val_loss = list(range(4))
                        projected_teacher_feature_maps = list(range(4))
                        for location in range(4):
                            projected_teacher_feature_maps[location] = self.feature_map_projection(teacher_feature_maps[location], self.attention_feature_maps[location])
                            attention_val_loss[location] = attention_criterion(projected_teacher_feature_maps[location], self.attention_feature_maps[location])
                            (self.writer).add_scalar("Attention Validation Loss {}".format(location), attention_val_loss[location].item(), len(train_loader.dataset) * epoch + i)
                        total_val_loss = self.total_loss(student_val_loss, attention_val_loss)
                        (self.writer).add_scalar("Total Validation Loss", total_val_loss.item(), len(train_loader.dataset) * epoch + i)

                        print("Epoch: {} [{}/{}({:.0f}%)]\n\tTotal Train Loss: {:.2f} Student/Attention Train Loss: {:.2f}/{:.2f},{:.2f},{:.2f},{:.2f} Train Top 1 Accuracy: {:.2f} Train Top 5 Accuracy: {:.2f}\n\tTotal Val Loss: {:.2f} Student/Attention Val Loss: {:.2f}/{:.2f},{:.2f},{:.2f},{:.2f} Val Top 1 Accuracy: {:.2f} Val Top 5 Accuracy: {:.2f}".format(epoch,
                            i * self.args.batch_size, len(train_loader.dataset), 100. * i / len(train_loader),
                            total_train_loss.item(), student_train_loss.item(), attention_train_loss[0].item(), attention_train_loss[1].item(), attention_train_loss[2].item(), attention_train_loss[3].item(), train_top_1_accuracy, train_top_5_accuracy, 
                            total_val_loss.item(), student_val_loss.item(), attention_val_loss[0].item(), attention_val_loss[1].item(), attention_val_loss[2].item(), attention_val_loss[3].item(), val_top_1_accuracy, val_top_5_accuracy))
            scheduler.step()
            
class IcarusNet_v1(IcarusNet):
    #loss = loss + sum(lambda * loss) | SE Attention
    def __init__(self, args, writer):
        super().__init__(args, writer)

        self.teacher = pretrained_models.resnet101(pretrained = True)
        for param in self.teacher.parameters():
            param.requires_grad = False

        #ResNet 18 + SE
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.pool1 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

        self.resblock1_a = ResBlock(in_channels = 64, out_channels = 64, stride = 1)
        self.resblock1_b = ResBlock(in_channels = 64, out_channels = 64, stride = 1)
        self.attention_1 = SE(in_channels = 64, reduction_ratio = 4)

        self.resblock2_a = ResBlock(in_channels = 64, out_channels = 128, stride = 2)
        self.resblock2_b = ResBlock(in_channels = 128, out_channels = 128, stride = 1)
        self.attention_2 = SE(in_channels = 128, reduction_ratio = 4)

        self.resblock3_a = ResBlock(in_channels = 128, out_channels = 256, stride = 2)
        self.resblock3_b = ResBlock(in_channels = 256, out_channels = 256, stride = 1)
        self.attention_3 = SE(in_channels = 256, reduction_ratio = 4)

        self.resblock4_a = ResBlock(in_channels = 256, out_channels = 512, stride = 2)
        self.resblock4_b = ResBlock(in_channels = 512, out_channels = 512, stride = 1)
        self.attention_4 = SE(in_channels = 512, reduction_ratio = 4)

        self.pool2 = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_features = 512, out_features = 1000, bias = True)

        self.attention_feature_maps = list(range(4))

    def forward(self, x):
        out = self.conv1(x)
        out = self.pool1(out)

        out = self.resblock1_a(out)
        out = self.resblock1_b(out)
        out = self.attention_1(out)
        self.attention_feature_maps[0] = out

        out = self.resblock2_a(out)
        out = self.resblock2_b(out)
        out = self.attention_2(out)
        self.attention_feature_maps[1] = out

        out = self.resblock3_a(out)
        out = self.resblock3_b(out)
        out = self.attention_3(out)
        self.attention_feature_maps[2] = out

        out = self.resblock4_a(out)
        out = self.resblock4_b(out)
        out = self.attention_4(out)
        self.attention_feature_maps[3] = out

        out = self.pool2(out)
        out = out.view(-1, 512)
        out = self.fc(out)
        return out
    
class IcarusNet_v2(IcarusNet):
    #loss = loss + sum(lambda * loss) | SRM Attention
    def __init__(self, args, writer):
        super().__init__(args, writer)

        self.teacher = pretrained_models.resnet101(pretrained = True)
        for param in self.teacher.parameters():
            param.requires_grad = False

        #ResNet 18 + SRM
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.pool1 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

        self.resblock1_a = ResBlock(in_channels = 64, out_channels = 64, stride = 1)
        self.resblock1_b = ResBlock(in_channels = 64, out_channels = 64, stride = 1)
        self.attention_1 = SRM(in_channels = 64)

        self.resblock2_a = ResBlock(in_channels = 64, out_channels = 128, stride = 2)
        self.resblock2_b = ResBlock(in_channels = 128, out_channels = 128, stride = 1)
        self.attention_2 = SRM(in_channels = 128)

        self.resblock3_a = ResBlock(in_channels = 128, out_channels = 256, stride = 2)
        self.resblock3_b = ResBlock(in_channels = 256, out_channels = 256, stride = 1)
        self.attention_3 = SRM(in_channels = 256)

        self.resblock4_a = ResBlock(in_channels = 256, out_channels = 512, stride = 2)
        self.resblock4_b = ResBlock(in_channels = 512, out_channels = 512, stride = 1)
        self.attention_4 = SRM(in_channels = 512)

        self.pool2 = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_features = 512, out_features = 1000, bias = True)

        self.attention_feature_maps = list(range(4))

    def forward(self, x):
        out = self.conv1(x)
        out = self.pool1(out)

        out = self.resblock1_a(out)
        out = self.resblock1_b(out)
        out = self.attention_1(out)
        self.attention_feature_maps[0] = out

        out = self.resblock2_a(out)
        out = self.resblock2_b(out)
        out = self.attention_2(out)
        self.attention_feature_maps[1] = out

        out = self.resblock3_a(out)
        out = self.resblock3_b(out)
        out = self.attention_3(out)
        self.attention_feature_maps[2] = out

        out = self.resblock4_a(out)
        out = self.resblock4_b(out)
        out = self.attention_4(out)
        self.attention_feature_maps[3] = out

        out = self.pool2(out)
        out = out.view(-1, 512)
        out = self.fc(out)
        return out

class IcarusNet_v3(IcarusNet):
    #loss = loss + sum(lambda * loss) | BAM Attention
    def __init__(self, args, writer):
        super().__init__(args, writer)

        self.teacher = pretrained_models.resnet101(pretrained = True)
        for param in self.teacher.parameters():
            param.requires_grad = False

        #ResNet 18 + BAM
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.pool1 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

        self.resblock1_a = ResBlock(in_channels = 64, out_channels = 64, stride = 1)
        self.resblock1_b = ResBlock(in_channels = 64, out_channels = 64, stride = 1)
        self.attention_1 = BAM(in_channels = 64, reduction_ratio = 4)

        self.resblock2_a = ResBlock(in_channels = 64, out_channels = 128, stride = 2)
        self.resblock2_b = ResBlock(in_channels = 128, out_channels = 128, stride = 1)
        self.attention_2 = BAM(in_channels = 128, reduction_ratio = 4)

        self.resblock3_a = ResBlock(in_channels = 128, out_channels = 256, stride = 2)
        self.resblock3_b = ResBlock(in_channels = 256, out_channels = 256, stride = 1)
        self.attention_3 = BAM(in_channels = 256, reduction_ratio = 4)

        self.resblock4_a = ResBlock(in_channels = 256, out_channels = 512, stride = 2)
        self.resblock4_b = ResBlock(in_channels = 512, out_channels = 512, stride = 1)
        self.attention_4 = BAM(in_channels = 512, reduction_ratio = 4)

        self.pool2 = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_features = 512, out_features = 1000, bias = True)

        self.attention_feature_maps = list(range(4))

    def forward(self, x):
        out = self.conv1(x)
        out = self.pool1(out)

        out = self.resblock1_a(out)
        out = self.resblock1_b(out)
        out = self.attention_1(out)
        self.attention_feature_maps[0] = out

        out = self.resblock2_a(out)
        out = self.resblock2_b(out)
        out = self.attention_2(out)
        self.attention_feature_maps[1] = out

        out = self.resblock3_a(out)
        out = self.resblock3_b(out)
        out = self.attention_3(out)
        self.attention_feature_maps[2] = out

        out = self.resblock4_a(out)
        out = self.resblock4_b(out)
        out = self.attention_4(out)
        self.attention_feature_maps[3] = out

        out = self.pool2(out)
        out = out.view(-1, 512)
        out = self.fc(out)
        return out
    
class IcarusNet_v4(IcarusNet):
    #loss = loss + sum(lambda * loss) | CBAM Attention
    def __init__(self, args, writer):
        super().__init__(args, writer)

        self.teacher = pretrained_models.resnet101(pretrained = True)
        for param in self.teacher.parameters():
            param.requires_grad = False

        #ResNet 18 + CBAM
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.pool1 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

        self.resblock1_a = ResBlock(in_channels = 64, out_channels = 64, stride = 1)
        self.resblock1_b = ResBlock(in_channels = 64, out_channels = 64, stride = 1)
        self.attention_1 = CBAM(in_channels = 64)

        self.resblock2_a = ResBlock(in_channels = 64, out_channels = 128, stride = 2)
        self.resblock2_b = ResBlock(in_channels = 128, out_channels = 128, stride = 1)
        self.attention_2 = CBAM(in_channels = 128)

        self.resblock3_a = ResBlock(in_channels = 128, out_channels = 256, stride = 2)
        self.resblock3_b = ResBlock(in_channels = 256, out_channels = 256, stride = 1)
        self.attention_3 = CBAM(in_channels = 256)

        self.resblock4_a = ResBlock(in_channels = 256, out_channels = 512, stride = 2)
        self.resblock4_b = ResBlock(in_channels = 512, out_channels = 512, stride = 1)
        self.attention_4 = CBAM(in_channels = 512)

        self.pool2 = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_features = 512, out_features = 1000, bias = True)

        self.attention_feature_maps = list(range(4))

    def forward(self, x):
        out = self.conv1(x)
        out = self.pool1(out)

        out = self.resblock1_a(out)
        out = self.resblock1_b(out)
        out = self.attention_1(out)
        self.attention_feature_maps[0] = out

        out = self.resblock2_a(out)
        out = self.resblock2_b(out)
        out = self.attention_2(out)
        self.attention_feature_maps[1] = out

        out = self.resblock3_a(out)
        out = self.resblock3_b(out)
        out = self.attention_3(out)
        self.attention_feature_maps[2] = out

        out = self.resblock4_a(out)
        out = self.resblock4_b(out)
        out = self.attention_4(out)
        self.attention_feature_maps[3] = out

        out = self.pool2(out)
        out = out.view(-1, 512)
        out = self.fc(out)
        return out

class IcarusNet_v5(IcarusNet):
    #loss = loss + sum(lambda * loss) | GE Attention
    def __init__(self, args, writer):
        super().__init__(args, writer)

        self.teacher = pretrained_models.resnet101(pretrained = True)
        for param in self.teacher.parameters():
            param.requires_grad = False

        #ResNet 18 + GE
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.pool1 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

        self.resblock1_a = ResBlock(in_channels = 64, out_channels = 64, stride = 1)
        self.resblock1_b = ResBlock(in_channels = 64, out_channels = 64, stride = 1)
        self.attention_1 = GE(in_channels = 64, location = 0)

        self.resblock2_a = ResBlock(in_channels = 64, out_channels = 128, stride = 2)
        self.resblock2_b = ResBlock(in_channels = 128, out_channels = 128, stride = 1)
        self.attention_2 = GE(in_channels = 128, location = 1)

        self.resblock3_a = ResBlock(in_channels = 128, out_channels = 256, stride = 2)
        self.resblock3_b = ResBlock(in_channels = 256, out_channels = 256, stride = 1)
        self.attention_3 = GE(in_channels = 256, location = 2)

        self.resblock4_a = ResBlock(in_channels = 256, out_channels = 512, stride = 2)
        self.resblock4_b = ResBlock(in_channels = 512, out_channels = 512, stride = 1)
        self.attention_4 = GE(in_channels = 512, location = 3)

        self.pool2 = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_features = 512, out_features = 1000, bias = True)

        self.attention_feature_maps = list(range(4))

    def forward(self, x):
        out = self.conv1(x)
        out = self.pool1(out)

        out = self.resblock1_a(out)
        out = self.resblock1_b(out)
        out = self.attention_1(out)
        self.attention_feature_maps[0] = out

        out = self.resblock2_a(out)
        out = self.resblock2_b(out)
        out = self.attention_2(out)
        self.attention_feature_maps[1] = out

        out = self.resblock3_a(out)
        out = self.resblock3_b(out)
        out = self.attention_3(out)
        self.attention_feature_maps[2] = out

        out = self.resblock4_a(out)
        out = self.resblock4_b(out)
        out = self.attention_4(out)
        self.attention_feature_maps[3] = out

        out = self.pool2(out)
        out = out.view(-1, 512)
        out = self.fc(out)
        return out

class MNasNet(MyNeuralNetwork):
    def __init__(self, args, writer):
        super().__init__(args, writer)

        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, stride = 2, padding = 1, bias = False)
        
        self.SepConv = SepConv(in_channels = 32, out_channels = 16, kernel_size = 3)
        
        self.MBConv6_1_a = MBConv(in_channels = 16, out_channels = 24, kernel_size = 3, stride = 2, padding = 1, expansion_ratio = 6) 
        self.MBConv6_1_b = MBConv(in_channels = 24, out_channels = 24, kernel_size = 3, stride = 1, padding = 1, expansion_ratio = 6)
        
        self.MBConv3_2_a = MBConv(in_channels = 24, out_channels = 40, kernel_size = 5, stride = 2, padding = 2, expansion_ratio = 3, attention = "SE")
        self.MBConv3_2_b = MBConv(in_channels = 40, out_channels = 40, kernel_size = 5, stride = 1, padding = 2, expansion_ratio = 3, attention = "SE")
        self.MBConv3_2_c = MBConv(in_channels = 40, out_channels = 40, kernel_size = 5, stride = 1, padding = 2, expansion_ratio = 3, attention = "SE")
        
        self.MBConv6_3_a = MBConv(in_channels = 40, out_channels = 80, kernel_size = 3, stride = 2, padding = 1, expansion_ratio = 6) 
        self.MBConv6_3_b = MBConv(in_channels = 80, out_channels = 80, kernel_size = 3, stride = 1, padding = 1, expansion_ratio = 6)
        self.MBConv6_3_c = MBConv(in_channels = 80, out_channels = 80, kernel_size = 3, stride = 1, padding = 1, expansion_ratio = 6)
        self.MBConv6_3_d = MBConv(in_channels = 80, out_channels = 80, kernel_size = 3, stride = 1, padding = 1, expansion_ratio = 6)
        
        self.MBConv6_4_a = MBConv(in_channels = 80, out_channels = 112, kernel_size = 3, stride = 1, padding = 1, expansion_ratio = 6, attention = "SE")
        self.MBConv6_4_b = MBConv(in_channels = 112, out_channels = 112, kernel_size = 3, stride = 1, padding = 1, expansion_ratio = 6, attention = "SE")
        
        self.MBConv6_5_a = MBConv(in_channels = 112, out_channels = 160, kernel_size = 5, stride = 2, padding = 2, expansion_ratio = 6, attention = "SE")
        self.MBConv6_5_b = MBConv(in_channels = 160, out_channels = 160, kernel_size = 5, stride = 1, padding = 2, expansion_ratio = 6, attention = "SE")
        self.MBConv6_5_c = MBConv(in_channels = 160, out_channels = 160, kernel_size = 5, stride = 1, padding = 2, expansion_ratio = 6, attention = "SE")
       
        self.MBConv6_6_a = MBConv(in_channels = 160, out_channels = 320, kernel_size = 3, stride = 1, padding = 1, expansion_ratio = 6)
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p = 0.2, inplace = True)
        self.fc = nn.Linear(in_features = 320, out_features = 1000, bias = True)
        #self.init_weights()

    def forward(self, x):
        out = self.conv1(x)
        
        out = self.SepConv(out)
        
        out = self.MBConv6_1_a(out)
        out = self.MBConv6_1_b(out)
        
        out = self.MBConv3_2_a(out)
        out = self.MBConv3_2_b(out)
        out = self.MBConv3_2_c(out)
        
        out = self.MBConv6_3_a(out)
        out = self.MBConv6_3_b(out)
        out = self.MBConv6_3_c(out)
        out = self.MBConv6_3_d(out)
        
        out = self.MBConv6_4_a(out)
        out = self.MBConv6_4_b(out)
        
        out = self.MBConv6_5_a(out)
        out = self.MBConv6_5_b(out)
        out = self.MBConv6_5_c(out)
        
        out = self.MBConv6_6_a(out)
        
        out = self.pool(out)
        out = out.view(-1, 320)
        out = self.dropout(out)
        out = self.fc(out)
        return out

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode = "fan_out", nonlinearity = "relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0.01)
                nn.init.zeros_(module.bias)

class EfficientNet(MyNeuralNetwork):
    def __init__(self, args, writer):
        super().__init__(args, writer)

        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, stride = 2, padding = 1, bias = False)
        
        self.MBConv1 = MBConv(in_channels = 32, out_channels = 16, kernel_size = 3, stride = 1, padding = 1, expansion_ratio = 1, attention = "SE")
        
        self.MBConv6_1_a = MBConv(in_channels = 16, out_channels = 24, kernel_size = 3, stride = 2, padding = 1, expansion_ratio = 6, attention = "SE")
        self.MBConv6_1_b = MBConv(in_channels = 24, out_channels = 24, kernel_size = 3, stride = 1, padding = 1, expansion_ratio = 6, attention = "SE")

        self.MBConv6_2_a = MBConv(in_channels = 24, out_channels = 40, kernel_size = 5, stride = 2, padding = 2, expansion_ratio = 6, attention = "SE")
        self.MBConv6_2_b = MBConv(in_channels = 40, out_channels = 40, kernel_size = 5, stride = 1, padding = 2, expansion_ratio = 6, attention = "SE")

        self.MBConv6_3_a = MBConv(in_channels = 40, out_channels = 80, kernel_size = 3, stride = 2, padding = 1, expansion_ratio = 6, attention = "SE")
        self.MBConv6_3_b = MBConv(in_channels = 80, out_channels = 80 ,kernel_size = 3, stride = 1, padding = 1, expansion_ratio = 6, attention = "SE")
        self.MBConv6_3_c = MBConv(in_channels = 80, out_channels = 80 ,kernel_size = 3, stride = 1, padding = 1, expansion_ratio = 6, attention = "SE")

        self.MBConv6_4_a = MBConv(in_channels = 80, out_channels = 112, kernel_size = 5, stride = 1, padding = 2, expansion_ratio = 6, attention = "SE")
        self.MBConv6_4_b = MBConv(in_channels = 112, out_channels = 112, kernel_size = 5, stride = 1, padding = 2, expansion_ratio = 6, attention = "SE")
        self.MBConv6_4_c = MBConv(in_channels = 112, out_channels = 112, kernel_size = 5, stride = 1, padding = 2, expansion_ratio = 6, attention = "SE")

        self.MBConv6_5_a = MBConv(in_channels = 112, out_channels = 192, kernel_size = 5, stride = 2, padding = 2, expansion_ratio = 6, attention = "SE")
        self.MBConv6_5_b = MBConv(in_channels = 192, out_channels = 192, kernel_size = 5, stride = 1, padding = 2, expansion_ratio = 6, attention = "SE")
        self.MBConv6_5_c = MBConv(in_channels = 192, out_channels = 192, kernel_size = 5, stride = 1, padding = 2, expansion_ratio = 6, attention =  "SE")
        self.MBConv6_5_d = MBConv(in_channels = 192, out_channels = 192, kernel_size = 5, stride = 1, padding = 2, expansion_ratio = 6, attention = "SE")

        self.MBConv6_6_a = MBConv(in_channels = 192, out_channels = 320, kernel_size = 3, stride = 1, padding = 1, expansion_ratio = 6, attention = "SE")
       
        self.conv2 = nn.Conv2d(in_channels = 320, out_channels = 1280, kernel_size = 1, stride = 1, bias = False)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p = 0.2, inplace = True)
        self.fc = nn.Linear(in_features = 1280, out_features = 1000, bias = True)

    def forward(self, x):
        out = self.conv1(x)

        out = self.MBConv1(out)
        out = self.MBConv6_1_a(out)
        out = self.MBConv6_1_b(out)
        
        out = self.MBConv6_2_a(out)
        out = self.MBConv6_2_b(out)
        
        out = self.MBConv6_3_a(out)
        out = self.MBConv6_3_b(out)
        out = self.MBConv6_3_c(out)
        
        out = self.MBConv6_4_a(out)
        out = self.MBConv6_4_b(out)
        out = self.MBConv6_4_c(out)
        
        out = self.MBConv6_5_a(out)
        out = self.MBConv6_5_b(out)
        out = self.MBConv6_5_c(out)
        out = self.MBConv6_5_d(out)
        
        out = self.MBConv6_6_a(out)
        
        out = self.conv2(out)
        
        out = self.pool(out)
        out = out.view(-1, 1280)
        out = self.dropout(out)
        out = self.fc(out)
        return out

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode = "fan_out", nonlinearity = "relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0.01)
                nn.init.zeros_(module.bias)

class ResNet_18(MyNeuralNetwork):
    def __init__(self, args, writer):
        super().__init__(args, writer)

        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.pool1 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

        self.resblock1_a = ResBlock(in_channels = 64, out_channels = 64, stride = 1)
        self.resblock1_b = ResBlock(in_channels = 64, out_channels = 64, stride = 1)

        self.resblock2_a = ResBlock(in_channels = 64, out_channels = 128, stride = 2)
        self.resblock2_b = ResBlock(in_channels = 128, out_channels = 128, stride = 1)

        self.resblock3_a = ResBlock(in_channels = 128, out_channels = 256, stride = 2)
        self.resblock3_b = ResBlock(in_channels = 256, out_channels = 256, stride = 1)

        self.resblock4_a = ResBlock(in_channels = 256, out_channels = 512, stride = 2)
        self.resblock4_b = ResBlock(in_channels = 512, out_channels = 512, stride = 1)

        self.pool2 = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_features = 512, out_features = 1000, bias = True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.pool1(out)

        out = self.resblock1_a(out)
        out = self.resblock1_b(out)

        out = self.resblock2_a(out)
        out = self.resblock2_b(out)

        out = self.resblock3_a(out)
        out = self.resblock3_b(out)

        out = self.resblock4_a(out)
        out = self.resblock4_b(out)

        out = self.pool2(out)
        out = out.view(-1, 512)
        out = self.fc(out)
        return out

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode = "fan_out", nonlinearity = "relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0.01)
                nn.init.zeros_(module.bias)

class ResNet_34(MyNeuralNetwork):
    def __init__(self, args, writer):
        super().__init__(args, writer)

        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.pool1 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

        self.resblock1_a = ResBlock(in_channels = 64, out_channels = 64, stride = 1)
        self.resblock1_b = ResBlock(in_channels = 64, out_channels = 64, stride = 1)
        self.resblock1_c = ResBlock(in_channels = 64, out_channels = 64, stride = 1)

        self.resblock2_a = ResBlock(in_channels = 64, out_channels = 128, stride = 2)
        self.resblock2_b = ResBlock(in_channels = 128, out_channels = 128, stride = 1)
        self.resblock2_c = ResBlock(in_channels = 128, out_channels = 128, stride = 1)
        self.resblock2_d = ResBlock(in_channels = 128, out_channels = 128, stride = 1)

        self.resblock3_a = ResBlock(in_channels = 128, out_channels = 256, stride = 2)
        self.resblock3_b = ResBlock(in_channels = 256, out_channels = 256, stride = 1)
        self.resblock3_c = ResBlock(in_channels = 256, out_channels = 256, stride = 1)
        self.resblock3_d = ResBlock(in_channels = 256, out_channels = 256, stride = 1)
        self.resblock3_e = ResBlock(in_channels = 256, out_channels = 256, stride = 1)
        self.resblock3_f = ResBlock(in_channels = 256, out_channels = 256, stride = 1)

        self.resblock4_a = ResBlock(in_channels = 256, out_channels = 512, stride = 2)
        self.resblock4_b = ResBlock(in_channels = 512, out_channels = 512, stride = 1)
        self.resblock4_c = ResBlock(in_channels = 512, out_channels = 512, stride = 1)

        self.pool2 = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_features = 512, out_features = 1000, bias = True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.pool1(out)

        out = self.resblock1_a(out)
        out = self.resblock1_b(out)
        out = self.resblock1_c(out)

        out = self.resblock2_a(out)
        out = self.resblock2_b(out)
        out = self.resblock2_c(out)
        out = self.resblock2_d(out)

        out = self.resblock3_a(out)
        out = self.resblock3_b(out)
        out = self.resblock3_c(out)
        out = self.resblock3_d(out)
        out = self.resblock3_e(out)
        out = self.resblock3_f(out)

        out = self.resblock4_a(out)
        out = self.resblock4_b(out)
        out = self.resblock4_c(out)

        out = self.pool2(out)
        out = out.view(-1, 512)
        out = self.fc(out)
        return out

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode = "fan_out", nonlinearity = "relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0.01)
                nn.init.zeros_(module.bias)

class ResNet_50(MyNeuralNetwork):
    def __init__(self, args, writer):
        super().__init__(args, writer)

        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.pool1 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

        self.resblock1_a = Bottleneck_Block(in_channels = 64, bottleneck_channels = 64, out_channels = 256, stride = 1)
        self.resblock1_b = Bottleneck_Block(in_channels = 256, bottleneck_channels = 64, out_channels = 256, stride = 1)
        self.resblock1_c = Bottleneck_Block(in_channels = 256, bottleneck_channels = 64, out_channels = 256, stride = 1)

        self.resblock2_a = Bottleneck_Block(in_channels = 256, bottleneck_channels = 128, out_channels = 512, stride = 2)
        self.resblock2_b = Bottleneck_Block(in_channels = 512, bottleneck_channels = 128, out_channels = 512, stride = 1)
        self.resblock2_c = Bottleneck_Block(in_channels = 512, bottleneck_channels = 128, out_channels = 512, stride = 1)
        self.resblock2_d = Bottleneck_Block(in_channels = 512, bottleneck_channels = 128, out_channels = 512, stride = 1)

        self.resblock3_a = Bottleneck_Block(in_channels = 512, bottleneck_channels = 256, out_channels = 1024, stride = 2)
        self.resblock3_b = Bottleneck_Block(in_channels = 1024, bottleneck_channels = 256, out_channels = 1024, stride = 1)
        self.resblock3_c = Bottleneck_Block(in_channels = 1024, bottleneck_channels = 256, out_channels = 1024, stride = 1)
        self.resblock3_d = Bottleneck_Block(in_channels = 1024, bottleneck_channels = 256, out_channels = 1024, stride = 1)
        self.resblock3_e = Bottleneck_Block(in_channels = 1024, bottleneck_channels = 256, out_channels = 1024, stride = 1)
        self.resblock3_f = Bottleneck_Block(in_channels = 1024, bottleneck_channels = 256, out_channels = 1024, stride = 1)

        self.resblock4_a = Bottleneck_Block(in_channels = 1024, bottleneck_channels = 512, out_channels = 2048, stride = 2)
        self.resblock4_b = Bottleneck_Block(in_channels = 2048, bottleneck_channels = 512, out_channels = 2048, stride = 1)
        self.resblock4_c = Bottleneck_Block(in_channels = 2048, bottleneck_channels = 512, out_channels = 2048, stride = 1)

        self.pool2 = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_features = 2048, out_features = 1000, bias = True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.pool1(out)
        out = self.resblock1_a(out)
        out = self.resblock1_b(out)
        out = self.resblock1_c(out)

        out = self.resblock2_a(out)
        out = self.resblock2_b(out)
        out = self.resblock2_c(out)
        out = self.resblock2_d(out)

        out = self.resblock3_a(out)
        out = self.resblock3_b(out)
        out = self.resblock3_c(out)
        out = self.resblock3_d(out)
        out = self.resblock3_e(out)
        out = self.resblock3_f(out)

        out = self.resblock4_a(out)
        out = self.resblock4_b(out)
        out = self.resblock4_c(out)

        out = self.pool2(out)
        out = out.view(-1, 2048)
        out = self.fc(out)
        return out

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode = "fan_out", nonlinearity = "relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0.01)
                nn.init.zeros_(module.bias)

class ResNet_101(MyNeuralNetwork):
    def __init__(self, args, writer):
        super().__init__(args, writer)

        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.pool1 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

        self.resblock1_a = Bottleneck_Block(in_channels = 64, bottleneck_channels = 64, out_channels = 256, stride = 1)
        self.resblock1_b = Bottleneck_Block(in_channels = 256, bottleneck_channels = 64, out_channels = 256, stride = 1)
        self.resblock1_c = Bottleneck_Block(in_channels = 256, bottleneck_channels = 64, out_channels = 256, stride = 1)

        self.resblock2_a = Bottleneck_Block(in_channels = 256, bottleneck_channels = 128, out_channels = 512, stride = 2)
        self.resblock2_b = Bottleneck_Block(in_channels = 512, bottleneck_channels = 128, out_channels = 512, stride = 1)
        self.resblock2_c = Bottleneck_Block(in_channels = 512, bottleneck_channels = 128, out_channels = 512, stride = 1)
        self.resblock2_d = Bottleneck_Block(in_channels = 512, bottleneck_channels = 128, out_channels = 512, stride = 1)

        self.resblock3_a = Bottleneck_Block(in_channels = 512, bottleneck_channels = 256, out_channels = 1024, stride = 2)
        self.resblock3_b = Bottleneck_Block(in_channels = 1024, bottleneck_channels = 256, out_channels = 1024, stride = 1)
        self.resblock3_c = Bottleneck_Block(in_channels = 1024, bottleneck_channels = 256, out_channels = 1024, stride = 1)
        self.resblock3_d = Bottleneck_Block(in_channels = 1024, bottleneck_channels = 256, out_channels = 1024, stride = 1)
        self.resblock3_e = Bottleneck_Block(in_channels = 1024, bottleneck_channels = 256, out_channels = 1024, stride = 1)

        self.resblock3_f = Bottleneck_Block(in_channels = 1024, bottleneck_channels = 256, out_channels = 1024, stride = 1)
        self.resblock3_g = Bottleneck_Block(in_channels = 1024, bottleneck_channels = 256, out_channels = 1024, stride = 1)
        self.resblock3_h = Bottleneck_Block(in_channels = 1024, bottleneck_channels = 256, out_channels = 1024, stride = 1)
        self.resblock3_i = Bottleneck_Block(in_channels = 1024, bottleneck_channels = 256, out_channels = 1024, stride = 1)
        self.resblock3_j = Bottleneck_Block(in_channels = 1024, bottleneck_channels = 256, out_channels = 1024, stride = 1)


        self.resblock3_k = Bottleneck_Block(in_channels = 1024, bottleneck_channels = 256, out_channels = 1024, stride = 1)
        self.resblock3_l = Bottleneck_Block(in_channels = 1024, bottleneck_channels = 256, out_channels = 1024, stride = 1)
        self.resblock3_m = Bottleneck_Block(in_channels = 1024, bottleneck_channels = 256, out_channels = 1024, stride = 1)
        self.resblock3_n = Bottleneck_Block(in_channels = 1024, bottleneck_channels = 256, out_channels = 1024, stride = 1)
        self.resblock3_o = Bottleneck_Block(in_channels = 1024, bottleneck_channels = 256, out_channels = 1024, stride = 1)

        self.resblock3_p = Bottleneck_Block(in_channels = 1024, bottleneck_channels = 256, out_channels = 1024, stride = 1)
        self.resblock3_q = Bottleneck_Block(in_channels = 1024, bottleneck_channels = 256, out_channels = 1024, stride = 1)
        self.resblock3_r = Bottleneck_Block(in_channels = 1024, bottleneck_channels = 256, out_channels = 1024, stride = 1)
        self.resblock3_s = Bottleneck_Block(in_channels = 1024, bottleneck_channels = 256, out_channels = 1024, stride = 1)
        self.resblock3_t = Bottleneck_Block(in_channels = 1024, bottleneck_channels = 256, out_channels = 1024, stride = 1)

        self.resblock3_u = Bottleneck_Block(in_channels = 1024, bottleneck_channels = 256, out_channels = 1024, stride = 1)
        self.resblock3_v = Bottleneck_Block(in_channels = 1024, bottleneck_channels = 256, out_channels = 1024, stride = 1)
        self.resblock3_w = Bottleneck_Block(in_channels = 1024, bottleneck_channels = 256, out_channels = 1024, stride = 1)

        self.resblock4_a = Bottleneck_Block(in_channels = 1024, bottleneck_channels = 512, out_channels = 2048, stride = 2)
        self.resblock4_b = Bottleneck_Block(in_channels = 2048, bottleneck_channels = 512, out_channels = 2048, stride = 1)
        self.resblock4_c = Bottleneck_Block(in_channels = 2048, bottleneck_channels = 512, out_channels = 2048, stride = 1)

        self.pool2 = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_features = 2048, out_features = 1000, bias = True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.pool1(out)
        out = self.resblock1_a(out)
        out = self.resblock1_b(out)
        out = self.resblock1_c(out)

        out = self.resblock2_a(out)
        out = self.resblock2_b(out)
        out = self.resblock2_c(out)
        out = self.resblock2_d(out)

        out = self.resblock3_a(out)
        out = self.resblock3_b(out)
        out = self.resblock3_c(out)
        out = self.resblock3_d(out)
        out = self.resblock3_e(out)

        out = self.resblock3_f(out)
        out = self.resblock3_g(out)
        out = self.resblock3_h(out)
        out = self.resblock3_i(out)
        out = self.resblock3_j(out)

        out = self.resblock3_k(out)
        out = self.resblock3_l(out)
        out = self.resblock3_m(out)
        out = self.resblock3_n(out)
        out = self.resblock3_o(out)

        out = self.resblock3_p(out)
        out = self.resblock3_q(out)
        out = self.resblock3_r(out)
        out = self.resblock3_s(out)
        out = self.resblock3_t(out)

        out = self.resblock3_u(out)
        out = self.resblock3_v(out)
        out = self.resblock3_w(out)

        out = self.resblock4_a(out)
        out = self.resblock4_b(out)
        out = self.resblock4_c(out)

        out = self.pool2(out)
        out = out.view(-1, 2048)
        out = self.fc(out)
        return out

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode = "fan_out", nonlinearity = "relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0.01)
                nn.init.zeros_(module.bias)

class MobileNet_v1(MyNeuralNetwork):
    def __init__(self, args, writer):
        super().__init__(args, writer)

        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, stride = 2, padding = 1, bias = False)

        self.depthwise_separable_conv1 = Depthwise_Separable_Conv(in_channels = 32, out_channels = 64, stride = 1)

        self.depthwise_separable_conv2 = Depthwise_Separable_Conv(in_channels = 64, out_channels = 128, stride = 2)

        self.depthwise_separable_conv3 = Depthwise_Separable_Conv(in_channels = 128, out_channels = 128, stride = 1)

        self.depthwise_separable_conv4 = Depthwise_Separable_Conv(in_channels = 128, out_channels = 256, stride = 2)

        self.depthwise_separable_conv5 = Depthwise_Separable_Conv(in_channels = 256, out_channels = 256, stride = 1)

        self.depthwise_separable_conv6 = Depthwise_Separable_Conv(in_channels = 256, out_channels = 512, stride = 2)

        self.depthwise_separable_conv7 = Depthwise_Separable_Conv(in_channels = 512, out_channels = 512, stride = 1)
        self.depthwise_separable_conv8 = Depthwise_Separable_Conv(in_channels = 512, out_channels = 512, stride = 1)
        self.depthwise_separable_conv9 = Depthwise_Separable_Conv(in_channels = 512, out_channels = 512, stride = 1)
        self.depthwise_separable_conv10 = Depthwise_Separable_Conv(in_channels = 512, out_channels = 512, stride = 1)
        self.depthwise_separable_conv11 = Depthwise_Separable_Conv(in_channels = 512, out_channels = 512, stride = 1)

        self.depthwise_separable_conv12 = Depthwise_Separable_Conv(in_channels = 512, out_channels = 1024, stride = 2)

        self.depthwise_separable_conv13 = Depthwise_Separable_Conv(in_channels = 1024, out_channels = 1024, stride = 1)

        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_features = 1024, out_features = 1000, bias = True)

    def forward(self, x):
        out = self.conv1(x)

        out = self.depthwise_separable_conv1(out)

        out = self.depthwise_separable_conv2(out)

        out = self.depthwise_separable_conv3(out)

        out = self.depthwise_separable_conv4(out)

        out = self.depthwise_separable_conv5(out)

        out = self.depthwise_separable_conv6(out)

        out = self.depthwise_separable_conv7(out)
        out = self.depthwise_separable_conv8(out)
        out = self.depthwise_separable_conv9(out)
        out = self.depthwise_separable_conv10(out)
        out = self.depthwise_separable_conv11(out)

        out = self.depthwise_separable_conv12(out)

        out = self.depthwise_separable_conv13(out)

        out = self.pool1(out)
        out = out.view(-1, 1024)
        out = self.fc(out)
        return out

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode = "fan_out", nonlinearity = "relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0.01)
                nn.init.zeros_(module.bias)

class MobileNet_v2(MyNeuralNetwork):
    def __init__(self, args, writer):
        super().__init__(args, writer)

        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, stride = 2, padding = 1, bias = False)
        
        self.MBConv1 = MBConv(in_channels = 32, out_channels = 16, kernel_size = 3, stride = 1, padding = 1, expansion_ratio = 1, residual = True)
        
        self.MBConv2 = MBConv(in_channels = 16, out_channels = 24, kernel_size = 3, stride = 2, padding = 1, expansion_ratio = 6, residual = False)
        self.MBConv3 = MBConv(in_channels = 24, out_channels = 24, kernel_size = 3, stride = 1, padding = 1, expansion_ratio = 6, residual = True)

        self.MBConv4 = MBConv(in_channels = 24, out_channels = 32, kernel_size = 3, stride = 2, padding = 1, expansion_ratio = 6, residual = False)
        self.MBConv5 = MBConv(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 1, padding = 1, expansion_ratio = 6, residual = True)
        self.MBConv6 = MBConv(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 1, padding = 1, expansion_ratio = 6, residual = True)

        self.MBConv7 = MBConv(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 2, padding = 1, expansion_ratio = 6, residual = False)
        self.MBConv8 = MBConv(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1, expansion_ratio = 6, residual = True)
        self.MBConv9 = MBConv(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1, expansion_ratio = 6, residual = True)
        self.MBConv10 = MBConv(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1, expansion_ratio = 6, residual = True)

        self.MBConv11 = MBConv(in_channels = 64, out_channels = 96, kernel_size = 3, stride = 1, padding = 1, expansion_ratio = 6, residual = True)
        self.MBConv12 = MBConv(in_channels = 96, out_channels = 96, kernel_size = 3, stride = 1, padding = 1, expansion_ratio = 6, residual = True)
        self.MBConv13 = MBConv(in_channels = 96, out_channels = 96, kernel_size = 3, stride = 1, padding = 1, expansion_ratio = 6, residual = True)

        self.MBConv14 = MBConv(in_channels = 96, out_channels = 160, kernel_size = 3, stride = 2, padding = 1, expansion_ratio = 6, residual = False)
        self.MBConv15 = MBConv(in_channels = 160, out_channels = 160, kernel_size = 3, stride = 1, padding = 1, expansion_ratio = 6, residual = True)
        self.MBConv16 = MBConv(in_channels = 160, out_channels = 160, kernel_size = 3, stride = 1, padding = 1, expansion_ratio = 6, residual = True)

        self.MBConv17 = MBConv(in_channels = 160, out_channels = 320, kernel_size = 3, stride = 1, padding = 1, expansion_ratio = 6, residual = True)

        self.conv2 = nn.Conv2d(in_channels = 320, out_channels = 1280, kernel_size = 1, stride = 1, bias = False)

        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p = 0.2)
        self.fc = nn.Linear(in_features = 1280, out_features = 1000, bias = True)

    def forward(self, x):
        out = self.conv1(x)

        out = self.MBConv1(out)

        out = self.MBConv2(out)
        out = self.MBConv3(out)

        out = self.MBConv4(out)
        out = self.MBConv5(out)
        out = self.MBConv6(out)

        out = self.MBConv7(out)
        out = self.MBConv8(out)
        out = self.MBConv9(out)
        out = self.MBConv10(out)

        out = self.MBConv11(out)
        out = self.MBConv12(out)
        out = self.MBConv13(out)

        out = self.MBConv14(out)
        out = self.MBConv15(out)
        out = self.MBConv16(out)

        out = self.MBConv17(out)

        out = self.conv2(out)

        out = self.pool1(out)
        out = out.view(-1, 1280)
        out = self.dropout(out)
        out = self.fc(out)
        return out

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode = "fan_out", nonlinearity = "relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0.01)
                nn.init.zeros_(module.bias)

class ShuffleNet_v1(MyNeuralNetwork):
    pass

class ShuffleNet_v2(MyNeuralNetwork):
    pass

class PeleeNet(MyNeuralNetwork):
    pass

class AmoebaNet(MyNeuralNetwork):
    pass

class ResNext(MyNeuralNetwork):
    pass

class DenseNet(MyNeuralNetwork):
    pass

class PyramidNet(MyNeuralNetwork):
    pass