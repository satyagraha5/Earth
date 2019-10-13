def fit(self, train_loader, test_loader, student_criterion, attention_criterion, student_optimizer, num_epochs):
        for epoch in tqdm(range(1, num_epochs + 1)):
            print()
            #Train
            for i, (train_images, train_labels) in enumerate(train_loader, start = 1):
                try:
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
                        for i, (val_images, val_labels) in enumerate(test_loader, start = 1):
                            with torch.no_grad():
                                self.eval()
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
                except IndexError:
                    continue