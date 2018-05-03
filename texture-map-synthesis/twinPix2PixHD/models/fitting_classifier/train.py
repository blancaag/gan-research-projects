from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import models, transforms
# import matplotlib.pyplot as plt
import time
import os
from PIL import Image

use_gpu = torch.cuda.is_available()
torch.cuda.is_available()

def on_hot_tensor(x):
    x_ = preds.view(len(x), 1)
    if use_gpu: 
        x_onehot = torch.cuda.FloatTensor(len(x_), 2).zero_().scatter_(1, x_, 1)
    else: x_onehot = torch.FloatTensor(len(x_), 2).zero_().scatter_(1, x_, 1)     
    return x_onehot  


def train_model(model, dataloaders, criterion, opt, datasets_sizes):
    
    if use_gpu: assert(torch.cuda.is_available())
    
    for param in model.parameters(): param.requires_grad = True
    
    train_hist = {phase: {'loss': [], 'acc': []} for phase in opt.phases}
    
    since = time.time()

    best_model_wts = model.state_dict()
    best_loss = 0.0
    best_loss_acc = 0.0
    ##
    best_acc = 0.0

    for epoch in range(opt.nepochs):
        print('Epoch {}/{}'.format(epoch, opt.nepochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in opt.phases:
            if phase == 'train':
                model.module.scheduler.step(); print(model.module.scheduler.get_lr())
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                paths, inputs, labels = data


                # ON_HOT LABELS IF CRITERION REQUIRES IT
                # print (labels, type(labels))
                # print(criterion)
                if isinstance(criterion, torch.nn.modules.loss.BCEWithLogitsLoss):
                        y = labels.view(len(labels), 1)
                        y_onehot = torch.FloatTensor(len(y), 2).zero_().scatter_(1, y, 1)
                        labels = y_onehot
                        
                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                model.module.optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                # print(outputs)
                _, preds = torch.max(outputs.data, 1)
                
                if isinstance(criterion, torch.nn.modules.loss.BCEWithLogitsLoss):
                    y = preds.view(len(preds), 1)
                    if use_gpu: y_onehot = torch.cuda.FloatTensor(len(y), 2).zero_().scatter_(1, y, 1)
                    else: y_onehot = torch.FloatTensor(len(y), 2).zero_().scatter_(1, y, 1)     
                    preds = y_onehot
                
                # print(preds, labels)
                loss = criterion(outputs, labels)
                acc = torch.sum(preds == labels.data) / (opt.batch_size * preds.shape[1])
                # print(loss, acc)
                
                # import torch.nn as nn
                # criterion1 = nn.CrossEntropyLoss()
                # criterion2 = nn.BCELoss()
                # criterion3 = nn.BCEWithLogitsLoss()
                # criterion4 = nn.MSELoss()
                #
                # l1 = criterion1(self.class_fit, 1)
                # l2 = criterion2(self.class_fit, 1)
                # l3 = criterion3(self.class_fit, 1)
                # l4 = criterion4(self.class_fit, 1)
                #
                # print(l1, l2, l3, l4)
                
#                 print('Batch loss/acc: %f/%f' %(loss, acc))
                
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    model.module.optimizer.step()

                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data) / preds.shape[1]
                
                train_hist[phase]['loss'].append(loss.data[0])
                train_hist[phase]['acc'].append(acc)

            epoch_loss = running_loss / datasets_sizes[phase]
            epoch_acc = running_corrects / datasets_sizes[phase]
            
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            # if phase == 'val' and epoch_acc > best_acc:
                # best_acc = epoch_acc
                # best_acc_loss = epoch_loss
                # best_model_wts = model.state_dict()
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_loss_acc = epoch_acc
                best_model_wts = model.state_dict()

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    
    output_dir = os.path.join(opt.checkpoint_dir, opt.name)
    
    torch.save(model.state_dict(), 
               os.path.join(output_dir, 'model_best_c%d_%f_%f_%d.pth.tar' %(opt.nc_input, best_loss, best_loss_acc, int(time.time()/1000))))
    torch.save(model, 
               os.path.join(output_dir, 'model_best_c%d_%f_%f_%d.pt' %(opt.nc_input, best_loss, best_loss_acc, int(time.time()/1000))))

    return model