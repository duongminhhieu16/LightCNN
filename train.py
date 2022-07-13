import os
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F

import numpy as np
from load_imglist import ImageList

from light_cnn import network_29layers_v2

model = network_29layers_v2(resblock, [1, 2, 3, 4], num_classes=8631)
model = torch.nn.DataParallel(model).cuda()

params = []
lr = 0.0001 # learning rate
for name, value in model.named_parameters():
    # print(value)
    if 'bias' in name:
        if 'fc2' in name:
            params += [{'params':value, 'lr': 20 * lr, 'weight_decay': 0}]
        else:
            params += [{'params':value, 'lr': 2 * lr, 'weight_decay': 0}]
    else:
        if 'fc2' in name:
            params += [{'params':value, 'lr': 10 * lr}]
        else:
            params += [{'params':value, 'lr': 1 * lr}]
            
# define loss function and optimizer
criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(params, lr, momentum=0.95,
                            weight_decay =1e-5)

#load dataset
train_path = "/home/jovyan/vggface2_train"
test_path = "/raid/vkistuser/hieu/vggface2_test"
train_list = "/home/jovyan/LightCNN/train_list.txt"
test_list = "/home/jovyan/LightCNN/test_list.txt"
batch_size = 512
workers = 8
train_loader = torch.utils.data.DataLoader(
    ImageList(root=train_path, fileList=train_list, 
        transform=transforms.Compose([ 
            transforms.Resize((128, 128)),
            transforms.RandomCrop(128),
            transforms.RandomHorizontalFlip(), 
            transforms.ToTensor(),
        ])),
    batch_size=batch_size, shuffle=True,
    num_workers=workers, pin_memory=False)

val_loader = torch.utils.data.DataLoader(
    ImageList(root=test_path, fileList=test_list, 
        transform=transforms.Compose([ 
            transforms.Resize((128, 128)),
            # transforms.RandomCrop(128),
            # transforms.RandomHorizontalFlip(), 
            transforms.ToTensor(),
        ])),
    batch_size=batch_size, shuffle=False,
    num_workers=workers, pin_memory=False)  
# device = torch.device('cpu')
cp = "/home/jovyan/LightCNN-master/VGG_full/LightCNN-29_SGD_epoch500_checkpoint.pth.tar" # path to checkpoint
if os.path.isfile(cp):
    print("=> loading checkpoint '{}'".format(cp))
    checkpoint = torch.load(cp)
    start_epoch = checkpoint['epoch']

    state_dict = checkpoint['state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module' not in k:
               k = 'module.' + k
        else:
            k = k.replace('features.module.', 'module.features.')
        new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
            
    print("=> loaded checkpoint '{}' (epoch {})"
                .format(cp, checkpoint['epoch']))
else:
    print("=> no checkpoint found at '{}'".format(cp))


epochs = 0

for epoch in range(epochs, 700):
    checkpoint = "/home/jovyan/LightCNN-master/VGG_full/LightCNN-29_SGD" + "_epoch" + str(epoch+1) + "_checkpoint.pth.tar"
    epoch_loss = 0
    epoch_accuracy = 0
    t0 = time.time()
    idx = 0
    for data, label in train_loader:
        idx += 1
        t1 = time.time()
        data = data.cuda()
        label = label.cuda()
        input_var  = torch.autograd.Variable(data)
        target_var = torch.autograd.Variable(label)
        print(len(train_loader))
        output, _ = model(input_var)
        loss = criterion(output, target_var)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        acc = ((output.argmax(dim=1) == label).float().mean())
        epoch_accuracy += acc/len(train_loader)
        epoch_loss += loss/len(train_loader)
        # print(time.time()-t1)
        print(idx)
    training_time = time.time()
    total_time = training_time - t0
    

    print('Epoch : {}, train accuracy : {}, train loss : {}, total training time : {}'.format(epoch+1, epoch_accuracy, epoch_loss, total_time))
    
    
    with torch.no_grad():
        epoch_val_accuracy=0
        epoch_val_loss =0
        t0 = time.time()
        for data, label in val_loader:
            data = data.cuda()
            label = label.cuda()
            
            val_output, _ = model(data)
            val_loss = criterion(val_output,label)
            
            
            acc = ((val_output.argmax(dim=1) == label).float().mean())
            epoch_val_accuracy += acc/ len(val_loader)
            epoch_val_loss += val_loss/ len(val_loader)
        eval_time = time.time()
        total_eval_time = eval_time - t0
        print('Epoch : {}, val_accuracy : {}, val_loss : {}, total evaluating time: {}'.format(epoch+1, epoch_val_accuracy, epoch_val_loss, total_eval_time))
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'acc': epoch_accuracy,
            'val_acc': epoch_val_accuracy,
            'total_time': total_time
        }, checkpoint)
