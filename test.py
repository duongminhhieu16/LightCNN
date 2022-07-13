'''
    implement training process for Light CNN
    @author: Alfred Xiang Wu
    @date: 2017.07.04
'''
from __future__ import print_function
import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import numpy as np

from light_cnn import LightCNN_9Layers, LightCNN_29Layers, LightCNN_test, LightCNN_29Layers_v2
from load_imglist import ImageList

parser = argparse.ArgumentParser(description='PyTorch Light CNN Training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='LightCNN')
parser.add_argument('--cuda', '-c', default=True)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--epochs', default=80, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 100)')
parser.add_argument('--model', default='', type=str, metavar='Model',
                    help='model type: LightCNN-9, LightCNN-29, LightCNN-29v2')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--root_path', default='', type=str, metavar='PATH',
                    help='path to root path of images (default: none)')
parser.add_argument('--train_list', default='', type=str, metavar='PATH',
                    help='path to training list (default: none)')
parser.add_argument('--val_list', default='', type=str, metavar='PATH',
                    help='path to validation list (default: none)')
parser.add_argument('--test_list', default='', type=str, metavar='PATH',
                    help='path to test list (default: none)')
parser.add_argument('--save_path', default='', type=str, metavar='PATH',
                    help='path to save checkpoint (default: none)')
parser.add_argument('--num_classes', default=99891, type=int,
                    metavar='N', help='number of classes (default: 99891)')

def main():
    global args
    args = parser.parse_args()

    # create Light CNN for face recognition
    if args.model == 'LightCNN-9':
        model = LightCNN_9Layers(num_classes=args.num_classes)
    elif args.model == 'LightCNN-29':
        model = LightCNN_29Layers(num_classes=args.num_classes)
    elif args.model == 'LightCNN-29v2':
        model = LightCNN_29Layers_v2(num_classes=args.num_classes)
    elif args.model == 'LightCNN-test':
        model = LightCNN_test(num_classes=args.num_classes)
    else:
        print('Error model type\n')

##    if args.cuda:
##        model = torch.nn.DataParallel(model).cuda()

    print(model)

    # large lr for last fc parameters
    params = []
    for name, value in model.named_parameters():
        if 'bias' in name:
            if 'fc2' in name:
                params += [{'params':value, 'lr': 20 * args.lr, 'weight_decay': 0}]
            else:
                params += [{'params':value, 'lr': 2 * args.lr, 'weight_decay': 0}]
        else:
            if 'fc2' in name:
                params += [{'params':value, 'lr': 10 * args.lr}]
            else:
                params += [{'params':value, 'lr': 1 * args.lr}]

    optimizer = torch.optim.SGD(params, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']

##            restore_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items()}
            model.load_state_dict(checkpoint['state_dict'])
            
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    #load image
    test_loader = torch.utils.data.DataLoader(
        ImageList(root=args.root_path, fileList=args.test_list, 
            transform=transforms.Compose([ 
                transforms.RandomCrop(128),
                transforms.RandomHorizontalFlip(), 
                transforms.ToTensor(),
            ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    
    for i, (input, target) in enumerate(test_loader):
        total, correct = 0, 0
        with torch.no_grad():
            input_var  = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)
        output, _ = model(input_var)
##        print(target)
        _, pred = torch.max(output.data, 1)
        for t, p in zip(target_var.view(-1), pred.view(-1)):
            total += target_var.size(0)
            correct += (pred==target_var).sum().item()
            print("pred: " + str(pred))
            print("target: " + str(target_var))
            print("\n")
    print(100*correct/total)
        

if __name__ == '__main__':
    main()
