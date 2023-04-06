import json
import math, random
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import BatchSampler

import torchvision
from torchvision import datasets, transforms



def get_transform(dataset_name, train=False):
    '''
    returns the transformation to be applied on a given 
    dataset, currently supported imagenet-100, cifar-10 and cifar-100
    '''
    mean, std, crop = {}, {}, {}
    mean['cifar10'] = [x / 255 for x in [125.3, 123.0, 113.9]]
    mean['cifar100'] = [x / 255 for x in [129.3, 124.1, 112.4]]
    mean['imagenet100'] = [0.485, 0.456, 0.406] 

    std['cifar10'] = [x / 255 for x in [63.0, 62.1, 66.7]]
    std['cifar100'] = [x / 255 for x in [68.2,  65.4,  70.4]]
    std['imagenet100'] = [0.229, 0.224, 0.225]

    crop['cifar10'] = 32
    crop['cifar100'] = 32
    crop['imagenet100'] = 224

    if dataset_name == "imagenet":
        if train:
            return transforms.Compose([transforms.RandomHorizontalFlip(),
                                        transforms.RandomResizedCrop(crop[dataset_name]),
                                        transforms.ToTensor(), 
                                        transforms.Normalize(mean[dataset_name], std[dataset_name])])
        else:
            return  transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(crop[dataset_name]),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean[dataset_name], std[dataset_name])])
    elif "cifar" in dataset_name:
        if train:
            return transforms.Compose([transforms.RandomHorizontalFlip(),
                                      transforms.RandomCrop(crop[dataset_name], padding=4),
                                      transforms.ToTensor(), 
                                      transforms.Normalize(mean[dataset_name], std[dataset_name])])
        else:
            return transforms.Compose([transforms.ToTensor(), 
                                        transforms.Normalize(mean[dataset_name], std[dataset_name])])
