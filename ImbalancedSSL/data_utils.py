import torch
import torchvision
from torchvision import datasets
from torch.utils.data import sampler, DataLoader
from torch.utils.data.sampler import BatchSampler
import torch.distributed as dist

import numpy as np
import random
import copy
import math
from fractions import Fraction

from ImbalancedSSL.DistributedProxySampler import DistributedProxySampler


def split_ssl_data(data, target, num_labels, num_classes, index=None, include_lb_to_ulb=True):
    """
    data & target is splitted into labeled and unlabeld data.
    
    Args
        index: If np.array of index is given, select the data[index], target[index] as labeled samples.
        include_lb_to_ulb: If True, labeled data is also included in unlabeld data
    """
    data, target = np.array(data), np.array(target)
    lb_data, lbs, lb_idx = sample_labeled_data(data, target, num_labels, num_classes, index)
    ulb_idx = np.array(sorted(list(set(range(len(data))) - set(lb_idx)))) #unlabeled_data index of data
    if include_lb_to_ulb:
        return lb_data, lbs, data, target
    else:
        return lb_data, lbs, data[ulb_idx], target[ulb_idx]


def split_ssl_dataset_imagenet(dset, total_samples):
    '''
    dset: a torch ImageFolder Dataset
    total_samples: total labeled samples to use
    
    Returns a dataset with the same number of samples as in total_samples(arg) and the entire dataset 
    '''
    print("Splitting the SSL dataset, seems to be an ImageFolder type dataset...")
    print("the total number of labelled samples taken are (pre long tail):", total_samples)
    dataset = copy.deepcopy(dset)
    selected_samples = {}
    num_classes = len(dataset.classes)
    for i in range(num_classes):
        selected_samples[i] = []

    for samp in dataset.samples:
        selected_samples[samp[1]].append(samp)

    print("The number of samples per class is (pre long tail):", total_samples//num_classes)
    for i in range(num_classes):
        num_samples = total_samples//num_classes
        selected_samples[i] = selected_samples[i][:num_samples]
    
    dataset.samples = []
    for i in range(num_classes):
        dataset.samples += selected_samples[i]
    
    dataset.imgs = dataset.samples
    dataset.targets = [samp[1] for samp in dataset.samples]
    return dataset, dset


def sample_labeled_data(data, target, 
                         num_labels,
                         num_classes,
                         index=None):
    '''
    samples for labeled data
    (sampling with balanced ratio over classes)
    '''
    assert num_labels % num_classes == 0
    if not index is None:
        index = np.array(index, dtype=np.int32)
        return data[index], target[index], index

    samples_per_class = int(num_labels / num_classes)

    lb_data = []
    lbs = []
    lb_idx = []
    for c in range(num_classes):
        idx = np.where(target == c)[0]
        idx = np.random.choice(idx, samples_per_class, False)
        lb_idx.extend(idx)

        lb_data.extend(data[idx])
        lbs.extend(target[idx])

    return np.array(lb_data), np.array(lbs), np.array(lb_idx)


def get_sampler_by_name(name):
    '''
    get sampler in torch.utils.data.sampler by name
    '''
    sampler_name_list = sorted(name for name in torch.utils.data.sampler.__dict__ 
                      if not name.startswith('_') and callable(sampler.__dict__[name]))
    try:
        if name == 'DistributedSampler':
            return torch.utils.data.distributed.DistributedSampler
        else:
            return getattr(torch.utils.data.sampler, name)
    except Exception as e:
        print(repr(e))
        print('[!] select sampler in:\t', sampler_name_list)


def get_data_loader(dset,
                    batch_size = None,
                    shuffle = False,
                    num_workers = 4,
                    pin_memory = True,
                    data_sampler = None,
                    replacement = True,
                    num_epochs = None,
                    num_iters = None,
                    generator = None,
                    drop_last=True,
                    distributed=False):
    """
    get_data_loader returns torch.utils.data.DataLoader for a Dataset.
    All arguments are comparable with those of pytorch DataLoader.
    However, if distributed, DistributedProxySampler, which is a wrapper of data_sampler, is used.
    
    Args
        num_epochs: total batch -> (# of batches in dset) * num_epochs 
        num_iters: total batch -> num_iters
    """
    
    assert batch_size is not None
        
    if data_sampler is None:
        return DataLoader(dset, batch_size=batch_size, shuffle=shuffle, 
                          num_workers=num_workers, pin_memory=pin_memory)

    else:
        if isinstance(data_sampler, str):
            data_sampler = get_sampler_by_name(data_sampler)
        
        if distributed:
            assert dist.is_available()
            num_replicas = dist.get_world_size()
        else:
            num_replicas = 1
        
        if (num_epochs is not None) and (num_iters is None):
            num_samples = len(dset)*num_epochs
        elif (num_epochs is None) and (num_iters is not None):
            num_samples = batch_size * num_iters * num_replicas
        else:
            num_samples = len(dset)
        
        if data_sampler.__name__ == 'RandomSampler':    
            data_sampler = data_sampler(dset, replacement, num_samples, generator)
        else:
            raise RuntimeError(f"{data_sampler.__name__} is not implemented.")
        
        if distributed:
            '''
            Different with DistributedSampler, 
            the DistribuedProxySampler does not shuffle the data (just wrapper for dist).
            '''
            data_sampler = DistributedProxySampler(data_sampler)

        batch_sampler = BatchSampler(data_sampler, batch_size, drop_last)
        return DataLoader(dset, batch_sampler=batch_sampler, 
                          num_workers=num_workers, pin_memory=pin_memory)

    
def get_onehot(num_classes, idx):
    onehot = np.zeros([num_classes], dtype=np.float32)
    onehot[idx] += 1.0
    return onehot


def split(dataset):
    print("Creating a 50/50 sample split for a cifar like dataset")
    split_size=0.5
    dataset_len = len(dataset)
    splits = int(split_size * dataset_len )
    indices = list(range(dataset_len))
    random.shuffle(indices)
    
    val_idx, test_idx = indices[:splits], indices[splits:]
    valset, testset = copy.deepcopy(dataset), copy.deepcopy(dataset)
    
    valset.data=dataset.data[np.array(val_idx)]
    valset.targets=list(dataset.targets[x] for x in val_idx)

    testset.data=dataset.data[np.array(test_idx)]
    testset.targets=list(dataset.targets[x] for x in test_idx)
    return testset, valset


def split_imagenet(dataset):
    print("Creating a 50/50 split of ImageFolder type dataset")
    len_dataset = len(dataset.samples)
    dset1, dset2 = copy.deepcopy(dataset), copy.deepcopy(dataset)
    samples1, samples2 = [], []
    for i, samp in enumerate(dataset.samples):
        if i%2:
            samples1.append(samp)
        else:
            samples2.append(samp)
    dset1.samples, dset1.imgs  = samples1, samples1
    dset1.targets = [samp[1] for samp in samples1]

    dset2.samples, dset2.imgs = samples2, samples2
    dset2.targets = [samp[1] for samp in samples2]
    return dset1, dset2


import random
import numpy as np
import torch
from math import floor


def SSLSplitCIFAR(targets, num_classes, N1, M1, val_size,\
                    rho_l, rho_u, inc_lb, seed=0, imb_type="exp"):
    def get_random_states():
        return random.getstate(), torch.get_rng_state(), np.random.get_state()

    def set_random_states(random_state, torch_state, np_state):
        random.setstate(random_state)
        torch.set_rng_state(torch_state)
        np.random.set_state(np_state)
        return

    def samp_per_class(N1, M1, rho_l, rho_u, num_classes, val_size, imb_type="exp"):
        if imb_type == "exp":
            gamma_l = (rho_l)**(-1/(num_classes-1))
            lb_samp_per_class = [floor(N1 * (gamma_l**i))\
                                    for i in range(num_classes)]
            
            gamma_u = (rho_u)**(-1/(num_classes-1))
            ulb_samp_per_class = [floor(M1 * (gamma_u**i))\
                                    for i in range(num_classes)]
        else :
            lb_samp_per_class = [floor(N1 * (1/rho_l) * (1- i/num_classes))\
                                                for i in range(num_classes)]
            ulb_samp_per_class = [floor(M1 * (1/rho_u) * (1- i/num_classes))\
                                                for i in range(num_classes)]

        val_samp_per_class = [floor(val_size//num_classes)\
                                    for i in range(num_classes)]

        return lb_samp_per_class, ulb_samp_per_class, val_samp_per_class

    random_state, torch_state, np_state = get_random_states()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    lb_samp_per_class, ulb_samp_per_class, val_samp_per_class = samp_per_class(
        N1, M1, rho_l, rho_u, num_classes, val_size, imb_type
    )

    targets = np.array(targets)
    lb_idx, ulb_idx, val_idx = [], [], []

    for c in range(num_classes):
        c_targets =  np.where(targets == c)[0].tolist()
        if len(c_targets) < lb_samp_per_class[c] + ulb_samp_per_class[c] + val_samp_per_class[c]:
            raise Exception("Too few samples present")
        lb_idx = lb_idx + c_targets[0:lb_samp_per_class[c]]
        ulb_idx = ulb_idx + c_targets[lb_samp_per_class[c] :\
                                    lb_samp_per_class[c] + ulb_samp_per_class[c]]
        val_idx = val_idx + c_targets[lb_samp_per_class[c] + ulb_samp_per_class[c]:\
                                      lb_samp_per_class[c] + ulb_samp_per_class[c] + val_samp_per_class[c]]
    if inc_lb:
        ulb_idx = ulb_idx + lb_idx

    set_random_states(random_state, torch_state, np_state)
    return lb_idx, ulb_idx, val_idx