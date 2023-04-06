import numpy as np
import copy

from PIL import Image

from torch.utils.data import Dataset
from torchvision import datasets, transforms

from .data_utils import SSLSplitCIFAR

from .augmentation.weak_transfroms import get_transform
from .augmentation.randaugment import RandAugment
from .data_utils import get_onehot

from .dataset import BasicDataset

def get_cifar10(root, N1, M1, val_size,\
                rho_l, rho_u, inc_lb, seed=0, imb_type="exp"):
    '''
    Generate the long tailed SSL dataset
    Args:
        root: root directeroy where the dataset files are stored
        N1: number of samples in class index 0 in labeled set
        M1: number of samples in class index 0 in unlabeled set
        val_size: size of the validation set
        rho_l: imbalance of the labeled set (N1/Nk)
        rho_u: imbalance of the unlabeled set (M1/Mk)
        inc_lb: whether to include the labeled set in unlabeled set or not
        seed: random seed
        imb_type: nature of the long tail distribution
    '''
    # get the transformation for the labeled set, val set and test set
    # the unlabeled set has sgtrong augmentations too
    test_transform = get_transform("cifar10", False)
    lb_transform = get_transform("cifar10", True)

    # the dataset_train will be used generate the labeled, unlabeled and val set
    dataset_train = datasets.CIFAR10(root, True, download=True, transform=lb_transform)
    lb_idx, ulb_idx, val_idx = SSLSplitCIFAR(targets=dataset_train.targets, num_classes=10,\
                                             N1=N1, M1=M1, val_size=val_size, rho_l=rho_l,\
                                             rho_u=rho_u, inc_lb=inc_lb, seed=seed, imb_type=imb_type)
    lb_data, lb_targets = dataset_train.data[lb_idx], np.array(dataset_train.targets)[lb_idx]
    ulb_data, ulb_targets = dataset_train.data[ulb_idx], np.array(dataset_train.targets)[ulb_idx]
    val_data, val_targets = dataset_train.data[val_idx], np.array(dataset_train.targets)[val_idx]

    lb_dataset = BasicDataset(lb_data, lb_targets, 10, lb_transform, use_strong_transform=False)
    ulb_dataset = BasicDataset(ulb_data, ulb_targets, 10, lb_transform, use_strong_transform=True)
    val_dataset = BasicDataset(val_data, val_targets, 10, lb_transform, use_strong_transform=False)

    test_dataset = datasets.CIFAR10(root, False, download=True, transform=test_transform)
    return lb_dataset, ulb_dataset, val_dataset, test_dataset
