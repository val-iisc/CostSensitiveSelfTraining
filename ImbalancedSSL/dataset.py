from torchvision import datasets, transforms
from torch.utils.data import Dataset
from .data_utils import get_onehot
from .augmentation.randaugment import RandAugment

from PIL import Image
import numpy as np
import copy


class BasicDataset(Dataset):
    """
    BasicDataset returns a pair of image and labels (targets).
    If targets are not given, BasicDataset returns None as the label.
    This class supports strong augmentation for Fixmatch,
    and return both weakly and strongly augmented images.
    """
    def __init__(self,
                 data,
                 targets=None,
                 num_classes=None,
                 transform=None,
                 use_strong_transform=False,
                 strong_transform=None,
                 onehot=False,
                 *args, **kwargs):
        """
        Args
            data: x_data
            targets: y_data (if not exist, None)
            num_classes: number of label classes
            transform: basic transformation of data
            use_strong_transform: If True, this dataset returns both weakly and strongly augmented images.
            strong_transform: list of transformation functions for strong augmentation
            onehot: If True, label is converted into onehot vector.
        """
        super(BasicDataset, self).__init__()
        self.data = data
        self.targets = targets
        
        self.num_classes = num_classes
        self.use_strong_transform = use_strong_transform
        self.onehot = onehot
        
        self.transform = transform
        if use_strong_transform:
            if strong_transform is None:
                self.strong_transform = copy.deepcopy(transform)
                self.strong_transform.transforms.insert(0, RandAugment(3,5))
        else:
            self.strong_transform = strong_transform
                
    
    def __getitem__(self, idx):
        """
        If strong augmentation is not used,
            return weak_augment_image, target
        else:
            return weak_augment_image, strong_augment_image, target
        """
        
        #set idx-th target
        if self.targets is None:
            target = None
        else:
            target_ = self.targets[idx]
            target = target_ if not self.onehot else get_onehot(self.num_classes, target_)
            
        #set augmented images
            
        img = self.data[idx]
        if self.transform is None:
            return transforms.ToTensor()(img), target
        else:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            img_w = self.transform(img)
            if not self.use_strong_transform:
                return img_w, target
            else:
                return img_w, self.strong_transform(img), target

    def __len__(self):
        return len(self.data)


class BasicDatasetImagenet(Dataset):
    """
    BasicDataset returns a pair of image and labels (targets).
    If targets are not given, BasicDataset returns None as the label.
    This class supports strong augmentation for Fixmatch,
    and return both weakly and strongly augmented images.
    """
    def __init__(self,
                 dataset,
                 use_strong_transform=False,
                 strong_transform=None,
                 onehot=False,
                 *args, **kwargs):
        """
        Args
            dataset: ImageFolderDataset
            use_strong_transform: If True, this dataset returns both weakly and strongly augmented images.
            strong_transform: list of transformation functions for strong augmentation
            onehot: If True, label is converted into onehot vector.
        """
        super(BasicDatasetImagenet, self).__init__()
        self.root = dataset.root
        self.transform = dataset.transform
        self.target_transform = dataset.target_transform
        self.transforms = dataset.transforms
        self.loader = dataset.loader
        self.extensions = dataset.extensions
        self.classes = dataset.classes
        self.class_to_idx = dataset.class_to_idx
        self.samples = dataset.samples
        self.targets = dataset.targets
        self.imgs = dataset.imgs

        self.num_classes = len(self.classes)
        self.use_strong_transform = use_strong_transform
        self.onehot = onehot

        if use_strong_transform:
            if strong_transform is None:
                self.strong_transform = copy.deepcopy(self.transform)
                self.strong_transform.transforms.insert(0, RandAugment(3,5))
        else:
            self.strong_transform = strong_transform

    def __getitem__(self, idx):
        """
        If strong augmentation is not used,
            return weak_augment_image, target
        else:
            return weak_augment_image, strong_augment_image, target
        """

        #set idx-th target
        if self.targets is None:
            target = None
        else:
            target_ = self.targets[idx]
            target = target_ if not self.onehot else get_onehot(self.num_classes, target_)
        #set augmented images
        path = self.samples[idx][0]
        image = self.loader(path)

        if self.transform is None:
            im, t = transforms.ToTensor()(image), target
            # print("im, t", im,t)
            return im, t
        else:
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            img_w = self.transform(image)
            if not self.use_strong_transform:
                # print("img_w, t", img_w, target[])
                return img_w, target
            else:
                return img_w, self.strong_transform(image), target

    def __len__(self):
        return len(self.targets)

