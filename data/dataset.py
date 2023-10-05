import os
import torch
import threading
import numpy as np
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset
from torchnet.dataset import TransformDataset

def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)

class LockedIterator(object):
    def __init__(self, it):
        self.lock = threading.Lock()
        self.it = it.__iter__()

    def __iter__(self): return self

    def __next__(self):
        self.lock.acquire()
        try:
            return next(self.it)
        finally:
            self.lock.release()

class SequentialSelect(object):
    def __pos(self, n):
        i = 0
        while True: 
            # print(i)
            yield i
            i = (i + 1) % n

    def __init__(self, transforms):
        self.transforms = transforms
        self.pos = LockedIterator(self.__pos(len(transforms)))

    def __call__(self, img):
        out = self.transforms[next(self.pos)](img)
        return out

class HSI2Tensor(object):
    """
    Transform a numpy array with shape (C, H, W)
    into torch 4D Tensor (1, C, H, W) or (C, H, W)
    """
    def __init__(self, use_2dconv=False):
        self.use_2dconv = use_2dconv

    def __call__(self, hsi):
        if self.use_2dconv:
            img = torch.from_numpy(hsi)
        else:
            img = torch.from_numpy(hsi[None])
        return img.float()

class ImageTransformDataset(Dataset):
    def __init__(self, dataset, transform, target_transform=None):
        super(ImageTransformDataset, self).__init__()

        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform
        self.length = len(self.dataset)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):        
        img = self.dataset[idx]
        target = img.copy()
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class LoadMatKey(object):
    def __init__(self, key):
        self.key = key
    
    def __call__(self, mat):
        item = mat[self.key][:].transpose((2,0,1))
        return item.astype(np.float32)

class LoadMatHSI(object):
    def __init__(self, input_key, gt_key, transform=None):
        self.gt_key = gt_key
        self.input_key = input_key
        self.transform = transform
    
    def __call__(self, mat):
        if self.transform:
            input = self.transform(mat[self.input_key][:].transpose((2,0,1)))
            gt = self.transform(mat[self.gt_key][:].transpose((2,0,1)))
        else:
            input = mat[self.input_key][:].transpose((2,0,1))
            gt = mat[self.gt_key][:].transpose((2,0,1))
        input = torch.from_numpy(input).float()
        gt = torch.from_numpy(gt).float()

        return input, gt

class MatDataFromFolder(Dataset):
    """Wrap mat data from folder"""
    def __init__(self, data_dir, load=loadmat, suffix='mat', fns=None, size=None):
        super(MatDataFromFolder, self).__init__()
        if fns is not None:
            self.filenames = [os.path.join(data_dir, fn) for fn in fns
                            ]
        else:
            self.filenames = [os.path.join(data_dir, fn) 
                              for fn in os.listdir(data_dir)
                              if fn.endswith(suffix)
                              ]
        self.load = load

        if size and size <= len(self.filenames):
            self.filenames = self.filenames[:size]

    def __getitem__(self, index):
        mat = self.load(self.filenames[index])
        return mat

    def __len__(self):
        return len(self.filenames)