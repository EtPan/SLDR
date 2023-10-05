from copyreg import pickle
import random
import pickle
import numpy as np
from os.path import join
from itertools import product
import torch.utils.data as data

def rand_crop(img, cropx, cropy):
    _,y,x = img.shape
    x1 = random.randint(0, x - cropx)
    y1 = random.randint(0, y - cropy)
    return img[:, y1:y1+cropy, x1:x1+cropx]

def crop_center(img,cropx,cropy):
    _,y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[:, starty:starty+cropy,startx:startx+cropx]

def minmax_normalize(array):    
    amin = np.min(array)
    amax = np.max(array)
    return (array - amin) / (amax - amin)

def data_augmentation(image, mode=None):
    """
    Args:
        image: np.ndarray, shape: C X H X W
    """
    axes = (-2, -1)
    flipud = lambda x: x[:, ::-1, :] 
    
    if mode is None:
        mode = random.randint(0, 7)
    if mode == 0:
        # original
        image = image
    elif mode == 1:
        # flip up and down
        image = flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        image = np.rot90(image, axes=axes)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        image = np.rot90(image, axes=axes)
        image = flipud(image)
    elif mode == 4:
        # rotate 180 degree
        image = np.rot90(image, k=2, axes=axes)
    elif mode == 5:
        # rotate 180 degree and flip
        image = np.rot90(image, k=2, axes=axes)
        image = flipud(image)
    elif mode == 6:
        # rotate 270 degree
        image = np.rot90(image, k=3, axes=axes)
    elif mode == 7:
        # rotate 270 degree and flip
        image = np.rot90(image, k=3, axes=axes)
        image = flipud(image)

    if random.random() < 0.5:
        image = image[::-1, :, :] 
    
    return np.ascontiguousarray(image)

def Data2Volume(data, ksizes, strides):
    """
    Construct Volumes from Original High Dimensional (D) Data
    """
    dshape = data.shape
    PatNum = lambda l, k, s: (np.floor( (l - k) / s ) + 1)   
    TotalPatNum = 1
    for i in range(len(ksizes)):
        TotalPatNum = TotalPatNum * PatNum(dshape[i], ksizes[i], strides[i])
    V = np.zeros([int(TotalPatNum)]+ksizes) # create D+1 dimension volume

    args = [range(kz) for kz in ksizes]
    for s in product(*args):
        s1 = (slice(None),) + s
        s2 = tuple([slice(key, -ksizes[i]+key+1 or None, strides[i]) for i, key in enumerate(s)])
        V[s1] = np.reshape(data[s2],(-1,))
        
    return V

def _get_info_from_lmdb(db_path):
    meta_info = pickle.load(open(join(db_path,'meta_info.pkl'),'rb'))
    fnames = meta_info['fnames']
    sizes = meta_info['resolution']
    if len(sizes) == 1:
        sizes = sizes * len(fnames)
    return fnames, sizes

class MyLMDBDataset(data.Dataset):
    def __init__(self, db_path, repeat=1):
        import lmdb
        self.db_path = db_path
        self.env = lmdb.open(db_path, max_readers=1, readonly=True, lock=False,
                             readahead=False, meminit=False)
        self.fnames, self.sizes = _get_info_from_lmdb(db_path)
        
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries']
        self.repeat = repeat

    def __getitem__(self, index):
        C,H,W = [int(s) for s in self.sizes[index].split('_')]
        index = index % self.length
        
        env = self.env
        with env.begin(write=False) as txn:
            raw_img = txn.get('{:08}'.format(index).encode('ascii'))
        flat_x = np.fromstring(raw_img, dtype=np.float32)
        x = flat_x.reshape(C,H,W)
        return x

    def __len__(self):
        return self.length * self.repeat

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'