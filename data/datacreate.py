"""generate dataset"""
import os
import lmdb
import h5py
import pickle
import numpy as np
from scipy.ndimage import zoom
from os.path import join, exists
from matplotlib import pyplot as plt
from scipy.io import loadmat, savemat
from datautil import *
from datasyn import *

def prefunc(data):
    data = np.rot90(data, k=-1, axes=(1,2))
    data = crop_center(data, 512, 512)
    data = minmax_normalize(data)
    return data

def create_dataset_valid(datadir, fnames, newdir, matkey, func=None, transform=None, load=h5py.File):
    if not exists(newdir):
        os.mkdir(newdir)

    for i, fn in enumerate(fnames):
        print('generate data(%d/%d)' %(i+1, len(fnames)))
        filepath = join(datadir, fn)
        mat = load(filepath)
        data = func(mat[matkey][...])

        noisedata = transform(data)
        # if not exists(join(newdir, fn)):
        savemat(join(newdir, fn), {'gt':data.transpose((2,1,0)),'input':noisedata.transpose((2,1,0))})


def create_icvl_val(datadir,fns):
    fnames = fns[:]
    case = ['icvl_stripe_H','icvl_stripe_L','icvl_stripe_B','icvl_stripe_M']
    transform = [AddStripeNoiseH(),AddStripeNoiseL(),AddStripeNoiseB(),AddMixedNoiseH()]

    newpath = './data/'
    for i in range(len(case)):
        print(case[i])
        newdir = join(newpath,case[i])
        create_dataset_valid(datadir,fnames, newdir,'rad', func=prefunc,
                  transform= transform[i])

def preprocess(data,crop_sizes,scales,ksizes,strides,augment):
    new_data = []
    data = minmax_normalize(data)
    data = np.rot90(data, k=2, axes=(1,2)) # ICVL
    # data = minmax_normalize(data.transpose((2,0,1))) # for Remote Sensing
    if crop_sizes is not None:
        data = crop_center(data, crop_sizes[0], crop_sizes[1])        
    
    for i in range(len(scales)):
        if scales[i] != 1:
            temp = zoom(data, zoom=(1, scales[i], scales[i]))
        else:
            temp = data
        temp = Data2Volume(temp, ksizes=ksizes, strides=list(strides[i]))            
        new_data.append(temp)
    new_data = np.concatenate(new_data, axis=0)
    if augment:
            for i in range(new_data.shape[0]):
                new_data[i,...] = data_augmentation(new_data[i, ...])
            
    return new_data.astype(np.float32)


def create_lmdb_train(datadir, fns, name, matkey,
    crop_sizes, scales, ksizes, strides,
    load=h5py.File, augment=True, seed=2022):
    """
    Create Augmented Dataset
    """
    np.random.seed(seed)
    scales = list(scales)
    ksizes = list(ksizes)        
    assert len(scales) == len(strides)

    # calculate the shape of dataset
    data = load(datadir + fns[0])[matkey]
    data = preprocess(data,crop_sizes,scales,ksizes,strides,augment)
    map_size = data.nbytes * len(fns) * 1.2
    print('map size (GB):', map_size / 1024 / 1024 / 1024)
    
    if os.path.exists(name+'.db'):
        raise Exception('database already exist!')
    env = lmdb.open(name+'.db', map_size=map_size, writemap=True)
    with env.begin(write=True) as txn:
        # txn is a Transaction object
        resolutions = []
        k = 0
        for i, fn in enumerate(fns):
            try:
                X = load(datadir + fn)[matkey]
            except:
                print('loading', datadir+fn, 'fail')
                continue
            X = preprocess(X,crop_sizes,scales,ksizes,strides,augment)        
            N = X.shape[0]
            for j in range(N):
                C,H,W = X[j].shape
                data = X[j].tobytes()
                str_id = '{:08}'.format(k)
                k += 1
                id_bytes = str_id.encode('ascii')
                resolutions.append('{:d}_{:d}_{:d}'.format(C,H,W))
                txn.put(id_bytes,data)               
            print('load mat (%d/%d): %s' %(i,len(fns),fn))
    env.close()
    print('Finish writing lmdb.')
    
    #create meta information
    meta_info = {}
    assert len(fns*N) == len(resolutions) 
    if len(set(resolutions))<=1:
        meta_info['resolution'] = [resolutions[0]]
        meta_info['fnames'] = [x for x in range(len(resolutions))]
        print('All HSIs have the same resolution. Simplify the meta info.')
    else:
        meta_info['resolution'] = resolutions
        meta_info['fnames'] = [x for x in range(len(resolutions))]
        print('Not all HSIs have the same resolution. Save meta info for each HSI.')
    
    pickle.dump(meta_info, open(join(name+'.db','meta_info.pkl'),'wb'))
    print('Finish creating lmdb meta info.')

def create_icvl_train(datadir,fns):
    print('create icvl64_31...')
    fnames = fns[:120]
    newdir = './data/ICVL64_31'

    create_lmdb_train(datadir, fnames, newdir, 'rad',  # your own dataset address
        crop_sizes=(512, 512),
        scales=(1, 0.5, 0.25),        
        ksizes=(31, 64, 64),
        strides=[(31, 64, 64), (31, 32, 32), (31, 32, 32)],        
        load=h5py.File, augment=True,)


import matplotlib.pyplot as plt
def displayband(mat,nband,save_dir):
    img = mat[:,:,nband]
    imgc = np.asarray(255*img)
    fig = plt.figure()
    ax1= fig.add_subplot(1,1,1)
    plt.axis('off')
    ax1.imshow(imgc,cmap='gray')
    plt.savefig(save_dir,dpi=400,pad_inches=0,bbox_inches='tight')
    #plt.show()

if __name__ == '__main__':
    rootdir = '../Data/ICVL/ALL/'
    allfns = os.listdir(rootdir)

    valfns = loadmat('./data/valfns.mat')['fns']
    valfns = [x.strip() for x in valfns]
    trainfns = [x for x in allfns if all(y not in x for y in valfns)]
    
    create_icvl_train(rootdir,trainfns)
    create_icvl_val(rootdir,valfns)


        