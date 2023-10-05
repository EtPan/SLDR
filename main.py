import os
import argparse
import model

from data import *
from model import *
from trainer import *
from functools import partial
from torchvision.transforms import Compose

model_names = sorted(name for name in model.__dict__
    if name.islower() and not name.startswith("__")
    and callable(model.__dict__[name]))

def options(parser):   
        # basic parameters
        parser.add_argument('--prefix', '-p', type=str, default='denoise',help='prefix')
        parser.add_argument('--arch', '-a', metavar='ARCH', required=True,choices=model_names,
                                                        help='model architecture: ' +' | '.join(model_names))
        parser.add_argument('--dataroot', '-d', type=str,default='/data/panet/destripe/D/data/ICVL64_31.db', help='data root')
        parser.add_argument('--loss', type=str, default='l1',help='which loss to choose.', 
                                                        choices=['l1', 'l2', 'smooth_l1','focalw'])
        parser.add_argument('--phase', type=str, default='train',choices=['train','ablation', 'valid','test'])
        
        # parameters for training phase
        parser.add_argument('--batchsize', '-b', type=int, default=16, help='training batch size.')         
        parser.add_argument('--lr', type=float, default=1e-3, help='learning rate. default=1e-3.')
        parser.add_argument('--wd', type=float, default=0, help='weight decay. default=0')
        parser.add_argument('--init', type=str, default='kn',help='which init scheme to choose.', 
                                                        choices=['kn', 'ku', 'xn', 'xu', 'edsr','xavier'])
        parser.add_argument('--no-cuda', action='store_true', help='disable cuda?')
        parser.add_argument('--no-log', action='store_true', help='disable logger?')
        parser.add_argument('--threads', type=int, default=8, help='number of threads for data loader to use')
        parser.add_argument('--seed', type=int, default=123, help='random seed to use. default=2022')
        
        parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
        parser.add_argument('--resumePath', '-rp', type=str, default=None, help='checkpoint to use.')

        parser.add_argument('--no-ropt', '-nro', action='store_true', help='not resume optimizer')          
        parser.add_argument('--chop', action='store_true', help='forward chop')                                      
        parser.add_argument('--clip', type=float, default=1e6)

        parser.add_argument('--gpu-ids', type=str, default='0', help='gpu ids')
        
        opt = parser.parse_args()
        return opt

def make_trainset(opt,train_transform,target_transform,length=False):
        common_transform = lambda x: x
        bsize = opt.batchsize       
        dataset = MyLMDBDataset(opt.dataroot, repeat=1)
        if length :
            dataset.length -= length
        dataset  = TransformDataset(dataset, common_transform)
        trainset = ImageTransformDataset(dataset, train_transform, target_transform)

        train_loader = DataLoader(trainset, batch_size=bsize, shuffle=True,
                                  num_workers=8, pin_memory= opt.no_cuda,worker_init_fn=worker_init_fn)
        return train_loader

def make_validset(opt,valid_names,valid_transform,basefolder):
        valid_datasets = [MatDataFromFolder(os.path.join(basefolder, name), size=5) for name in valid_names]
        valid_datasets = [TransformDataset(mat_dataset, valid_transform)
                          for mat_dataset in valid_datasets]
        valid_loaders  = [DataLoader(mat_dataset, batch_size=1, shuffle=False,worker_init_fn=worker_init_fn,
                                     num_workers=1, pin_memory=opt.no_cuda) for mat_dataset in valid_datasets]
        return valid_loaders

def datasetting(opt,engine,itransforms,valid_names,basefolder,HSI2Tensor,l=False):
        print('==> Preparing data..')
        target_transform = HSI2Tensor()
        train_transform_h = Compose([SequentialSelect(transforms=[itransforms[0],
                                                                  itransforms[1],
                                                                  ]),HSI2Tensor()])
        train_transform_m = Compose([SequentialSelect(transforms=[itransforms[-2],
                                                                  itransforms[-1],
                                                                  ]),HSI2Tensor()])
        train_loader_h = make_trainset(opt, train_transform_h,target_transform, length=l)
        train_loader_m = make_trainset(opt, train_transform_m,target_transform, length=l)
        train_loaders = [train_loader_h,train_loader_m]
        
        if not engine.get_net().use_2dconv:
            valid_transforms = Compose([LoadMatHSI(input_key='input', gt_key='gt',
                                        transform=lambda x:x[:, ...][None]),])
        else:
            valid_transforms = Compose([LoadMatHSI(input_key='input', gt_key='gt'),])
        valid_loaders = make_validset(opt,valid_names,valid_transforms,basefolder)  
        return train_loaders, valid_loaders

def train(opt,engine,train_loaders,valid_loaders,valid_names):
        base_lr = opt.lr
        epoch_per_save = 5
        adjust_learning_rate(engine.optimizer, base_lr)
        while engine.epoch < 50:#20:#50:
            np.random.seed()
            if engine.epoch == 10:
                adjust_learning_rate(engine.optimizer, base_lr*0.1)
            
            if engine.epoch == 15:
                adjust_learning_rate(engine.optimizer, base_lr)

            if engine.epoch == 30:
                adjust_learning_rate(engine.optimizer, base_lr*0.1)

            if engine.epoch == 40:
                adjust_learning_rate(engine.optimizer, base_lr*0.01)

            if engine.epoch < 15 :
                engine.train(train_loaders[0])
                engine.validate(valid_loaders[0], valid_names[0])
                engine.validate(valid_loaders[1], valid_names[1])
            else: 
                engine.train(train_loaders[1])
                engine.validate(valid_loaders[-2], valid_names[-2])
                engine.validate(valid_loaders[-1], valid_names[-1])

            display_learning_rate(engine.optimizer)
            print('Latest Result Saving...')
            model_latest_path = os.path.join(engine.basedir, engine.prefix, 'model_latest.pth')
            engine.save_checkpoint(model_out_path=model_latest_path)

            display_learning_rate(engine.optimizer)
            if engine.epoch % epoch_per_save == 0:
                engine.save_checkpoint()



def validate(engine,valid_names,basefolder,resfolder):
        cuda = not opt.no_cuda
        opt.no_log = True

        r_mean = []
        i_mean = []
    
        for i in range(len(valid_names)):
            datadir = os.path.join(basefolder, valid_names[i])
            resdir = os.path.join(resfolder, valid_names[i])
            if not os.path.exists(resdir):
                os.mkdir(resdir)
            mat_dataset = MatDataFromFolder(datadir, size=None)

            if not engine.get_net().use_2dconv:
                mat_transform = Compose([LoadMatHSI(input_key='input', gt_key='gt',transform=lambda x:x[:, ...][None]),])
            else:
                mat_transform = Compose([LoadMatHSI(input_key='input', gt_key='gt'),])           
            mat_dataset = TransformDataset(mat_dataset, mat_transform)
            mat_loader = DataLoader(mat_dataset,batch_size=1, shuffle=False,num_workers=1, pin_memory=cuda)    
            
            res_arr, input_arr = engine.test_syns(mat_loader, savedir=None, verbose=False)
            r_mean.append(res_arr.mean(axis=0))
            i_mean.append(input_arr.mean(axis=0))
        
        for i in range(len(r_mean)):
            print(valid_names[i],'\n',i_mean[i],'\n',r_mean[i]) 


def test(engine,testmat):
        cuda = not opt.no_cuda
        opt.no_log = True
        mat_dataset = MatDataFromFolder(testmat[0], size=None)
        mat_transform = Compose([LoadMatKey(key=testmat[1]),
                                 lambda x:x[:,:,:][None], 
                                 minmax_normalize,])
        if not engine.get_net().use_2dconv:
                mat_transform = Compose([LoadMatKey(key=testmat[1]),
                                 lambda x:x[:,:,:][None], 
                                 minmax_normalize,])
        else:
                mat_transform = Compose([LoadMatKey(key=testmat[1]),
                                 lambda x:x,minmax_normalize,])
        mat_dataset = TransformDataset(mat_dataset, mat_transform)
        mat_loader = DataLoader(mat_dataset,batch_size=1, shuffle=False,num_workers=1, pin_memory=cuda)    
        engine.test_real(mat_loader, savedir=testmat[-1])

if __name__ == '__main__':
        """Training settings"""
        parser = argparse.ArgumentParser()
        opt = options(parser)
        print(opt)
        engine = Engine3(opt)

        resfolder = "./result/"
        if not os.path.exists(resfolder):
            os.mkdir(resfolder)
        
        basefolder = '/data/panet/destripe/JORDER/data/'
        itransforms = [AddStripeNoiseH(),AddStripeNoiseB(),AddStripeNoiseL(),AddMixedNoiseH()]
        valid_names = ['icvl_stripe_H','icvl_stripe_B','icvl_stripe_L','icvl_stripe_M']

        HSI2Tensor = partial(HSI2Tensor, use_2dconv=engine.get_net().use_2dconv)

        if opt.phase == 'train':
            print('======================Train======================')
            train_loaders, valid_loaders = datasetting(opt,engine,itransforms,valid_names,basefolder,HSI2Tensor)
            train(opt,engine,train_loaders,valid_loaders,valid_names)
        
        elif opt.phase == 'valid':
            print('======================Valid======================')
            _, valid_loaders = datasetting(opt,engine,itransforms,valid_names,basefolder,HSI2Tensor)
            validate(engine,valid_names,basefolder,resfolder)
        else:
            print('===================Test real HSI===================')
            testmat=['/data/panet/Data/Noise-GF5/wh/','savemat','../D/result_real/wuhan/']
            test(engine,testmat)


