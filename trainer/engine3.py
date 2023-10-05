from audioop import avg
import os
from os.path import join,basename,exists
from scipy.io import savemat
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

import model
from trainer.loss import TVLoss, WTVLoss
from .tools import *
from .metrics import *

class Engine3(object):
    def __init__(self, opt):
        self.prefix = opt.prefix
        self.opt = opt
        self.net = None
        self.optimizer = None
        self.criterion = None
        self.basedir = None
        self.iteration = None
        self.epoch = None
        self.best_psnr = None
        self.best_loss = None
        self.writer = None

        self.__setup()

    def __setup(self):
        self.basedir = join('checkpoints', self.opt.arch)
        if not os.path.exists(self.basedir):
            os.makedirs(self.basedir)

        self.best_psnr = 0
        self.best_loss = 1e6
        self.epoch = 0
        self.iteration = 0

        cuda = not self.opt.no_cuda
        self.device = 'cuda' if cuda else 'cpu'
        print('Cuda Acess: %d' % cuda)
        if cuda and not torch.cuda.is_available():
            raise Exception("No GPU found, please run without --cuda")

        torch.manual_seed(self.opt.seed)
        if cuda:
            torch.cuda.manual_seed(self.opt.seed)

        """Model"""
        print("=> creating model '{}'".format(self.opt.arch))
        self.net = model.__dict__[self.opt.arch]()
        init_params(self.net, init_type=self.opt.init) 
        
        if self.opt.loss == 'l2':
            self.criterion = nn.MSELoss()
        if self.opt.loss == 'l1':
            self.criterion = nn.L1Loss()
        if self.opt.loss == 'smooth_l1':
            self.criterion = nn.SmoothL1Loss()
        
        self.mask_loss = torch.nn.CrossEntropyLoss(reduce=False, size_average=True)
        self.tv_loss = TVLoss()
        self.wtv_loss = WTVLoss()
        print("Loss:",self.criterion)

        if cuda:
            self.net.to(self.device)
            self.criterion = self.criterion.to(self.device)

        """Logger Setup"""
        log = not self.opt.no_log
        if log:
            self.writer = get_summary_writer(os.path.join(self.basedir, 'logs'), self.opt.prefix)

        """Optimization Setup"""
        self.optimizer = optim.Adam(
            self.net.parameters(), lr=self.opt.lr, weight_decay=self.opt.wd, amsgrad=False)

        """Resume previous model"""
        if self.opt.resume:
            # Load checkpoint.
            self.load(self.opt.resumePath, not self.opt.no_ropt)
        else:
            print('==> Building model..')

    def forward(self, inputs):        
        if self.opt.chop:           
            output = self.forward_chop(inputs)
        else:
            output,mask,intensity,noise = self.net(inputs)
        
        return output,mask,intensity,noise

    def forward_chop(self, x, base=16):        
        #n, c, b, h, w = x.size()
        if self.net.use_2dconv:
            n, b, h, w = x.size()
        else:
            n, c, b, h, w = x.size()
        h_half, w_half = h // 2, w // 2
        shave_h = np.ceil(h_half / base) * base - h_half
        shave_w = np.ceil(w_half / base) * base - w_half

        shave_h = shave_h if shave_h >= 10 else shave_h + base
        shave_w = shave_w if shave_w >= 10 else shave_w + base

        h_size, w_size = int(h_half + shave_h), int(w_half + shave_w)        
        
        inputs = [ x[..., 0:h_size, 0:w_size],
                            x[..., 0:h_size, (w - w_size):w],
                            x[..., (h - h_size):h, 0:w_size],
                            x[..., (h - h_size):h, (w - w_size):w] ]

        outputs=[]
        for input_i in inputs:
          output_i,_,_ =self.net(input_i)
          outputs.append(output_i)

        output = torch.zeros_like(x)
        output_w = torch.zeros_like(x)
        
        output[..., 0:h_half, 0:w_half] += outputs[0][..., 0:h_half, 0:w_half]
        output_w[..., 0:h_half, 0:w_half] += 1
        output[..., 0:h_half, w_half:w] += outputs[1][..., 0:h_half, (w_size - w + w_half):w_size]
        output_w[..., 0:h_half, w_half:w] += 1
        output[..., h_half:h, 0:w_half] += outputs[2][..., (h_size - h + h_half):h_size, 0:w_half]
        output_w[..., h_half:h, 0:w_half] += 1
        output[..., h_half:h, w_half:w] += outputs[3][..., (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]
        output_w[..., h_half:h, w_half:w] += 1
        
        output /= output_w

        return output

    def __step(self, train, inputs, targets):        
        if train:
            self.optimizer.zero_grad()
        loss_data = 0
        total_norm = None
        
        outputs, mask, intensity, noise = self.net(inputs)  
        size_batch, size_b,size_w,size_h = inputs.size()[0], inputs.size()[-3],inputs.size()[-2],inputs.size()[-1]
        
        loss_outputs = self.criterion(outputs, targets)
        if train:
            w1,w2,w3 = 1, 1e-2, 1e-2
            per_pixel_detection_loss = self.mask_loss(mask,((targets-inputs)[:,0,:,:,:]>0.03).type(torch.cuda.LongTensor))
            loss_mask = per_pixel_detection_loss.sum()/(size_b*size_w*size_h*size_batch)
            loss_intensity = self.criterion(intensity, targets-inputs)
            loss_wtv = self.wtv_loss(outputs, mask)
            loss = loss_outputs + w1*loss_intensity + w2*loss_mask + w3*loss_wtv
            loss.backward()
        else:
            loss = loss_outputs
            loss_wtv = loss_outputs
            loss_intensity = loss_outputs
            loss_mask = loss_outputs
 
        loss_data += loss.item()
        
        if train:
            total_norm = nn.utils.clip_grad_norm_(self.net.parameters(), self.opt.clip)
            self.optimizer.step()
        if self.opt.phase == 'train':
            return outputs, loss_data, total_norm, loss_outputs, loss_intensity, loss_mask, loss_wtv
        elif self.opt.phase == "valid" :
            return outputs, mask, intensity, noise, loss_data
        else:
            return outputs, mask, intensity, noise, loss_data

    def load(self, resumePath=None, load_opt=True):
        model_best_path = join(self.basedir, self.prefix, 'model_latest.pth')
        if os.path.exists(model_best_path):
            best_model = torch.load(model_best_path)

        print('==> Resuming from checkpoint %s..' % resumePath)
        assert os.path.isdir('checkpoints'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(resumePath or model_best_path)
        #### comment when using memnet
        self.epoch = checkpoint['epoch'] 
        self.iteration = checkpoint['iteration']
        self.get_net().load_state_dict(checkpoint['net'])

    def train(self, train_loader):
        self.net.train()
        train_loss = 0
        pbar = tqdm(enumerate(train_loader),total = len(train_loader),leave=True,ncols=180)
        for batch_idx, (inputs, targets) in pbar:
            if not self.opt.no_cuda:
                inputs, targets = inputs.to(self.device), targets.to(self.device)            
            _, loss_data, total_norm, loss_x, loss_i, loss_m, loss_w = self.__step(True, inputs, targets)
            train_loss += loss_data
            avg_loss = train_loss / (batch_idx+1)

            if not self.opt.no_log:
                self.writer.add_scalar(join(self.prefix,  'train_loss'), loss_data, self.iteration)
                self.writer.add_scalar( join(self.prefix, 'train_avg_loss'), avg_loss, self.iteration)

                self.writer.add_scalar(join(self.prefix,  'train_lossx'), loss_x, self.iteration)                
                self.writer.add_scalar(join(self.prefix,  'train_lossm'), loss_m, self.iteration)
                self.writer.add_scalar(join(self.prefix,  'train_lossi'), loss_i, self.iteration)
                self.writer.add_scalar(join(self.prefix,  'train_lossw'), loss_w, self.iteration)

            self.iteration += 1
            pdict = {'AvgLoss': '{:.4e}'.format(avg_loss), 
                     'Loss'  : '{:.4e}'.format(loss_data), 
                     'LossX' : '{:.4e}'.format(loss_x), 
                     'LossI' : '{:.4e}'.format(loss_i),
                     'LossM' : '{:.4e}'.format(loss_m), 
                     'LossW' : '{:.4e}'.format(loss_w), 
                     'Norm'  : '{:.4e}'.format(total_norm)}
            pbar.set_description(f'Epoch [{self.epoch}]')
            pbar.set_postfix(pdict)

        self.epoch += 1
        if not self.opt.no_log:
            self.writer.add_scalar(join(self.prefix, 'train_loss_epoch'), avg_loss, self.epoch)

    def validate(self, valid_loader, name):
        self.net.eval()
        validate_loss = 0
        total_psnr,total_ssim,total_msam = 0,0,0
        print('[i] Eval dataset {}...'.format(name))
        pbar = tqdm(enumerate(valid_loader),total = len(valid_loader),leave=True,ncols=110)
        with torch.no_grad():
            for batch_idx, (inputs, targets) in pbar:
                if not self.opt.no_cuda:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)                

                outputs,loss_data, _,_,_,_,_= self.__step(False, inputs, targets)
                psnr    = np.mean(cal_bwpsnr(outputs, targets))
                ssim    = np.mean(cal_bwssim(outputs, targets))
                msam    = np.mean(cal_sam(outputs,targets)) 

                validate_loss += loss_data
                avg_loss = validate_loss / (batch_idx+1)

                total_psnr      += psnr
                avg_psnr          = total_psnr / (batch_idx+1)
                total_ssim     += ssim
                avg_ssim         = total_ssim / (batch_idx+1)
                total_msam += msam
                avg_msam      = total_msam / (batch_idx+1)

                pdict = {'Loss': '{:.4e}'.format(avg_loss), 
                         'PSNR': '{:.4f}'.format(avg_psnr),
                         'SSIM': '{:.4f}'.format(avg_ssim),
                         'MSAM': '{:.4f}'.format(avg_msam)}
                pbar.set_description(f'Epoch [{self.epoch-1}]')
                pbar.set_postfix(pdict)

        if not self.opt.no_log:
            self.writer.add_scalar(join(self.prefix, name, 'val_loss_epoch'), avg_loss, self.epoch)
            self.writer.add_scalar(join(self.prefix, name, 'val_psnr_epoch'), avg_psnr, self.epoch)

        return avg_psnr, avg_loss

    def save_checkpoint(self, model_out_path=None, **kwargs):
        if not model_out_path:
            model_out_path = join(self.basedir, self.prefix, "model_epoch_%d_%d.pth" % (
                                                             self.epoch, self.iteration))

        state = {'net': self.get_net().state_dict(),
                 'optimizer': self.optimizer.state_dict(),
                 'epoch': self.epoch,
                 'iteration': self.iteration, }    
        state.update(kwargs)

        if not os.path.isdir(join(self.basedir, self.prefix)):
            os.makedirs(join(self.basedir, self.prefix))

        torch.save(state, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

    def test_syns(self, test_loader, savedir=None, verbose=True):

        def torch2numpy(hsi):
            if self.net.use_2dconv:
                R_hsi = hsi.data[0].cpu().numpy().transpose((1,2,0))
            else:
                R_hsi = hsi.data[0].cpu().numpy()[0,...].transpose((1,2,0))
            return R_hsi    

        self.net.eval()
        test_loss = 0
        dataset = test_loader.dataset.dataset

        res_arr = np.zeros((len(test_loader), 3+1))
        input_arr = np.zeros((len(test_loader), 3+1))

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                start.record()              
                if not self.opt.no_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                outputs, _,_,_,loss_data= self.__step(False, inputs, targets)
                
                torch.cuda.synchronize()
                end.record()
                test_loss += loss_data


                res_arr[batch_idx, :3] = MSIQA(outputs, targets)
                input_arr[batch_idx, :3] = MSIQA(inputs, targets)
                res_arr[batch_idx, 3:] = start.elapsed_time(end)

                psnr = res_arr[batch_idx, 0]
                ssim = res_arr[batch_idx, 1]
                sam = res_arr[batch_idx,2]
                time =res_arr[batch_idx,3]
                if verbose:
                    print("================",batch_idx)
                    print("---noisy:", input_arr[batch_idx, 0], input_arr[batch_idx, 1],input_arr[batch_idx, 2])
                    print("denoised:", round(psnr,3), round(ssim,4),round(sam,5),round(time,2))

                if savedir:
                    filedir = join(savedir, basename(dataset.filenames[batch_idx]).split('.')[0])  
                    outpath = join(filedir, '{}.mat'.format(self.opt.arch))
                    if not exists(filedir):
                        os.mkdir(filedir)
                    savemat(outpath, {'R_hsi': torch2numpy(outputs)}) 
                 
        return res_arr, input_arr

    def test_real(self, test_loader, savedir=None):
        """Warning: this code is not compatible with bandwise flag"""
        self.net.eval()
        dataset = test_loader.dataset.dataset

        def torch2numpy(hsi):
            if self.net.use_2dconv:
                R_hsi = hsi.data[0].cpu().numpy().transpose((1,2,0))
            else:
                R_hsi = hsi.data[0].cpu().numpy()[0,...].transpose((1,2,0))

            R_hsi = np.rot90(R_hsi,-1)
            return R_hsi    

        with torch.no_grad():
            for batch_idx, inputs in enumerate(test_loader):

                if not self.opt.no_cuda:
                    inputs = inputs.cuda()           

                outputs,mask,intensity,noise = self.forward(inputs)
                outputs = torch.rot90(outputs, k=1, dims=(3, 4))
                if savedir:
                    filedir = join(savedir, basename(dataset.filenames[batch_idx]).split('.')[0])  
                    savepath = join(filedir, '{}.mat'.format(self.opt.arch))    
                    if not os.path.exists(filedir):
                        os.mkdir(filedir)
                    savemat(savepath, {'R_hsi': torch2numpy(outputs)})
                    print(savepath,'done')

        return outputs

    def get_net(self):
        if len(self.opt.gpu_ids) > 1:
            return self.net.module
        else:
            return self.net           
