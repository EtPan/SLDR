import numpy as np

class AddNoiseMixed(object):
    """add mixed noise to the given numpy array (B,H,W)
    Args:
        noise_bank: list of noise maker (e.g. AddNoiseImpulse)
        num_bands: list of number of band which is corrupted by each item in noise_bank"""
    def __init__(self, noise_bank, num_bands):
        assert len(noise_bank) == len(num_bands)
        self.noise_bank = noise_bank
        self.num_bands = num_bands

    def __call__(self, img):
        B, H, W = img.shape
        all_bands = np.random.permutation(range(B))
        pos = 0
        for noise_maker, num_band in zip(self.noise_bank, self.num_bands):
            if 0 < num_band <= 1:
                num_band = int(np.floor(num_band * B))
            bands = all_bands[pos:pos+num_band]
            pos += num_band
            img = noise_maker(img, bands)
        return img

class _AddNoiseNoniid(object):
    """add non-iid gaussian noise to the given numpy array (B,H,W)"""
    def __init__(self, sigmas):
        self.sigmas = np.array(sigmas) / 255.
    
    def __call__(self, img, bands):
        bwsigmas = self.sigmas[np.random.randint(0, len(self.sigmas), len(bands))]
        B, H, W = img.shape
        img = img + np.random.randn(*img.shape) * 3/255.
        for i, n in zip(bands,bwsigmas): 
            noise = np.random.randn(1,H,W)*n
            img[i,:,:] = img[i,:,:]+ noise
        return img 

class _AddVerticalStripeNoise(object):
    """add stripe noise to the given numpy array (B,H,W)"""
    def __init__(self, min_amount, max_amount, lam, var):
        assert max_amount > min_amount
        self.min_amount = min_amount
        self.max_amount = max_amount 
        self.lam = lam   
        self.var = var   
    
    def __call__(self, img, bands):
        B, H, W = img.shape
        stype = H
        num_stripe = np.random.randint(np.floor(self.min_amount*stype), 
                                                    np.floor(self.max_amount*stype), len(bands))
        for i, n in zip(bands, num_stripe):
            loc = np.random.permutation(range(stype))
            loc = loc[:n]
            stripe = np.random.uniform(0,1, size=(len(loc),))*self.lam-self.var            
            sstripe = np.reshape(stripe, (-1, 1))
            img[i, loc, :] -= sstripe
        return img

class _AddHorizontalStripeNoise(object):
    """add stripe noise to the given numpy array (B,H,W)"""
    def __init__(self, min_amount, max_amount, lam, var):
        assert max_amount > min_amount
        self.min_amount = min_amount
        self.max_amount = max_amount 
        self.lam = lam   
        self.var = var   
    
    def __call__(self, img, bands):
        B, H, W = img.shape
        stype = W
        num_stripe = np.random.randint(np.floor(self.min_amount*stype), 
                                       np.floor(self.max_amount*stype), len(bands))
        for i, n in zip(bands, num_stripe):
            loc = np.random.permutation(range(stype))
            loc = loc[:n]
            stripe = np.random.uniform(0,1, size=(len(loc),))*self.lam-self.var            
            img[i, :, loc] -= np.reshape(stripe, (-1, 1))
        return img

class _AddLengthStripeNoise(object):
    """add stripe noise to the given numpy array (B,H,W)"""
    def __init__(self, min_amount,max_amount,lam,var):
        assert max_amount > min_amount
        self.min_amount = min_amount
        self.max_amount = max_amount 
        self.lam = lam   
        self.var = var   
    
    def __call__(self, img, bands):
        B, H, W = img.shape
        stype = H
        num_stripe = np.random.randint(np.floor(self.min_amount*stype), 
                                       np.floor(self.max_amount*stype), 
                                       len(bands)) 
        for i, n in zip(bands, num_stripe):
            #import ipdb
            #ipdb.set_trace()

            loc = np.random.permutation(range(stype)) 
            loc = loc[:n]     

            for k in loc :
                length = np.random.randint(0,W,1) 
                begin = np.random.randint(0,W-length+1,1) 

                stripe = np.random.uniform(0,1, size=(int(length),))*self.lam-self.var            
                img[i, k, int(begin):(int(begin)+int(length))] -= stripe

        return img


class _AddBroadStripeNoise(object):
    """add stripe noise to the given numpy array (B,H,W)"""
    def __init__(self, min_amount,max_amount,lam,var):
        assert max_amount > min_amount
        self.min_amount = min_amount
        self.max_amount = max_amount 
        self.lam = lam   
        self.var = var   
    
    def __call__(self, img, bands):
        B, H, W = img.shape
        
        stype = H
        num_stripe = np.random.randint(np.floor(self.min_amount*stype), 
                                       np.floor(self.max_amount*stype), 
                                       len(bands))  
        for i, n in zip(bands, num_stripe):
            perio= 10 
            loc = np.random.permutation(range(0,(stype-perio),perio))
            loc = loc[:int(n//perio)] 
            for k in loc :
                stripe = np.random.uniform(0,1, size=(perio,))*self.lam-self.var
                img[i, k:(k+int(perio)), :] -= np.transpose(np.tile(stripe,(W,1)),(1,0))
        
        return img


class _AddPerioStripeNoise(object):
    """add stripe noise to the given numpy array (B,H,W)"""
    def __init__(self, lam,var):
        self.lam = lam   
        self.var = var   
    
    def __call__(self, img, bands):
        B, H, W = img.shape
        
        stype = H
        perio= 80
        num_stripe = int(W//perio) 
        
        stripe = np.random.uniform(0,1, size=(perio-30,))*self.lam-self.var
        for i in bands:            
            for k in range(num_stripe):
                img[i, k*int(perio)+16:(k+1)*int(perio)-30+16, :] -= np.transpose(np.tile(stripe,(W,1)),(1,0))
        return img

class AddMixedNoiseH(AddNoiseMixed):
    def __init__(self):
        self.noise_bank = [_AddNoiseNoniid([30,50,70]),
                           _AddVerticalStripeNoise(0.05, 0.25, 0.5, 0.25)]
        self.num_bands = [2/3,2/3]

class AddMixedNoiseW(AddNoiseMixed):
    def __init__(self):
        self.noise_bank = [_AddNoiseNoniid([10,30,50,70]),
                           _AddHorizontalStripeNoise(0.05, 0.35, 0.5, 0.25)]
        self.num_bands = [1/3,1/3]

class AddStripeNoiseH(AddNoiseMixed):
    def __init__(self):
        self.noise_bank = [_AddNoiseNoniid([0]),
                           _AddVerticalStripeNoise(0.05, 0.45, 0.7, 0.5)]
        self.num_bands = [0,2/3]

class AddStripeNoiseW(AddNoiseMixed):
    def __init__(self):
        self.noise_bank = [_AddNoiseNoniid([0]),
                           _AddHorizontalStripeNoise(0.05, 0.45, 0.7, 0.5)]
        self.num_bands = [0,2/3]


class AddStripeNoiseL(AddNoiseMixed):
    def __init__(self):
        self.noise_bank = [_AddNoiseNoniid([0]),
                           _AddLengthStripeNoise(0.05, 0.45, 0.7, 0.5)]
        self.num_bands = [0,2/3]

class AddStripeNoiseB(AddNoiseMixed):
    def __init__(self):
        self.noise_bank = [_AddNoiseNoniid([0]),
                           _AddBroadStripeNoise(0.02, 0.15, 0.7, 0.5)]
        self.num_bands = [0,2/3]

class AddStripeNoiseP(AddNoiseMixed):
    def __init__(self):
        self.noise_bank = [_AddNoiseNoniid([0]),
                           _AddPerioStripeNoise(0.7, 0.3)]
        self.num_bands = [0,2/3]


class AddNoiidNoise(AddNoiseMixed):
    def __init__(self):
        self.noise_bank = [_AddNoiseNoniid([0]),
                           _AddNoiseNoniid([50])]
        self.num_bands = [0,2/3]

class AddNoiidBlind(AddNoiseMixed):
    def __init__(self):
        self.noise_bank = [_AddNoiseNoniid([0]),
                           _AddNoiseNoniid([30,50,70])]
        self.num_bands = [0,2/3]