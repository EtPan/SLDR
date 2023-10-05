
import torch
import torch.nn as nn
import model.basicblocks as B

class SLDR(nn.Module):
    def __init__(self, in_channels, out_channels, num_features,bias=True):
        super(SLDR, self).__init__()

        self.num_features = num_features
        
        self.encoder = nn.Sequential(
                          nn.Conv3d(in_channels, self.num_features, 
                                       3,1,1, bias=bias),
                          B.ResidualDenseBlock_5C(self.num_features, self.num_features//2))

        self.estimator_mask      = B.StripeAttributeAwareBlock(self.num_features, 2*in_channels)
        self.estimator_intensity = B.StripeAttributeAwareBlock(self.num_features, 1*in_channels)

        self.mask_w      = B.SandwichConv(2*in_channels, self.num_features) 
        self.intensity_w = B.SandwichConv(1*in_channels, self.num_features)
        
        self.mask_b      =  B.SandwichConv(2*in_channels, self.num_features)
        self.intensity_b =  B.SandwichConv(1*in_channels, self.num_features)
            
            
        self.decoder = nn.Sequential(
                          nn.Conv3d(self.num_features,   self.num_features//2, 
                                       3,1,1, bias=bias),
                          nn.Conv3d(self.num_features//2, self.num_features//4,
                                       3,1,1, bias=bias),
                          nn.Conv3d(self.num_features//4, 1*in_channels,
                                       1,1,0, bias=bias),)

    def forward(self, x):
        x_f = self.encoder(x)
        
        mask = self.estimator_mask(x_f)
        
        intensity = self.estimator_intensity(torch.mul(x_f,torch.argmax(mask,1).unsqueeze(1)))

        xf1 = self.mask_w(mask)*self.intensity_w(intensity)*x_f
        xf2 = self.mask_b(mask)+self.intensity_b(intensity)+x_f

        output = self.decoder(xf1+xf2)

        return x-output,mask,intensity,output

if __name__ == '__main__':
    from torchsummary import summary
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = JORDER_MIX(in_channels=1, out_channels=1, num_features=32,).to(device)
    summary(net, (1, 31, 64, 64))
