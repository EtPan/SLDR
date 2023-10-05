from .sldr import SLDR

def sldr():
    net = SLDR(in_channels=1, out_channels=1, num_features=32,)
    net.use_2dconv = False
    net.bandwise = False
    return net

from .jorder8 import JORDER_IMX, JORDER_MIX

def imx():
    net = JORDER_IMX(in_channels=1, out_channels=1, num_features=32,)
    net.use_2dconv = False
    net.bandwise = False
    return net

def mix():
    net = JORDER_MIX(in_channels=1, out_channels=1, num_features=32,)
    net.use_2dconv = False
    net.bandwise = False
    return net