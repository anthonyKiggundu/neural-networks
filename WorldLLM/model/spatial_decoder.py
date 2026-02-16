import torch.nn as nn
from .spatial_encoder import ConvBlock

class SpatialDecoder(nn.Module):
    def __init__(self, cout):
        super().__init__()
        self.u1=nn.ConvTranspose2d(256,256,2,2); self.c1=ConvBlock(256,128)
        self.u2=nn.ConvTranspose2d(128,128,2,2); self.c2=ConvBlock(128,64)
        self.u3=nn.ConvTranspose2d(64,64,2,2); self.c3=ConvBlock(64,32)
        self.final=nn.Conv2d(32,cout,1)

    def forward(self,z):
        z=self.c1(self.u1(z))
        z=self.c2(self.u2(z))
        z=self.c3(self.u3(z))
        return self.final(z)

