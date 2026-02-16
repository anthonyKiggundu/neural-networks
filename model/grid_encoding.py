import torch
import torch.nn as nn
import math

class GridCellEncoding(nn.Module):
    def __init__(self, H, W, scales=6, orientations=6):
        super().__init__()
        self.wavelengths = [2**i for i in range(2,2+scales)]

        y,x = torch.meshgrid(
            torch.linspace(0,1,H),
            torch.linspace(0,1,W),
            indexing="ij"
        )
        self.register_buffer("coords", torch.stack([x,y],dim=-1))

        self.orientations = orientations

    def forward(self, B):
        enc=[]
        for wl in self.wavelengths:
            k = 2*math.pi/wl
            for o in range(self.orientations):
                theta=o*math.pi/3
                d=torch.tensor([math.cos(theta),math.sin(theta)],device=self.coords.device)
                proj=(self.coords@d)*k
                enc += [torch.sin(proj), torch.cos(proj)]

        pos=torch.stack(enc).unsqueeze(0).repeat(B,1,1,1)
        return pos

