import torch.nn as nn
from .grid_encoding import GridCellEncoding
from .spatial_encoder import SpatialEncoder
from .temporal_transformer import TemporalTransformer
from .spatial_decoder import SpatialDecoder
import torch

class GridCellWorldModel(nn.Module):
    def __init__(self,H=128,W=128,in_channels=4):
        super().__init__()
        self.grid=GridCellEncoding(H,W)
        grid_ch=72
        self.enc=SpatialEncoder(in_channels+grid_ch)
        self.dyn=TemporalTransformer()
        self.dec=SpatialDecoder(in_channels)

    def encode(self,S):
        B=S.shape[0]
        return self.enc(torch.cat([S,self.grid(B)],dim=1))

    def predict_step(self,S_t,events):
        z=self.encode(S_t)
        z_next=self.dyn(z,events)
        return self.dec(z_next), z_next

    def forward(self,S_t,events):
        S_next,_=self.predict_step(S_t,events)
        return S_next

