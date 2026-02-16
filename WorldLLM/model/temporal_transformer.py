import torch.nn as nn

class TemporalTransformer(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        layer=nn.TransformerEncoderLayer(
            d_model=dim, nhead=8, batch_first=True
        )
        self.tr=nn.TransformerEncoder(layer,6)
        self.event_proj=nn.Linear(16,dim)

    def forward(self,z,events):
        B,C,H,W=z.shape
        tok=z.flatten(2).permute(0,2,1)
        tok=tok+self.event_proj(events).unsqueeze(1)
        tok=self.tr(tok)
        return tok.permute(0,2,1).reshape(B,C,H,W)

