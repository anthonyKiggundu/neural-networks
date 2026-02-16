import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1),
            nn.GroupNorm(8, cout),
            nn.GELU(),
            nn.Conv2d(cout, cout, 3, padding=1),
            nn.GroupNorm(8, cout),
            nn.GELU()
        )
    def forward(self,x): return self.net(x)

class SpatialEncoder(nn.Module):
    def __init__(self, cin):
        super().__init__()
        self.d1=ConvBlock(cin,64); self.p1=nn.MaxPool2d(2)
        self.d2=ConvBlock(64,128); self.p2=nn.MaxPool2d(2)
        self.d3=ConvBlock(128,256); self.p3=nn.MaxPool2d(2)
        self.b=ConvBlock(256,256)

    def forward(self,x):
        x=self.p1(self.d1(x))
        x=self.p2(self.d2(x))
        x=self.p3(self.d3(x))
        return self.b(x)

