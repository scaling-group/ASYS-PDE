"""Fresh fixed-clock neural transient field; no searched remainder or fitted state."""
import math
import torch
TF=.020385594
TC=.020406
class Neural(torch.nn.Module):
    def __init__(self,seed):
        super().__init__()
        torch.manual_seed(seed)
        layers=[torch.nn.Linear(2,64),torch.nn.Tanh()]
        for _ in range(3):layers.extend([torch.nn.Linear(64,64),torch.nn.Tanh()])
        layers.append(torch.nn.Linear(64,1));self.net=torch.nn.Sequential(*layers)
        torch.nn.init.zeros_(self.net[-1].weight);torch.nn.init.zeros_(self.net[-1].bias)
    def forward(self,p):
        r,t=p[:,:1],p[:,1:2];z=(r/5).square();remaining=1-t/TC
        eta=-torch.log1p(-t/TC)/(-math.log1p(-TF/TC))
        return 5*torch.cos(math.pi*r/10)+(t/TF)*(1-z)*self.net(torch.cat([z,eta],1))/torch.sqrt(remaining)
