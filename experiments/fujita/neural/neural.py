"""Supplied concentrating seed times neural exponential correction; no searched remainder."""
import math
import torch
from additive_neural import Neural as OriginalNeural,TF,TC
class Neural(OriginalNeural):
    def __init__(self,seed):
        super().__init__(seed)
        # Exact n=0 Polynomial initialization/parameterization.
        self.clock=torch.nn.Parameter(torch.tensor(math.log(math.expm1((TC-TF)/TC))))
        self.width=torch.nn.Parameter(torch.tensor(math.log(.01)))
    def forward(self,points):
        r,t=points[:,:1],points[:,1:];T=TF+TC*torch.nn.functional.softplus(self.clock);b=self.width.exp()
        seed=5*torch.cos(math.pi*r/10)*torch.sqrt((T+b*r*r)/(T-t+b*r*r))
        z=(r/5).square();eta=-torch.log1p(-t/TC)/(-math.log1p(-TF/TC))
        return seed*torch.exp((t/TF)*self.net(torch.cat([z,eta],1)))
