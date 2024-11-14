import torch
import torch.nn as nn

class C_Constraint(nn.Module):
    def __init__(self, alpha):
        super(C_Constraint, self).__init__()
        self.alpha = alpha

    def forward(self, x, t):
        x_mu = torch.mean(x)
        t_mu = torch.mean(t)
        cov = torch.sum((x - x_mu) * (t - t_mu))
        std = torch.sqrt(torch.sum((x - x_mu)**2)) * torch.sqrt(torch.sum((t - t_mu)**2))
        loss = cov / std
        loss = torch.mean(loss)

        return x, self.alpha*loss
    
class F_constraint(nn.Module):
    def __init__(self, alpha):
        super(F_constraint, self).__init__()
        self.alpha = alpha
        
        # Constantes
        self.A = 3.24689126e-04
        self.B = 6.75139693e-01
        self.C = 1.02044605e+00
        self.BETA = 3.5
        self.P = 0.5

    def forward(self, x, t):
        HI = self.C - ((t * (-torch.log(torch.tensor(1.0) - self.P))**(-1/self.BETA)) * self.A)**self.B
        HI = torch.where(HI < 0, torch.tensor(0.0, dtype=HI.dtype, device=HI.device), HI)
        loss = torch.mean(torch.abs(x - HI))
        
        return x, self.alpha * loss