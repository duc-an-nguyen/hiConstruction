import torch
import numpy as np
import torch.nn as nn
from RulConstraint import C_Constraint, F_constraint

class ModelHI(nn.Module):
    def __init__(self, dim_x, dim_w, out_channels, window_size=100, z_size=1, kernel_size=85, stride=2, alpha=1, constraint="C"):
        super(ModelHI, self).__init__()

        self.alpha = alpha
        self.stride = stride
        self.kernel_size = kernel_size
        self.padding = kernel_size//2
        self.constraint = constraint

        # Valores de canales de salida
        self.out_channels = [dim_x] + out_channels
        self.reversed_out_channels = [dim_w+z_size] + out_channels[::-1]

        # Valores de los largos de secuencias de salida
        self.seq_len = self.calc_signal_seq(L_in=window_size, num_layers=len(out_channels))

        # Capa de activación
        self.activation = nn.GELU()

        # Encoder
        self.encoder = nn.Sequential(*[nn.Conv1d(in_channels = self.out_channels[i//2], 
                                                 out_channels = self.out_channels[i//2+1],
                                                 kernel_size = kernel_size, 
                                                 stride = stride,
                                                 padding = self.padding) 
                                                 if i%2==0 else self.activation for i in range(len(self.out_channels)*2-2)],
                                    nn.Flatten(),
                                    nn.Linear(out_channels[-1]*self.seq_len[-1], z_size))
        
        # Función de pérdida
        if constraint=='C':
            self.constraint = C_Constraint(alpha)
        elif constraint=='F':
            self.constraint = F_constraint(alpha)
        else:
            self.constraint = None

        # Capa pre-decoder
        self.repeat_vector = nn.Linear(z_size, window_size) 

        # Decoder
        self.decoder = nn.Sequential(*[nn.ConvTranspose1d(in_channels = self.reversed_out_channels[i//2],
                                                          out_channels = self.reversed_out_channels[i//2+1],
                                                          kernel_size = kernel_size,
                                                          stride = 1,
                                                          padding = self.padding)
                                                          if i%2==0 else self.activation for i in range(len(self.reversed_out_channels)*2-2)])
        
        
        # Capa de salida
        self.out_fc = nn.Linear(out_channels[0], dim_x)

    def calc_signal_seq(self, L_in, num_layers):
        L_out = L_in
        signal_out = [L_out]
        for i in range(num_layers):
            L_out = np.floor((L_out + 2*self.padding - (self.kernel_size-1) - 1)/self.stride + 1)
            signal_out.append(int(L_out))
        return signal_out

    def forward(self, inpX, inputT, inpW):
        x = inpX
        batch_size, num_var, window_size = x.shape

        # Encoder
        z = self.encoder(x)
        z, c_loss = self.constraint(z, inputT)
        
        # Procesamiento de z
        z_repeated = self.repeat_vector(z)
        z_repeated = torch.cat((z_repeated.unsqueeze(dim=1), inpW), dim=1)

        # Decoder
        x = self.decoder(z_repeated)
        x = x.permute(0,2,1).contiguous().view(batch_size*window_size, -1)
        x = self.out_fc(x)
        x = x.view(batch_size, num_var, -1)
        return x, c_loss