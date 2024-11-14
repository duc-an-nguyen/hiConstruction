import torch
import numpy as np
from torch.utils.data       import Dataset
from sklearn.preprocessing  import StandardScaler
from torch.utils.data       import DataLoader, TensorDataset


class RulDataLoader():
    def __init__(self, data, window_size, input_var, output_var, batch_size):
        self.data = data

        # Ventanas deslizantes para las variables de entrada y salida
        input_ventanas=[] #Ventanas de las variables W
        output_ventanas=[] #Ventanas de las Variables X
        for trajectory in self.data.trajectory.unique():
            try:
                #Se normaliza los datos
                scaler = StandardScaler()
                df_trajectory = self.data[self.data['trajectory']==trajectory]
                df_input = scaler.fit_transform(df_trajectory[input_var])
                df_output = scaler.fit_transform(df_trajectory[output_var])

                # Se crean las ventanas
                input_ventana = np.lib.stride_tricks.sliding_window_view(df_input, window_shape=(window_size,), axis=0)#.transpose(0,2,1)
                output_ventana = np.lib.stride_tricks.sliding_window_view(df_output, window_shape=(window_size,), axis=0)#.transpose(0,2,1)
                input_ventanas.extend(input_ventana)
                output_ventanas.extend(output_ventana)
            except:
                continue 

        self.input_ventanas = np.array(input_ventanas)
        self.output_ventanas = np.array(output_ventanas)

        self.df_sliding = self.data.groupby('trajectory').apply(lambda x: x.iloc[window_size-1:]).reset_index(drop=True)
        self.cycles_ventanas = self.df_sliding['cycles'].values.reshape(-1, 1) # Ventanas con los ciclos

        # Se convierte las ventanas a tensores
        self.output_ventanas_tensor = torch.tensor(self.output_ventanas, dtype=torch.float32)
        self.cycles_ventanas_tensor = torch.tensor(self.cycles_ventanas, dtype=torch.float32)
        self.input_ventanas_tensor = torch.tensor(self.input_ventanas, dtype=torch.float32)

        # Se crea el dataset y el dataloader
        self.dataset = TensorDataset(self.output_ventanas_tensor, self.cycles_ventanas_tensor, self.input_ventanas_tensor)
        self.train_loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        self.eval_loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False)