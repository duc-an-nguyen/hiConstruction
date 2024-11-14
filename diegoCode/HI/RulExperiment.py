import numpy                        as np
import pandas                       as pd
import torch
import torch.optim                  as optim
import matplotlib.pyplot            as plt
import plotly.express               as px
from torch.optim.lr_scheduler       import OneCycleLR

from RulDataloader        import RulDataLoader
from RulModel             import ModelHI
from trainRulFunctions    import train_vae_model
from config               import device

class ExperimentRul(object):
    def __init__(self, data, input_columns, output_columns, window_size, batch_size, 
                 criterion, epochs, lr, plot_loss, directory, model_params):
        self.data = data
        self.epochs = epochs
        self.directory = directory
        self.window_size = window_size
        self.plot_loss = plot_loss

        self.model = ModelHI(**model_params)
        self.rul_processing = RulDataLoader(data, window_size, input_columns, output_columns, batch_size)
        self.train_loader = self.rul_processing.train_loader

        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('Número de parámetros a entrenar:', num_params)

        self.criterion = criterion
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        self.scheduler = OneCycleLR(self.optimizer, max_lr=lr*10, steps_per_epoch=len(self.train_loader), epochs=self.epochs)

    def train(self):
        self.results = train_vae_model(model             = self.model, 
                                       criterion         = self.criterion,
                                       optimizer         = self.optimizer,
                                       scheduler         = self.scheduler,  
                                       train_iterator    = self.train_loader, 
                                       epochs            = self.epochs,
                                       device            = device,
                                       plot_loss         = self.plot_loss,
                                       dir_model         = self.directory)

    def extract_latent_space(self):
        iterator = self.rul_processing.eval_loader

        z = []
        self.model.eval()
        with torch.no_grad():
            for data in iterator:
                inpX, inputT, inpW = data
                inpX, inputT, inpW = inpX.to(device), inputT.to(device), inpW.to(device)
                latent_representations = self.model.encoder(inpX)
                z.extend(latent_representations.cpu().numpy())
        self.z = np.array(z)

    def plot_learning_curves(self):
        fig = plt.figure(figsize=(12, 4))
        epochs_range = range(1, self.epochs + 1)
        
        # Plot the loss curves
        plt.subplot(1, 3, 1)
        plt.plot(epochs_range, self.results['train_loss_curve'], label='Train Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss Curves')
        plt.legend()
        
        # Plot the mse curves
        plt.subplot(1, 3, 2)
        plt.plot(epochs_range, self.results['train_mse_curve'], label='Train MSE')
        plt.xlabel('Epochs')
        plt.ylabel('MSE')
        plt.title('MSE Curves')
        plt.legend()
        
        # Plot the mae curves
        plt.subplot(1, 3, 3)
        plt.plot(epochs_range, self.results['train_mae_curve'], label='Train MAE')
        plt.xlabel('Epochs')
        plt.ylabel('MAE')
        plt.title('MAE Curves')
        plt.legend()
        
        fig.tight_layout()
        plt.show()
    
    def plot_rul(self):
        df_sliding = self.rul_processing.df_sliding.copy()
        df_sliding['rul'] = self.z

        # Se crea el dataframe con el rul obtenido 
        df_rul = self.data.join(df_sliding[['tiempo', 'rul']].set_index('tiempo'), on='tiempo')
        df_rul['rul_avg'] = df_rul['rul'].rolling(window=6, min_periods=1).mean()
        df_rul['index'] = df_rul.index.values

        # Se crea la figura usando plotly con fondo blanco
        fig = px.line(df_rul, x='index', y='rul', color='trajectory')
        fig.update_layout(
                    template='plotly_white',
                    plot_bgcolor='rgba(255, 255, 255, 1)',
                    paper_bgcolor='rgba(255, 255, 255, 1)',
                    xaxis=dict(showgrid=False, zeroline=False, showline=True, linecolor='black'),
                    yaxis=dict(showgrid=False, zeroline=False, showline=True, linecolor='black'),
                    )
        fig.show()
