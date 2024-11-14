import time
import torch
import torch.nn                     as nn
import matplotlib.pyplot            as plt

from tqdm                           import tqdm
from torchmetrics                   import MeanSquaredError, MeanAbsoluteError

# --------------------------------
# Función para graficar la pérdida
# del entrenamiento en tiempo real
# --------------------------------
def plot_loss_curve(loss_values):
    plt.clf()  
    plt.plot(loss_values) 
    plt.xlabel('Iteraciones')
    plt.ylabel('Loss')
    plt.title('Curva Loss de entrenamiento')
    plt.draw()  
    plt.pause(0.01)

# ----------------------
# function to calculate 
# the time of each epoch
# ----------------------
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

# -------------------------
# function to train a model
# -------------------------
def train(model, iterator, optimizer, criterion, scheduler, epoch, device, loss_curve, plot_loss=False):
    model.train()
    total_loss = 0
    mse_metric = MeanSquaredError().to(device)
    mae_metric = MeanAbsoluteError().to(device)

    k=0
    for batch in tqdm(iterator, desc=f'Epoch: {epoch+1:02}', mininterval=1):
        inpX, inputT, inpW = batch
        inpX, inputT, inpW = inpX.to(device), inputT.to(device), inpW.to(device)

        optimizer.zero_grad()
        output, add_loss = model(inpX, inputT, inpW)
        loss = criterion(output, inpX)
        loss += add_loss

        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

        # Update the MSE and MAE metrics
        mse_metric.update(output, inpX)
        mae_metric.update(output, inpX)

        k+=1
        if plot_loss and k%20==0:
            loss_curve.append(loss.item())
            plot_loss_curve(loss_curve)

    # Calculate Mean Squared Error and Mean Absolute Error
    mse = mse_metric.compute()
    mae = mae_metric.compute()
    mse_metric.reset()
    mae_metric.reset()

    return total_loss / len(iterator), mse.item(), mae.item()


#-------------------------------------------------------
# function that contains the entire training cycle of a
# classification problem and returns the learning curves
# ------------------------------------------------------
def train_vae_model(model, 
                    criterion,
                    optimizer,
                    scheduler, 
                    train_iterator, 
                    epochs, 
                    device,
                    plot_loss = False, 
                    dir_model = 'saved_models/saved-model.pt'):
    
    model     = model.to(device)
    criterion = criterion.to(device)

    # Learning curves
    best_valid_loss   = float('inf')
    train_mse_curve   = []
    train_mae_curve   = []
    train_loss_curve  = []

    real_time_loss_curve = []
    if plot_loss:
        plt.ion()  
    for epoch in range(epochs):
        start_time = time.time()

        # Train cycles
        train_loss, train_mse, train_mae = train(model, train_iterator, optimizer, criterion, scheduler, 
                                                 epoch, device, real_time_loss_curve, plot_loss)

        # The best model is saved
        if train_loss < best_valid_loss:
            best_valid_loss = train_loss
            torch.save(model.state_dict(), dir_model)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        # Metric values ​​are saved
        train_mse_curve.append(train_mse)
        train_mae_curve.append(train_mae)
        train_loss_curve.append(train_loss)

        print(f'\tEpoch Time: {epoch_mins}m {epoch_secs}s | Train Loss: {train_loss:.3f} | Train MSE: {train_mse:.3f} | Train MAE: {train_mae:.3f}')

    if plot_loss:
        plt.ioff()  # Apaga el modo interactivo
        plt.show()
        
    # Dictionary with results
    results = {'train_mse_curve':   train_mse_curve, 
               'train_mae_curve':   train_mae_curve, 
               'train_loss_curve':  train_loss_curve
               }

    return results 