#! "C:\Users\CAEX\Desktop\Diego\Crusher Project\2ndrealdatajournalchile\mmodal\Scripts\python.exe"

import time
import torch
import torch.nn             as nn 
import torch.optim          as optim
import torch.nn.functional  as F
from model.config           import device

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def train(model, iterator, optimizer, criterion, device):
    model.train()
    total_loss  = 0
    total_acc   = 0

    for batch in iterator:
        input_ids       = batch['input_ids'].to(device)
        attention_mask  = batch['attention_mask'].to(device)
        labels          = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask).logits
        loss    = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        # Calculate accuracy
        predictions = torch.argmax(outputs, dim=1)
        total_acc += (predictions == labels).sum().item()

    return total_loss/len(iterator), total_acc/len(iterator.dataset)


def evaluate(model, iterator, criterion, device):
    model.eval()
    total_loss = 0
    total_acc = 0

    with torch.no_grad():
        for batch in iterator:
            input_ids       = batch['input_ids'].to(device)
            attention_mask  = batch['attention_mask'].to(device)
            labels          = batch['labels'].to(device)  
            
            outputs = model(input_ids, attention_mask).logits
            
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # Calculate accuracy
            predictions = torch.argmax(outputs, dim=1)
            total_acc += (predictions == labels).sum().item()

    return total_loss/len(iterator), total_acc/len(iterator.dataset)


def train_classification_model(model, 
                               optimizer, 
                               train_iterator, 
                               valid_iterator, 
                               test_iterator, 
                               epochs, 
                               device    = device, 
                               dir_model = 'saved_models/saved-model.pt'):
    
    model     = model.to(device)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    # Learning curves
    best_valid_loss   = float('inf')
    train_acc_curve   = []
    val_acc_curve     = []
    train_loss_curve  = []
    val_loss_curve    = []

    for epoch in range(epochs):
        start_time = time.time()

        # Train + validation cycles
        train_loss, train_acc = train(model, train_iterator, optimizer, criterion, device)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion, device)

        # The best model is saved
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), dir_model)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        # Metric values ​​are saved
        val_acc_curve.append(valid_acc)
        val_loss_curve.append(valid_loss)
        train_acc_curve.append(train_acc)
        train_loss_curve.append(train_loss)

        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

    # The best model is loaded
    model.load_state_dict(torch.load(dir_model))
    test_loss , test_acc = evaluate(model, test_iterator, criterion, device)
    print('------------------------------------------------')
    print('Performance on the test set:')
    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')

    return train_acc_curve, val_acc_curve, train_loss_curve, val_loss_curve