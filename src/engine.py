"""
    Functions to handle model training.
"""
import tqdm, torch
from torch.utils.data import DataLoader
def train_step(
        dataloader:DataLoader,  
        model:torch.nn.Module, 
        optimizer:torch.optim.Optimizer, 
        loss_fn:torch.nn.Module,
        device:torch.device):
    
    model.train()
    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    # Loop through data loader data batches
    for batch, (X, y, cols, validation) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        """
            Test this training step.
            Each row in clock data needs to be used as input to model.
        """

        for i, data in enumerate(X):
            param, y = data[0][i], y[0][i]

            # 1. Forward pass
            y_pred = model(param)

            # 2. Calculate  and accumulate loss
            loss = loss_fn(y_pred, y)
            train_loss += loss.item() 

            # 3. Optimizer zero grad
            optimizer.zero_grad()

            # 4. Loss backward
            loss.backward()

            # 5. Optimizer step
            optimizer.step()

            # Calculate and accumulate accuracy metric across all batches
            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc


