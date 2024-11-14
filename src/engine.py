"""
    Functions to handle model training.
"""
import torch, sys
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

def train_step(
        dataloader:DataLoader,  
        model:torch.nn.Module, 
        optimizer:torch.optim.Optimizer, 
        loss_fn:torch.nn.Module,
        device:torch.device):
    
    model.train()
    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0
    train_loss_each_clock = []
    train_acc_each_clock = []
    clock_num = 0
    label_in_row = 0

    # Loop through data loader data batches
    for batch, (x, y, cols, validation) in enumerate(dataloader):
        # Send data to target device
        x, y = x.to(device), y.to(device)

        """
            Each row in clock data needs to be used as input to model.

            X: Number of clocks in a particular set of data
            y: Pass/Fail labels for every clock in X
            clock: parameter data of a single clock in X

            To Train:
                1) Pick a clock in X
                2) Loop though each row in clock and match row with corresponding label in y
                3) Accumulate loss and accuracy metrics per clock 
        """

        for clock in x:
            print(f"Training Clock: {clock_num} | {validation}")

            for row in clock:
                # 1. Forward pass
                y_pred = model(row)
                pred = 0 if torch.tanh(y_pred) < 0 else 1
                # print(f"y_pred: {pred} | Act: {y[0][label_in_row].item()} | On row number: {label_in_row}", end='\r')

                # 2. Calculate  and accumulate loss
                loss = loss_fn(y_pred, y[0][label_in_row].float())
                train_loss += loss.item()
                if pred ==  y[0][label_in_row].item():
                    train_acc = train_acc + 1
                label_in_row = label_in_row + 1

                # 3. Optimizer zero grad
                optimizer.zero_grad()

                # 4. Loss backward
                loss.backward()

                # 5. Optimizer step
                optimizer.step()

                # Calculate and accumulate accuracy metric across all batches
                # y_pred_class = torch.argmax(torch.softmax(y_pred, dim=0), dim=1)
                # train_acc += (y_pred_class == y).sum().item()/len(y_pred)
            clock_num = clock_num + 1
            train_loss_each_clock.append(train_loss/len(clock))
            train_acc_each_clock.append(train_acc/len(clock))
            train_loss, train_acc, label_in_row = 0, 0, 0
            print()

    # Adjust metrics to get average loss and accuracy per batch 
    return train_loss_each_clock, train_acc_each_clock

def test_step(
        dataloader:DataLoader,  
        model:torch.nn.Module, 
        loss_fn:torch.nn.Module,
        device:torch.device):

    model.eval()
    test_loss, test_acc, label_in_row, clock_num = 0, 0, 0, 0
    test_loss_each_clock, test_acc_each_clock = [], []

    with torch.inference_mode():
        for batch, (x, y, cols, validation) in enumerate(dataloader):
            # Send data to target device
            x, y = x.to(device), y.to(device)

            print(f"Testing Clock: {clock_num} | {validation[0]}")
            for row in x[0]: #x[0] == clock loop
                # 1. Forward pass
                y_test_pred = model(row)
                
                pred = 0 if torch.tanh(y_test_pred) < 0 else 1
                print(f"y_pred: {pred} | Act: {y[0][label_in_row].item()} | On row number: {label_in_row}", end='\r')

                # 2. Calculate  and accumulate loss and accuracy
                loss = loss_fn(y_test_pred, y[0][label_in_row].float())
                test_loss += loss.item()

                if pred ==  y[0][label_in_row].item():
                    test_acc = test_acc + 1
                label_in_row = label_in_row + 1

            test_loss_each_clock.append(test_loss/len(x[0]))
            test_acc_each_clock.append(test_acc/len(x[0]))
            clock_num = clock_num + 1
            test_loss, test_acc, label_in_row = 0, 0, 0
            print()
            
    return test_loss_each_clock, test_acc_each_clock

def train(model: torch.nn.Module, 
          train_dataloader: DataLoader, 
          test_dataloader: DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device):
    
    # Create empty results dictionary
    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        print(f'Epoch {epoch+1}')
        train_loss, train_acc = train_step(model=model,
                                            dataloader=train_dataloader,
                                            loss_fn=loss_fn,
                                            optimizer=optimizer,
                                            device=device)
        test_loss, test_acc = test_step(model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device)

        # Print out what's happening
        
        # print(
        #     f"Epoch: {epoch+1} | "
        #     f"train_loss: {train_loss:.4f} | "
        #     f"train_acc: {train_acc:.4f} | "
        #     f"test_loss: {test_loss:.4f} | "
        #     f"test_acc: {test_acc:.4f}"
        # )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    # Return the filled results at the end of the epochs
    return results