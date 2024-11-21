"""
    Functions to handle model training.
"""
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np


'''
    Train on single instance of clock data.
'''
def train_step(
        dataloader:DataLoader,  
        model:torch.nn.Module, 
        optimizer:torch.optim.Optimizer, 
        loss_fn:torch.nn.Module,
        device:torch.device):
    
    # Set model to train
    model.train()

    # Initialize training metrics for loss, accuracy
    # Keep counting merics for clock number and data training
    train_loss, train_acc, label_in_row, clock_num = 0, 0, 0, 0
    train_loss_each_clock = []
    train_acc_each_clock = []

    # Loop through data loader data batches
    for batch, (X, y, cols, validation) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

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

        for row in X[0]: #X[0] represents entire clock data
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

        clock_num = clock_num + 1
        train_loss_each_clock.append(train_loss/len(X[0]))
        train_acc_each_clock.append(train_acc/len(X[0]))
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
                # Forward pass
                y_test_pred = model(row)
                
                pred = 0 if torch.tanh(y_test_pred) < 0 else 1
                print(f"y_pred: {pred} | Act: {y[0][label_in_row].item()} | On row number: {label_in_row}", end='\r')

                # Calculate  and accumulate loss and accuracy
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

'''
    Batch Training
'''

def train_step_batch(
        dataloader:DataLoader,  
        model:torch.nn.Module, 
        optimizer:torch.optim.Optimizer, 
        loss_fn:torch.nn.Module,
        device:torch.device,
        batch_size:int):
    
    # Initializing parameters and put model into training
    train_loss, train_acc, clock_num = 0, 0, 0
    train_loss_per_clock, train_acc_per_clock = {}, {}
    model.train()


    # Get data from the dataloader and send data to target device
    for b, (X, y, cols, validation) in enumerate(dataloader):
        print(f"Training Clock | {clock_num}")
        X, y = X.to(device), y.to(device)

        # Loop through single clock data with size of batch
        for i in range(0, len(X[0]), batch_size): # X[0] is clock data
            
            # Forward pass on batch data
            y_preds_batch = model(X[0][i:i+batch_size])

            '''
                Since we are using BCEWithLogitsLoss, it provides a Sigmoid clamp function for input x.
                Providing a batch of inputs will return an average loss accross the batch.
            '''
            # Calculate Loss
            loss = loss_fn(y_preds_batch, y[0][i:i+batch_size].float())
            train_loss += loss.item()

            # 6) Calculate Accuracy
            y_preds_batch_acc = torch.round(torch.sigmoid(y_preds_batch).squeeze()).to(torch.int64)
            train_acc = torch.sum(torch.eq(y_preds_batch_acc, y[0][i:i+batch_size].squeeze()))

            # 7) Ensure backpropagation without gradient tracking
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss_per_clock["Clock_"+str(clock_num)] = []
        train_loss_per_clock["Clock_"+str(clock_num)].append(train_loss) #number of batches

        train_acc_per_clock["Clock_"+str(clock_num)] = []
        train_acc_per_clock["Clock_"+str(clock_num)].append(train_acc/len(X[0]))
        #train_loss_per_clock.append(train_loss/batch_size)
        clock_num += 1
        train_loss, train_acc = 0, 0
        
    return train_loss_per_clock, train_acc_per_clock


def test_step_batch(
        dataloader:DataLoader,  
        model:torch.nn.Module, 
        loss_fn:torch.nn.Module,
        device:torch.device,
        batch_size:int):
    '''
        Test on a batch size of individual clock data. The batches correspond to a seconds worth of 
        parameter data. 
    '''
    # Set model to training and initialize variables
    model.eval()
    test_loss, test_acc, clock_num = 0, 0, 0
    test_loss_each_clock, test_acc_each_clock = [], []

    # Turn of grad and optimize testing
    with torch.inference_mode():
        for b, (X, y, cols, validation) in enumerate(dataloader):

            # Send data to target device
            X, y = X.to(device), y.to(device)
            print(f"Testing Clock: {clock_num} | {validation[0]}")

            # Loop through clock data and perform batch testing
            for i in range(0, len(X[0]), batch_size):
                y_preds_batch = model(X[0][i:i+batch_size])
                
                # Calculate Loss
                loss = loss_fn(y_preds_batch, y[0][i:i+batch_size].float())
                test_loss += loss.item()

                # Convert predicted batch into prediction values of 0 - PASS or 1 - FAIL
                # Calculate Accuracy
                y_preds_batch_acc = torch.round(torch.sigmoid(y_preds_batch).squeeze()).to(torch.int64)
                test_acc = torch.sum(torch.eq(y_preds_batch_acc, y[0][i:i+batch_size].squeeze()))

            test_loss_each_clock.append(test_loss)
            test_acc_each_clock.append(test_acc/len(X[0]))
            clock_num += 1
            test_loss, test_acc = 0, 0

    return test_loss_each_clock, test_acc_each_clock


def train_batch(
        model: torch.nn.Module, 
        train_dataloader: DataLoader, 
        test_dataloader: DataLoader, 
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        epochs: int,
        device: torch.device,
        batch_size:int):
    
    for epoch in tqdm(range(epochs)):
        print(f"Epoch: {epoch}")

        train_loss, train_acc = train_step_batch(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            batch_size=batch_size
        )
        
        test_loss, test_acc = test_step_batch(
            dataloader=test_dataloader,
            model=model,
            loss_fn=loss_fn,
            batch_size=batch_size,
            device=device
        )

    return train_loss, train_acc, test_loss, test_acc


def train_across_clock_dataset(
        model: torch.nn.Module, 
        train_dataloader: DataLoader, 
        test_dataloader: DataLoader, 
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        epochs: int,
        device: torch.device,
        skip_pass_dataset:bool = False):

        #Initialize params
        train_loss, train_acc = {}, {}
        test_loss, test_acc = {}, {}
        t_loss, acc, clock_num = 0, 0, 0
        t_list, a_list = [], []
    
        # Training
        print("Model Training")
        model.train()
        for batch, (clock, labels, _, validation) in enumerate(train_dataloader):
            if skip_pass_dataset:
                if validation[0] == 'pass':
                    continue
            
            for epoch in range(epochs):
                preds:torch.Tensor = model(clock[0].to(device))
                labels:torch.Tensor = labels.to(device)

                loss = loss_fn(preds, labels[0].to(torch.float))
                t_loss += loss.item()
                preds = torch.round(torch.sigmoid(preds)).to(torch.int64)
                #acc = torch.sum(torch.eq(preds, labels[0].squeeze()))
                acc = np.sum(np.equal(preds.squeeze().cpu().numpy(), labels[0].squeeze().cpu().numpy()))
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                t_list.append(loss.item()/len(clock[0]))
                a_list.append(acc/len(labels[0]))
                print(f'Clock {clock_num} | {validation[0]} | Loss: {loss.item()/len(clock[0]):.2f} | Acc: {acc/len(labels[0]):.2f}', end='\r')
            
            train_loss['clock_'+str(clock_num)] = t_list
            train_acc['clock_'+str(clock_num)] = a_list
            t_list, a_list = [], []
            t_loss, acc = 0, 0
            clock_num += 1

        # Testing
        print()
        print("Model Testing")

        t_loss, acc, clock_num = 0, 0, 0
        t_list, a_list = [], []
        model.eval()
        with torch.inference_mode():
            for batch, (clock, labels, _, validation) in enumerate(test_dataloader):
                for epoch in range(epochs):
                    preds = model(clock[0].to(device))
                    labels = labels.to(device)

                    loss = loss_fn(preds, labels[0].to(torch.float))

                    t_loss += loss.item()
                    preds = torch.round(torch.sigmoid(preds)).to(torch.int64)
                    acc = np.sum(np.equal(preds.squeeze().cpu().numpy(), labels[0].squeeze().cpu().numpy()))

                    t_list.append(t_loss)
                    a_list.append(acc/len(labels[0]))
                    print(f'Clock {clock_num} | {validation[0]} | Loss: {t_loss:.2f} | Acc: {acc/len(labels[0]):.2f}', end='\r')
                
                test_loss['clock_'+str(clock_num)] = t_list
                test_acc['clock_'+str(clock_num)] = a_list
                t_list, a_list = [], []
                t_loss, acc = 0, 0
                clock_num += 1

        return train_loss, train_acc, test_loss, test_acc