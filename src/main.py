"""
    Main file used to train classifiers and view classifier results.
"""
#%% Imports 
""" Machine Learning """
import torch
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torchmetrics
import torchmetrics.classification

""" User created files to handle model creation and CSAC LOG data files. """
import classifiers, dataHandler, time


#%% 1) Obtain clock data from specified data folders. Set device agnostic code and get model.
""" Obtain clock data, set loss and optimizer functions """
clock_data_total_fail: list[dict] = dataHandler.logFileToDict(dataHandler.DATAFOLDER_FAIL)

device = "cuda" if torch.cuda.is_available() else "cpu"


"""
    Change below model number and class number to train either linear or non-linear model.
    v1 = linear model
    v2 = nonlinear model

    Model Number convention: v{1 or 2}_{version}
"""
model_num = "v1_3"
model = classifiers.PtClassifier_V2(
    parameters=len(clock_data_total_fail[0]['df'].columns[1:-1]),
    classes=1
    )
model.to(device)
optimizer = optim.Adam(model.parameters())
criterion = torch.nn.BCEWithLogitsLoss()
tm_binary_acc = torchmetrics.classification.BinaryAccuracy().to(device)
tm_cm = torchmetrics.ConfusionMatrix(task="binary", num_classes=2).to(device)

EPOCHS = 100
# BATCH_SIZE = 8
v_loss_train = []
v_loss_test  = []
v_acc_train  = []
v_acc_test   = []
c_epoch = list(range(1, EPOCHS + 1, 10))
cm_train = []
total_loss_train = []
total_loss_test = []
total_acc_train = []
total_acc_test = []

torch.manual_seed(42)
#%% 2) Split clock data into test and training sets
""" Split data to Train/Test and train model """
for count, clock in enumerate(clock_data_total_fail):
    X_train, X_test, y_train, y_test = train_test_split(
        clock['df'].iloc[:, 1:-1].values,
        clock['df'].iloc[:, -1].values,
        test_size=0.30,
        random_state=42
    )

    print("Clock data: ", clock['sn'])   
    print("Training size: X: ", len(X_train), ' y:', len(y_train))
    print("Testing size: X: ", len(X_test), ' y:', len(y_test))

    # 3) Scale both train/test datasets and fit training dataset.
    sc_Standard = StandardScaler()
    X_train_sc = sc_Standard.fit_transform(X_train)
    X_test_sc = sc_Standard.transform(X_test)

    # Transfer data to device.
    X_train_sc = torch.Tensor(X_train_sc).to(device)
    X_test_sc = torch.Tensor(X_test_sc).to(device)
    y_train = torch.Tensor(y_train).to(device)
    y_test = torch.Tensor(y_test).to(device)

    torch.manual_seed(42)
    v_loss_train = []
    v_loss_test  = []
    v_acc_train  = []
    v_acc_test   = []

    # Tracking Loss and Metric values

    for epoch in range(EPOCHS):

        # Training loop
        model.train()

        # Forward Pass
        out = model(X_train_sc)

        loss = criterion(out.squeeze(), y_train)

        p_train = torch.round(torch.sigmoid(out.squeeze()))

        # Evaluation Metrics
        acc = tm_binary_acc(p_train.squeeze(), y_train)
        cm = tm_cm(p_train, y_train)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()
    

    # Testing loop
        model.eval()
        with torch.inference_mode():
            predictions = model(X_test_sc)
            p_test = torch.round(torch.sigmoid(predictions))

            test_loss  = criterion(predictions.squeeze(), y_test)
            test_acc = tm_binary_acc(p_test.squeeze(), y_test)

            

            if epoch % 10 == 0:
                print(f'Epoch: {epoch} Train Loss: {loss:.4f} Test Loss: {test_loss:.4f}')
                v_loss_train.append(loss.item())
                v_loss_test.append(test_loss.item())
                v_acc_train.append(acc.item())
                v_acc_test.append(test_acc.item())


            # if count == (clock_data_total_fail.__len__() - 1):
            #     CLOCK_TO_CHECK = 0
            #     secs = clock_data_total_fail[CLOCK_TO_CHECK]['df']['SECS'].values
            #     actual = clock_data_total_fail[CLOCK_TO_CHECK]['df']['FAIL'].values
            #     clock_total_data = clock_data_total_fail[CLOCK_TO_CHECK]['df'].iloc[:, 1:-1].values

            #     clock_total_data_scaled = sc_Standard.transform(clock_total_data)

            #     clock_total_data_scaled = torch.Tensor(clock_total_data_scaled).to(device)
            #     predicted = torch.round(torch.sigmoid(model(clock_total_data_scaled)))

            #     print(clock['sn'])
            #     print(predicted.squeeze(), " Count True: ", predicted.count_nonzero(), "len: ", predicted.__len__())
            
            #     predicted = predicted.squeeze().cpu()
            #     #predicted = 1 - predicted
                

            #     fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(6,7))
            #     ax[0].plot(secs, predicted, marker= 'o', markersize='1', color='r', label='predicted')
            #     ax[0].set_xlabel('secs')
            #     ax[0].set_title(' Predictions ')
            #     ax[0].set_yticks([0, 1])

            #     ax[1].plot(secs, actual, color='b', label='actual')
            #     ax[1].set_xlabel('secs')
            #     ax[1].set_title(' Actual ')
            #     ax[1].set_yticks([0, 1])

            #     plt.tight_layout()
    #cm_train.append(tm_cm(p_train, y_train))
    total_loss_train.append(v_loss_train)
    total_loss_test.append(v_loss_test)
    total_acc_train.append(v_acc_train)
    total_acc_test.append(v_acc_test)
            

#%% Length of loss and accuracy
print(f'Loss: {total_loss_train.__len__()}')
print(f'Accuracy: {total_acc_train.__len__()}')
clock_num = 2

#%% Plotting Metrics
""" Plot Loss and Accuracy """

fig, axs = plt.subplots(4, 4, figsize=(11, 8))

for clock, ax in enumerate(axs.flat):
    ax.plot(c_epoch, total_loss_train[clock], label='Loss Train')
    ax.plot(c_epoch, total_loss_test[clock], label='Loss Test')
    ax.set_title(f'Clock {clock}')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')

plt.legend()
plt.suptitle(f"Training Loss Across Clocks: Model {model_num}")
plt.tight_layout()

fig, axs = plt.subplots(4, 4, figsize=(11, 8))
for clock, ax in enumerate(axs.flat):
    ax.plot(c_epoch, total_acc_train[clock], label='Accuracy Train')
    ax.plot(c_epoch, total_acc_test[clock], label='Accuracy Test')
    ax.set_title(f'Clock {clock}')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')

plt.legend()
plt.suptitle(f"Accuracy Across Clocks: Model {model_num}")
plt.tight_layout()

#print(cm_train)

#%% Save models
""" Save model to disk """
path_model = dataHandler.DATAFOLDER_FAIL + r'/Models/csac_ml_' + str(model_num) + r'.pt'
torch.save(model.state_dict(), f=path_model)
# %%
