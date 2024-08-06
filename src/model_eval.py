"""
    File used to load trained models
"""
#%% Imports
import dataHandler, torch, classifiers
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from torchmetrics import ConfusionMatrix

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1) Obtain clock data from specified data folders. Set device agnostic code and get model.
clock_data_total_fail: list[dict] = dataHandler.logFileToDict(dataHandler.DATAFOLDER_FAIL)

# %% Loading saved models.
""" Load model for evaluation NOTE: Adjust model_num to select proper model """
model_num = "v1_3"
path_model = dataHandler.DATAFOLDER_FAIL + r'/Models/csac_ml_' + str(model_num) + r'.pt'
model_load = classifiers.PtClassifier_V2(
    parameters=len(clock_data_total_fail[0]['df'].columns[1:-1]),
    classes=1
)

model_load.load_state_dict(
    torch.load(path_model)
)

#%% Test Loaded model

sc_Standard = StandardScaler()
tm_cm = ConfusionMatrix(task="binary", num_classes=2).to(device)
model_load.to(device)

model_load.eval()
clock_model_predictions = []
clock_model_actuals = []
clock_model_cm = []
with torch.inference_mode():
    for clock in clock_data_total_fail:
        print(clock['sn'])
        actual = clock['df']['FAIL'].values
        clock_total_data = clock['df'].iloc[:, 1:-1].values

        clock_total_data_scaled = sc_Standard.fit_transform(clock_total_data)

        clock_total_data_scaled = torch.Tensor(clock_total_data_scaled).to(device)
        predicted = torch.round(torch.sigmoid(model_load(clock_total_data_scaled)))
        # cm = tm_cm(predicted, torch.tensor(actual).unsqueeze())
        # print(predicted.squeeze(), " Count True: ", predicted.count_nonzero(), "len: ", predicted.__len__())

        predicted = predicted.squeeze().cpu()
        clock_model_predictions.append(predicted)
        clock_model_actuals.append(actual)
        # clock_model_cm.append(cm.cpu().numpy())

#%% Plot results
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(22, 6))
f_figs = dataHandler.createDataFolder(dataHandler.DATAFOLDER_FAIL, "Figs")
plot_num = 0
for i, p in enumerate(clock_model_predictions):
    #secs = clock_data_total_fail[i]['df']['SECS']
    #fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(22, 6))
    graph_check = i % 3

    print("Graph Check: ", graph_check)
    ax[0][graph_check].plot(p.numpy(), color='r')
    ax[0][graph_check].set_xlabel('secs')
    ax[0][graph_check].set_title(clock_data_total_fail[i]['sn'] + f' Model {model_num} Predictions | Count: ' + str(np.sum(p.numpy())), loc='center')
    ax[0][graph_check].set_yticks([0, 1])

    ax[1][graph_check].plot(clock_model_actuals[i], color='b')
    ax[1][graph_check].set_xlabel('secs')
    ax[1][graph_check].set_title(' Actual | Count: ' + str(np.sum(clock_model_actuals[i])), loc='center')
    ax[1][graph_check].set_yticks([0, 1])

    plt.tight_layout()
    
    #plt.suptitle("", horizontalalignment='center')
    if graph_check == 2:
        
        plt.savefig(f_figs + r'/Validations/' + f"Validate_Predicts_{model_num}_" + str(plot_num) +'.png')
        plt.close()
        plt.clf()
        fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(22, 6))
        plot_num+=1



#%% Plotting the confusion matrix using Matplotlib
# fig, ax = plt.subplots(figsize=(8, 6))
# cax = ax.matshow(clock_model_cm[0], cmap='Blues')
# num_classes = 2
# # Add color bar
# fig.colorbar(cax)

# # Add labels and title
# classes = [f'Class {i}' for i in range(num_classes)]
# ax.set_xticks(np.arange(num_classes))
# ax.set_yticks(np.arange(num_classes))
# ax.set_xticklabels(classes)
# ax.set_yticklabels(classes)
# ax.set_xlabel('Predicted Label')
# ax.set_ylabel('True Label')
# ax.set_title('Confusion Matrix')

# # Annotate the matrix with text
# for i in range(num_classes):
#     for j in range(num_classes):
#         ax.text(j, i, clock_model_cm[0][i, j], ha='center', va='center', color='black')
# %%
