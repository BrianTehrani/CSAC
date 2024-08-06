"""
    PyTorch implementations of ML Classifiers for CSAC validation. 
"""
import torch
from torch import nn

class PtClassifier_V1(nn.Module):
    '''
        Linear model used to predict Fail states in filterd TempCo data.
    '''
    def __init__(self, parameters, classes) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features=parameters, out_features=32),
            nn.Linear(in_features=32, out_features=8),
            nn.Linear(in_features=8, out_features=4),
            nn.Linear(in_features=4, out_features=classes)
        )
    
    def forward(self, x:torch.Tensor):
        return self.layers(x)
    
    def save(state_dict, folderPath:str, model_num:int):
        torch.save(
            obj=state_dict,
            f=folderPath + r'/Models/csac_ml_' + str(model_num) + r'.pt'
        )
        print("Model saved to: " + folderPath + r'/Models/csac_ml_' + str(model_num) + r'.pt')

class PtClassifier_V2(nn.Module):
    '''
        Non-Linear model used to predict Fail states in filterd TempCo data.
    '''
    def __init__(self, parameters, classes) -> None:
        super().__init__()
        self.layers = nn.Sequential(
        nn.Linear(in_features=parameters, out_features=32),
        nn.ReLU(),
        nn.Linear(in_features=32, out_features=8),
        nn.ReLU(),
        nn.Linear(in_features=8, out_features=4),
        nn.ReLU(),
        nn.Linear(in_features=4, out_features=classes),
        )
    
    def forward(self, x:torch.Tensor):
        return self.layers(x)
    
    def save(state_dict, folderPath:str, model_num:int):
        torch.save(
            obj=state_dict,
            f=folderPath + r'/Models/csac_ml_' + str(model_num) + r'.pt'
        )
        print("Model saved to: " + folderPath + r'/Models/csac_ml_' + str(model_num) + r'.pt')