'''
    File use to test different scalars among clock data.
'''
#%% Get data.
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, PowerTransformer
import matplotlib.pylab as plt

import dataHandler
import numpy as np

def plotScaledData(logFolderPath:str):
    """
        Saves plots of various scaled data into each folder of TempCo data.
    """
    scalars = [StandardScaler(), MinMaxScaler(), MaxAbsScaler(), RobustScaler()]
    clock_data_Fail: list[dict] = dataHandler.logFileToDict(logFolderPath)
    f_figs = dataHandler.createDataFolder(logFolderPath, "FigsScaledData")
    
    for clock in clock_data_Fail:
        clock_params_list:list = clock['df'].columns[1:-1].to_list() #Remove SECS and FAIL
        
        for param in clock_params_list:
            
            param_col_values = np.reshape(clock['df'][param].values, (-1, 1))
            #param_col_values_std = np.std(param_col_values)

            fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(11, 8.5))

            for i, scalar in enumerate(scalars):
                scaled_values = scalar.fit_transform(param_col_values)
                axes[i].plot(scaled_values)
                axes[i].set_title(str(scalar))
            
            plt.suptitle(clock['sn'] + " " + param, fontsize=24)
            fig.tight_layout()

            plt.savefig(f_figs + '/' + clock['sn'] + '_' + param +'.png')
            plt.close()
        
        print("Plotted: ", clock['sn'])

if __name__ == "__main__":
    plotScaledData(dataHandler.DATAFOLDER_FAIL)