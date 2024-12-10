'''
    File used to handle Tempco LOG files.
'''
#%%
###
# Common Imports
###
import os, torch 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

DATAFOLDER_FAIL = r"D:\Data\Tempco\fail"
# DATAFOLDER_PASS = r"D:\Data\Tempco\pass"
# DATAFOLDER_FAIL_PASSEDLATER = r"D:\Data\Tempco\fail_passedLater"

#List of paramaters to drop from tempco data
#NOTE: Trial and error. Can adjust to data model needs
L_DROP_PARAMS = [
                    ' PDS1', ' DCI', ' VFALIM', ' FTUNE', ' PM_CNTS', ' RFLW',
                    ' PM_NSEC', ' VQPA', ' VQP', ' TEMPCO', ' HTUNE', ' DATE',
                    'PDS1', 'DCI', 'VFALIM', 'FTUNE', 'PM_CNTS', 'RFLW',
                    'PM_NSEC', 'VQPA', 'VQP', 'TEMPCO', 'HTUNE', 'DATE', 'SECS', ' SECS'
                ]

def createDataFolder(dataFolderPath:str, dataFolder:str) -> str:
    '''
        Checks to see if folder exists in path.
        If not, it will create folder.

        Args:
            dataFolderPath: Path where general data is stored.
            dataFolder: Name of folder to create.
    '''
    folderPath = os.path.join(dataFolderPath, dataFolder)
    if not os.path.exists(folderPath):
        os.mkdir(folderPath)

    return folderPath

def parseLogDataFrame(df:pd.DataFrame) -> pd.DataFrame:
    """
        Assort LOG data for model input.

        Returns: Parsed log data
    """
    l_df_cols = df.columns.to_list()
    for param in l_df_cols:
        if param in L_DROP_PARAMS:
            df.drop(param, axis=1, inplace=True)
        
        if param.find(' ') == 0: #Labeling issue with data titles, fixing ' ' in params
            df = df.rename(columns={param: param[1:]})
    
    return df

"""

    PLOTTING FUNCTIONS
    Uses matplotlib to save plotted graphs to specified folders in PATH DATAFOLDER.

"""
def plotLogData(logFolderPath:str) -> None:
    """
        Plot data columns of LOG files as PNG.
    """
    if not os.path.exists(logFolderPath):
        print("Data folder for LOG files not found.")
        return
    
    # Read LOG files in dir
    with os.scandir(logFolderPath) as logFiles:
        for logFile in logFiles:
            if logFile.is_file():
                #clock_freq = logFile.name[:logFile.name.find("-")]
                #if clock_freq == "SN10": #Focus on 10MHz clocks
                df = parseLogDataFrame(pd.read_csv(logFile.path, sep=','))
                
                secs = df[df.columns[0]]
                fig, axes = plt.subplots(3, 6, figsize=(45,18))
                axes = axes.flatten()
                
                for i, param in enumerate(df.columns[1:]):
                    axes[i].plot(secs, df[param])
                    axes[i].set_xlabel('secs')
                    axes[i].set_ylabel(param)

                plt.suptitle(logFile.name)
                plt.subplots_adjust(wspace=0.25, hspace=0.25)
                print("Plotting: ", logFile.name)

                f_figs = createDataFolder(logFolderPath + r"/Figs", "Plots")
                plt.savefig(f_figs + '/' + logFile.name[:-4] +'.png')
                plt.close()

def boxPlotLogData(logFolderPath:str) -> None:
    """
        Plot data columns of LOG files as PNG.
    """
    if not os.path.exists(logFolderPath):
        print("Data folder for LOG files not found.")
        return
    
    # Read LOG files in dir
    with os.scandir(logFolderPath) as logFiles:
        for logFile in logFiles:
            if logFile.is_file():
                clock_freq = logFile.name[:logFile.name.find("-")]
                if clock_freq == "SN10": #Focus on 10MHz clocks
                    
                    df = parseLogDataFrame(pd.read_csv(logFile.path, sep=','))
                    fig, axes = plt.subplots(3, 6, figsize=(24,18))
                    axes = axes.flatten()
                    
                    for i, param in enumerate(df.columns[1:]):
                        axes[i].boxplot(df[param].to_list())
                        axes[i].set_xlabel(param, fontsize=18)

                    plt.subplots_adjust(wspace=0.25, hspace=0.25)
                    plt.suptitle("BoxPlot_" + logFile.name, fontsize=24)
                    print("Plotting: ", logFile.name)

                    f_figs = createDataFolder(logFolderPath + r"/Figs", "Boxplots")
                    plt.savefig(f_figs + '/' + "BoxPlot_" + logFile.name[:-4] +'.png')
                    plt.close()

    print("Saved log files to: ", f_figs)

"""

    TYPING FUNCTIONS
    Converts LOG data to formats compatable with used Python modules.

"""

def logFileToList(logFolderPath:str) -> list[pd.DataFrame]:
    """
        Acquire clock data from a log file and append it to a list.
        Parameters are parsed based off list in 'parseLogDataFrame()'.

        Args:
            logFolderPath (str) - folder path which contains log files to convert to pd.DataFrame
    
        Return:
            List of appended dataframes containing filtered clock data.
    """

    if not os.path.exists(logFolderPath):
        print("Data folder for LOG files not found.")
        return
    
    clock_data_total: list[pd.DataFrame] = []
    with os.scandir(logFolderPath) as logFiles:
        for logFile in logFiles:
            if logFile.is_file():
                clock_freq = logFile.name[:logFile.name.find("-")]

                if clock_freq == "SN10": #Focus on 10MHz clocks
                    df_data_parsed = parseLogDataFrame(pd.read_csv(logFile.path, sep=','))
                    clock_data_total.append(df_data_parsed)
    
    return clock_data_total

def logToExcel(logFolderPath:str) -> None:
    '''
        Convert log files to CSV and store them in Excel folder.
    '''

    # Check to see if CSV folder exists in path to store converted LOG files
    excel_folder = createDataFolder(DATAFOLDER_FAIL, 'Excel')

    num_logFiles = len(os.listdir(logFolderPath))

    with os.scandir(logFolderPath) as logFiles:
        for logFile in logFiles:
            if logFile.is_file():
                df = pd.read_csv(logFile.path, sep=',')

                #removing .log from filepath and adding .csv
                df.to_excel(
                    os.path.join(excel_folder, logFile.name[:-4] + r".xlsx"), 
                    index=False
                )
                num_logFiles -= 1
                print("Number of LOG files to convert to Excel: ", num_logFiles)

def logFileToDict(logFolderPath:str) -> list[dict]:
    """
        Convert a log file to a python dictionary which contains parsed clock data with 
        additional clock information.

        Additional information in dictonary includes (keys):
            - sn   : Serial Number
            - freq : Frequency
            - df   : Pandas dataframe of parsed clock data
    """

    if not os.path.exists(logFolderPath):
        print("Data folder for LOG files not found.")
        return
    
    clock_data_total: list[dict] = []
    with os.scandir(logFolderPath) as logFiles:
        for logFile in logFiles:
            if logFile.is_file():
                
                clock_data = {
                    "sn" : "",
                    "freq": "",
                    "df": pd.DataFrame()
                }
                clock_sn:str   = logFile.name[logFile.name.find("-")+1:logFile.name.find("_")]
                clock_freq:str = logFile.name[:logFile.name.find("-")]

                # if clock_freq == "SN10": #Focus on 10MHz clocks
                clock_data['sn'] = clock_sn
                clock_data['freq'] = clock_freq
                clock_data['df'] = parseLogDataFrame(pd.read_csv(logFile.path, sep=','))
                
                clock_data_total.append(clock_data)

    return clock_data_total

def statsLogData(logFolderPath:str) -> pd.DataFrame:
    """
        Get statistics of LOG Data and convert them to excel.
    """ 

    if not os.path.exists(logFolderPath):
        print("Data folder for LOG files not found.")
        return
    
    df_temp_mean = pd.DataFrame()
    df_temp_sd = pd.DataFrame()
    df_temp_min = pd.DataFrame()
    df_temp_max = pd.DataFrame()
    df_temp_var = pd.DataFrame()
    df_temp_range = pd.DataFrame()
    
    temp_mean = {}
    temp_sd = {} 
    temp_min = {} 
    temp_max = {}
    temp_var = {}
    temp_range = {}

    print("Converting dataframe stats to excel file.")
    # Read LOG files in dir
    with os.scandir(logFolderPath) as logFiles:
        for logFile in logFiles:
            if logFile.is_file():

                clock_freq = logFile.name[:logFile.name.find("-")]
                clock_sn   = logFile.name[logFile.name.find("-")+1:logFile.name.find("_")]

                temp_mean = {'Clock_SN': clock_sn, 'Frequency': clock_freq[2:4]}
                temp_sd   = {'Clock_SN': clock_sn, 'Frequency': clock_freq[2:4]} 
                temp_min  = {'Clock_SN': clock_sn, 'Frequency': clock_freq[2:4]} 
                temp_max  = {'Clock_SN': clock_sn, 'Frequency': clock_freq[2:4]}
                temp_var  = {'Clock_SN': clock_sn, 'Frequency': clock_freq[2:4]}
                temp_range  = {'Clock_SN': clock_sn, 'Frequency': clock_freq[2:4]}
                

                #if clock_freq == "SN10": #Focus on 10MHz clocks
                #print(f"Clock SN: {clock_sn}")
                df = parseLogDataFrame(pd.read_csv(logFile.path, sep=','))

                for param in df.columns.to_list():
                    if param in L_DROP_PARAMS:
                        df.drop(param, axis=1, inplace=True)
                    else:
                        temp_mean[param] = np.mean(df.iloc[:, df.columns.get_loc(param)].values)
                        temp_sd[param]   = np.std(df.iloc[:, df.columns.get_loc(param)].values)
                        temp_min[param]  = np.amin(df.iloc[:, df.columns.get_loc(param)].values)
                        temp_max[param]  = np.amax(df.iloc[:, df.columns.get_loc(param)].values)
                        temp_var[param]  = np.var(df.iloc[:, df.columns.get_loc(param)].values)
                        temp_range[param]  = np.ptp(df.iloc[:, df.columns.get_loc(param)].values, axis=0)

                df_temp_mean = pd.concat([df_temp_mean, pd.DataFrame([temp_mean])], axis=0, ignore_index=True)
                df_temp_sd   = pd.concat([df_temp_sd, pd.DataFrame([temp_sd])], axis=0, ignore_index=True)
                df_temp_min  = pd.concat([df_temp_min, pd.DataFrame([temp_min])], axis=0, ignore_index=True)
                df_temp_max  = pd.concat([df_temp_max, pd.DataFrame([temp_max])], axis=0, ignore_index=True)
                df_temp_var  = pd.concat([df_temp_var, pd.DataFrame([temp_var])], axis=0, ignore_index=True)
                df_temp_range  = pd.concat([df_temp_range, pd.DataFrame([temp_range])], axis=0, ignore_index=True)


                #    df_temp = pd.concat([df_temp, pd.DataFrame(
                #             [{
                #                 param:np.mean(df.iloc[:, df.columns.get_loc(param)].values)
                #             }]
                #         )], axis=0, ignore_index=True  
                #     )
                f_Excel = createDataFolder(logFolderPath, "Excel")
                with pd.ExcelWriter(f_Excel + '/clock_stats.xlsx') as writer:
                    df_temp_mean.to_excel(writer, sheet_name='mean')
                    df_temp_sd.to_excel(writer, sheet_name='sd')
                    df_temp_min.to_excel(writer, sheet_name='min')
                    df_temp_max.to_excel(writer, sheet_name='max')
                    df_temp_var.to_excel(writer, sheet_name='var')
                    df_temp_range.to_excel(writer, sheet_name='range')

    print(f"Successfully written to: {f_Excel}" + r'/clock_stats.xlsx')


class ClockDataset(Dataset):
    '''
        TempCo clock dataset wrapped in a Pytorch dataset class.
        
        Will only include clock params that are not in list L_DROP_PARAMS.
    '''
    def __init__(self, f_data:str, transform=None, training_data=False) -> None:
        super().__init__()
        self.f_clock_paths = [os.path.join(dirpath,f) for (dirpath, dirnames, filenames) in os.walk(f_data) for f in filenames]
        self.transform = transform
        self.training_data = training_data
    
    def __len__(self) -> int:
        return len(self.f_clock_paths)
    
    def __getitem__(self, index):
        clock_path = self.f_clock_paths[index]
        clock_validation = os.path.basename(os.path.dirname(clock_path))
        clock_sn = os.path.basename(clock_path)
        clock_sn = clock_sn[clock_sn.find("-")+1:clock_sn.find("_")]
        
        clock_data = parseLogDataFrame(pd.read_csv(clock_path))
        clock_params = clock_data.iloc[:, :-1].to_numpy()
        clock_param_labels = torch.Tensor(clock_data.iloc[:, -1].to_numpy()).unsqueeze(1).to(torch.int64)
        clock_columns = clock_data.columns.to_list()

        if self.transform:
            s = StandardScaler()
            clock_params = s.fit_transform(clock_data.iloc[:, :-1].to_numpy())
            clock_params = torch.Tensor(clock_params).to(torch.float32)
        else:
            clock_params = torch.Tensor(clock_params).to(torch.float32)

        return clock_params, clock_param_labels, clock_columns, clock_sn, clock_validation

#%%
if __name__ == "__main__":
    TEST_DATASET  = os.path.join(os.getcwd(), r"data\test")
    TRAIN_DATASET = os.path.join(os.getcwd(), r"data\train")
    clock_dataset_test = ClockDataset(f_data=TEST_DATASET)
    clock_dataset_train = ClockDataset(f_data=TRAIN_DATASET)

    # print(f"Length of test dataset:  {clock_dataset_test.__len__()}")
    # print(f"Length of train dataset: {clock_dataset_train.__len__()}")
    # print(clock_dataset_train[0])

    train_dataloader = DataLoader(dataset=clock_dataset_train,
                                  batch_size=1,
                                  shuffle=False)
    
    print(next(iter(train_dataloader)))

# %%
