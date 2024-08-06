# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 18:00:18 2024

@author: btehrani


NOTE: File is used for tensorflow implementations.
      As of this note (7-2-24), tensorflow's latest version only runs on Windows CPU.
"""

#%% Imports
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import dataHandler #Spyder has issues locating file src.dataHandler as
import tf.models as models
#import tensorflow as tf

import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

#%% 1) Obtain clock data from specified data folders.
clock_data_total_fail: list[dict] = dataHandler.logFileToDict(dataHandler.DATAFOLDER_FAIL)

model = models.ClockClassifier_V1(input_dim=len(clock_data_total_fail[0]['df'].columns[1:-1]), output_dim=1)
#%% 2) Split clock data into test and training sets
for clock in clock_data_total_fail[0:2]:
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

    # 4) Train Model
    history = model.train(X_train_sc, y_train, epochs=10, batch_size=8, validation_data=(X_test_sc, y_test))

#%% 5) Evaluate the model.
test_loss, test_acc = model.evaluate(X_test_sc, y_test)
print("\nTest accuracy: ", test_acc, " Test loss: ", test_loss)

fig = plt.plot()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

#%% Test model on sample data:

clock_num = 10 

print("Clock sn: ", clock_data_total_fail[clock_num]['sn'])
predictions = model.predict(clock_data_total_fail[clock_num]['df'].iloc[:, 1:-1].values)
print(len(predictions))
plt.plot(predictions)
plt.title("Fail Predictions")
plt.xlabel('secs')
plt.ylabel('pass/fail')
plt.show()

#%% 6) Save the model
model.save(dataHandler.DATAFOLDER_FAIL, 4)


# %%
