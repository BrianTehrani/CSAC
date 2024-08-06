"""
    File used to create models over CSAC data.
    Tensorflow based.
"""
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# from tensorflow.python.keras.losses import 

import numpy as np

class ClockClassifier_V1():
    """
        ML model used to predict failure regions in CSAC clock data. 
    """
    def __init__(self, input_dim:tuple, output_dim:int) -> None:
        self.model = Sequential(
            [
                Dense(units=32, activation=None, input_dim=input_dim),
                Dense(units=8, activation=None),
                Dense(units=output_dim, activation='sigmoid')
            ]
        )
        self.model.compile(
            optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy', 'val_accuracy']
        )
        self.model.summary()
    
    def train(
            self, X_train:np.array, 
            y_train:np.array, 
            epochs:int = 10, 
            batch_size:int = 32, 
            validation_data:tuple=None
        ):

        history =  self.model.fit(
            x=X_train,
            y=y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data
        )
        return history

    def evaluate(self, X_test:np.array, y_test:np.array):
        return self.model.evaluate(
            X_test,
            y_test
        )
    
    def predict(self, X):
        return self.model.predict(
            x=X
        )
    
    def save(self, folderPath:str, model_num:int):
        self.model.save(
            filepath=folderPath + r'/Models/csac_ml_' + str(model_num) + r'.keras'
        )
        print("Model saved to: " + folderPath + r'/Models/csac_ml_' + str(model_num) + r'.keras')