## Machine Learning Production Validation of Chip-Scale Atomic Clocks (CSAC)

### Introduction
Teledyne Scientific and Imaging Internal Research and Development Program to create a machine learning model which can predict fail regions within a span of trail run  CSAC data.

### About Repository
#### Goal
To house data handling and machine learning code behind program effort.

#### File Descriptions
- main.py
  - Train ML models after data handling and model creation
- dataHandler.py
  - Data analysis scripts to visualize and select CSAC experiment data for ML model input
- classifiers.py
  - Store variations of ML models (PyTorch) to apply after data handling CSAC parameters
- model_eval.py
  - Evaluate ML models after training. Creates validation plots to view ML metrics and performance.
- Folders
  - tests
    - Scripts to test various applications to data handling and model creation prior to integrating to above files
  - docs
    - Documentation related to descriptions behind experimentation and programatics
  - tf (depricated)
    - TensorFlow implemented ML models 
