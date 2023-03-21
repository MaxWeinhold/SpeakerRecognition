# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 17:54:40 2022

@author: MaxWe
"""

import matplotlib.pyplot as plt
import numpy as np

#Bibliothek um CSV Dateien zu lesen und zu speichen
import pandas as pd
import csv

import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split as tts

#Ließ die CSV Datei, die alle Namen der SoundFiles enthält
with open("D:\ComicandSonsProductions\GameJam1\SpeakerRecognition\TestData\\Test.csv", newline='') as f:
    reader = csv.reader(f)
    data = list(reader)
        
voice_data = pd.DataFrame(data)
voice_column = voice_data.iloc[:, 1]
sound_columns = voice_data.iloc[: , 2:]

###Splitting train/test data
X_tr, X_tst, y_tr, y_tst = tts(sound_columns , voice_column , test_size=25/100,random_state=109)

# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 500, random_state = 42)
# Train the model on training data
rf.fit(X_tr, y_tr);

# Make predictions and determine the error
predictions = rf.predict(X_tst)
errors = abs(predictions - y_tst)
# Display the performance metrics
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
mape = np.mean(100 * (errors / y_tst))
accuracy = 100 - mape
print('Accuracy:', round(accuracy, 2), '%.')
