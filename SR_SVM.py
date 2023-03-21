# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 23:16:01 2022

@author: MaxWe
"""
#About SVM
#Effective in high dimensional spaces.
#Still effective in cases where number of dimensions is greater than the number of samples.
#Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.
#Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also

import matplotlib.pyplot as plt
import numpy as np

#Bibliothek um CSV Dateien zu lesen und zu speichen
import pandas as pd
import csv

#Import SVM Library
#from sklearn import svm
#https://holypython.com/svm/support-vector-machine-step-by-step/

import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split as tts
from sklearn import svm
from sklearn import metrics

#Ließ die CSV Datei, die alle Namen der SoundFiles enthält
with open("D:\ComicandSonsProductions\GameJam1\SpeakerRecognition\TestData\\Frequencies60.csv", newline='') as f:
    reader = csv.reader(f)
    data = list(reader)

voice_data = pd.DataFrame(data)
voice_column = voice_data.iloc[:, 1]
sound_columns = voice_data.iloc[: , 2:]

###Splitting train/test data
X_tr, X_tst, y_tr, y_tst = tts(sound_columns , voice_column , test_size=25/100,random_state=109)

###Creating Support Vector Machine Model
clf = svm.SVC(kernel='rbf', shrinking=True)

###Training the Model
clf.fit(X_tr, y_tr)

###Making Predictions
y_pr = clf.predict(X_tst)
print(y_pr)

###Evaluating Prediction Accuracy
print("Accuracy:",metrics.accuracy_score(y_tst, y_pr))
