# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 22:15:50 2022

@author: MaxWeinhold
"""

#Bibliotheken für Audio Dateien
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
import numpy as np

#Bibliotheken für Mel Frequency Cepstral Coefficients (MFCC)
import librosa
import librosa.display
import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np

#Lade Audio Datei und erstelle einen Plott

samplerate, data = wavfile.read("D:\STUDIUM\Münster\VPRonaRaspberryPi\greeting_4_sean.wav")
plt.plot(data)
plt.ylabel("dB")
plt.xlabel('Time')
plt.title("Full Audio File _a")
plt.show()

#Wie viele Bruchstücke machst du aus der Audiodatei

l=3000 #Stichprobenlänge
sections=len(data)//l
print(len(data))

for i in range(0,sections-1):
    
    #Spalte Audio Datei in mehrere Sektionen
    
    if i*l+l < len(data):
        y=data[i*l+1:i*l+l]
    else:
        y=data[i*l+1:len(data)]
    plt.plot(y)
    plt.ylabel("dB")
    plt.xlabel('Time')
    plt.title("Audio Section "+str(i))
    plt.show()
    
    #Extracting MFCCs
    #mfccs = librosa.feature.mfcc(y=signal, n_mfcc=13, sr=y)