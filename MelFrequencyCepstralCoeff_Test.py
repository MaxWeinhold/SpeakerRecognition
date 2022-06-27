# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 23:25:16 2022

@author: @musikalkemist (GitHub)
"""

#Mel Frequency Cepstral Coefficients (MFCC)

#Bibliotheken für Mel Frequency Cepstral Coefficients (MFCC)
import librosa
import librosa.display
import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np

audio_file = "D:\STUDIUM\Münster\VPRonaRaspberryPi\greeting_4_sean.wav"
ipd.Audio(audio_file)

# load audio files with librosa
signal, sr = librosa.load(audio_file)

#Extracting MFCCs
mfccs = librosa.feature.mfcc(y=signal, n_mfcc=13, sr=sr)
mfccs.shape

#Visualising MFCCs
plt.figure(figsize=(25, 10))
librosa.display.specshow(mfccs, 
                         x_axis="time", 
                         sr=sr)
plt.colorbar(format="%+2.f")
plt.show()

print(mfccs)

#Computing first / second MFCCs derivatives
delta_mfccs = librosa.feature.delta(mfccs)

delta2_mfccs = librosa.feature.delta(mfccs, order=2)

delta_mfccs.shape

plt.figure(figsize=(25, 10))
librosa.display.specshow(delta_mfccs, 
                         x_axis="time", 
                         sr=sr)
plt.colorbar(format="%+2.f")
plt.show()

plt.figure(figsize=(25, 10))
librosa.display.specshow(delta2_mfccs, 
                         x_axis="time", 
                         sr=sr)
plt.colorbar(format="%+2.f")
plt.show()

mfccs_features = np.concatenate((mfccs, delta_mfccs, delta2_mfccs))

mfccs_features.shape

print(mfccs_features)