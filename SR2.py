# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 23:21:33 2022

@author: Max Weinhold
"""

#Bibliotheken für Mel Frequency Cepstral Coefficients (MFCC)
import librosa
import librosa.display
import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np

#Bibliothek um CSV Dateien zu speichen
import pandas as pd

#Dateinamen
Sound_Files = [r"\09b09Nd", 
               r"\03a02Wb", 
               r"\08b01Wa",
               r"\08b02Tc",
               r"\11a05Ad"
               ]

i=0

for audio_file in Sound_Files:
    
    #Fortschrittsanzeige
    i=i+1
    print(str(i/len(Sound_Files)*100)+" %")
    
    #Dateipfad
    str1 = "D:\STUDIUM\Münster\VPRonaRaspberryPi\EmoDB\wav"
    str2 = ".wav"
    #str3 = audiofile
    wavFileName= str1 + audio_file + str2
    
    ipd.Audio(wavFileName,) 

    # load audio files with librosa
    signal, sr = librosa.load(wavFileName)

    #Extracting MFCCs
    mfccs = librosa.feature.mfcc(y=signal, n_mfcc=13, sr=sr)
    mfccs.shape

    #Visualising MFCCs
    #plt.figure(figsize=(25, 10))
    #librosa.display.specshow(mfccs, 
    #                         x_axis="time", 
    #                         sr=sr)
    #plt.colorbar(format="%+2.f")
    #plt.show()
    
    #Teste ob der Frequenz Output die Mindestgröße hat
    if mfccs.size/13 > 50:
        #Alle Outputs auf den selben Stichproben Umfang bringen
        mfccs_resized=np.resize(mfccs,(13,50))
    
        #Speichere MFCCs Output als CSV Datei
        df = pd.DataFrame(mfccs_resized)
        #Dateipfad
        filename="D:\ComicandSonsProductions\GameJam1\SpeakerRecognition\TestData"+audio_file+".csv"
        df.to_csv(filename)
    