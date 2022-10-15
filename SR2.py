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

#Bibliothek um CSV Dateien zu lesen und zu speichen
import pandas as pd
import csv

#Ließ die CSV Datei, die alle Namen der SoundFiles enthält
with open('D:\STUDIUM\Münster\VPRonaRaspberryPi\EmoDB\wav\\file_list.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)

print(data)

#Dateinamen
Sound_Files=data

i=0

#Variable Adjustments
mfcc_freq_quantity = 26
minimum_size = 50


#bigDF = np.empty((0, 13*50))
big_array = np.empty((len(Sound_Files), mfcc_freq_quantity*minimum_size))
#lst = []

for audio_file in Sound_Files:
    
    #Fortschrittsanzeige
    i=i+1
    print(str(i/len(Sound_Files)*100)+" %")
    
    #Dateipfad
    str1 = "D:\STUDIUM\Münster\VPRonaRaspberryPi\EmoDB\wav\\"

    wavFileName = str1 + audio_file[0]
    
    ipd.Audio(wavFileName,) 

    # load audio files with librosa
    signal, sr = librosa.load(wavFileName)

    #Extracting MFCCs
    mfccs = librosa.feature.mfcc(y=signal, n_mfcc=mfcc_freq_quantity, sr=sr)
    mfccs.shape
    
    #Teste ob der Frequenz Output die Mindestgröße hat
    if mfccs.size/mfcc_freq_quantity > minimum_size:
        #Alle Outputs auf den selben Stichproben Umfang bringen
        mfccs_resized=np.resize(mfccs,(mfcc_freq_quantity,minimum_size))
    
        #Kovertiere den zwei diemnsionalen Array in eine Dimension
        flat_array = mfccs_resized.flatten()
    
        #Speichere MFCCs Output als CSV Datei
        df = pd.DataFrame(flat_array)
        #Dateipfad
        filename="D:\ComicandSonsProductions\GameJam1\SpeakerRecognition\TestData\\"+audio_file[0]+".csv"
        df.to_csv(filename)
        
        #Kombiniere alle Outputs
        big_array[i-1]=flat_array
        big_array=big_array
        
        #Speichere Nummer der Stimme
        name = audio_file[0]+"name"
        big_array[i-1][0]=name[:2]
        
        #Speichere MFCCs Output als CSV Datei
        bigDF = pd.DataFrame(big_array)
        
        for j in range(len(bigDF.index)):
           bigDF[0][j]="Voice"+str(bigDF[0][j])
        
        #Dateipfad
        filename="D:\ComicandSonsProductions\GameJam1\SpeakerRecognition\TestData\\Test.csv"
        bigDF.to_csv(filename)
    