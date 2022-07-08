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

Sound_Files = [r"D:\STUDIUM\Münster\VPRonaRaspberryPi\EmoDB\wav\09b09Nd.wav", 
               r"D:\STUDIUM\Münster\VPRonaRaspberryPi\EmoDB\wav\03a02Wb.wav", 
               r"D:\STUDIUM\Münster\VPRonaRaspberryPi\EmoDB\wav\08b01Wa.wav",
               r"D:\STUDIUM\Münster\VPRonaRaspberryPi\EmoDB\wav\08b02Tc.wav",
               r"D:\STUDIUM\Münster\VPRonaRaspberryPi\EmoDB\wav\11a05Ad.wav"
               ]


for audio_file in Sound_Files:
    print(audio_file)
    
    ipd.Audio(audio_file,) 

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
    
    #Teste ob der Frequenz Output die Mindestgröße hat
    print(mfccs.size/13)
    
    if mfccs.size/13 > 50:
        #Alle Outputs auf den selben Stichproben Umfang bringen
        mfccs_resized=np.resize(mfccs,(13,50))
    
        #Speichere MFCCs Output als CSV Datei
        df = pd.DataFrame(mfccs_resized)
        df.to_csv(r'D:\ComicandSonsProductions\GameJam1\SpeakerRecognition\TestData\TestData.csv')
        #mfccs.to_csv (r'D:\ComicandSonsProductions\GameJam1\SpeakerRecognition\TestData', index = False, header=True)
    