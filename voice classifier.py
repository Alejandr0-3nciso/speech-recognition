# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 20:11:43 2021

@author: Usuario
"""

import IPython.display as ipd
# % pylab inline
import os
import pandas as pd
import librosa
import glob 
import librosa.display
import random

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn import metrics 

from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout 
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

from keras.callbacks import EarlyStopping

from keras import regularizers

from sklearn.preprocessing import LabelEncoder

from datetime import datetime

import os



table_candidate1["label"]=np.where(table_candidate1["candidato"]=="Ivan Duque",0,
                          np.where(table_candidate1["candidato"]=="Humberto de la Calle",1,
                          np.where(table_candidate1["candidato"]=="German Vargas",2,
                          np.where(table_candidate1["candidato"]=="Sergio Fajardo",3,4))))

# Although this function was modified and many parameteres were explored with, most of it
# came from Source 8 (sources in the READ.ME)

def extract_features(files):
    
    # Sets the name to be the path to where the file is in my computer
    file_name = os.path.join(os.path.abspath(str(files.archivo)))

    # Loads the audio file as a floating point time series and assigns the default sample rate
    # Sample rate is set to 22050 by default
    X, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 

    # Generate Mel-frequency cepstral coefficients (MFCCs) from a time series 
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)

    # Generates a Short-time Fourier transform (STFT) to use in the chroma_stft
    stft = np.abs(librosa.stft(X))

    # Computes a chromagram from a waveform or power spectrogram.
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)

    # Computes a mel-scaled spectrogram.
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)

    # Computes spectral contrast
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)

    # Computes the tonal centroid features (tonnetz)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
    sr=sample_rate).T,axis=0)
        
    
    # We add also the classes of each file as a label at the end
    label = files.label

    return mfccs, chroma, mel, contrast, tonnetz, label


# Code to start the timer to see how long it takes to extract the features
startTime = datetime.now()

for i in range(0,4):
    features_label = table_candidate1.apply(extract_features, axis=1)

print(datetime.now() - startTime)



features_label 

np.save('features_label', features_label)


# The next code loads the saved numpy array of our extracted features
# features_label = np.load('features_label.npy', allow_pickle=True)


features = []
for i in range(0, len(features_label)):
    for j in range(0,len(features_label[i])-1)
    features.append(np.concatenate((features_label[0][i], features_label[i][1], 
                features_label[i][2], features_label[i][3],
                features_label[i][4]), axis=0))



