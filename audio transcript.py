# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 20:57:35 2021

@author: Usuario
"""

import speech_recognition as sr
from os import path
from pydub import AudioSegment
import pydub
import pyaudio
import os
import pandas as pd
import numpy as np

os.chdir('C:\\Users\\Usuario\\web development\\Git\\speech-recognition')



# convert mp3 file to wav  

pydub.AudioSegment.ffmpeg = "/absolute/path/to/ffmpeg"                                                   
sound = AudioSegment.from_mp3("Debate presidencial Colombia Noticias Caracol.mp3")

table_times=pd.read_excel("speech times per candidate.xlsx")


for i in set(table_times.candidato):
    table_candidate=table_times[table_times.candidato==i].reset_index()
    for j in np.arange(table_candidate.shape[0]):
    
        startMin = table_candidate['min_inicio'].iloc[j]
        startSec = table_candidate['seg_inicio'].iloc[j]

        endMin = table_candidate['min_fin'].iloc[j]
        endSec = table_candidate['seg_fin'].iloc[j]

        # Time to miliseconds
        startTime = startMin*60*1000+startSec*1000
        endTime = endMin*60*1000+endSec*1000

        extract = sound[startTime:endTime]
        file_name=i+str(j)+'.wav'
        # Saving
        extract.export( file_name, format="wav")
                     

        # use the audio file as the audio source                                        
        r = sr.Recognizer()
        with sr.AudioFile(file_name) as source:
            audio = r.record(source)  # read the entire audio file                  

            print("Transcription: " + r.recognize_google(audio,language="es-ES"))






sound.export("transcript.wav", format="wav")



halfway_point = len(sound) // 800
len(sound)//halfway_point 
halfway_point
len(sound)
# transcribe audio file                                                         
AUDIO_FILE = "transcript.wav"                     

# use the audio file as the audio source                                        
r = sr.Recognizer()
with sr.AudioFile(AUDIO_FILE) as source:
        audio = r.record(source)  # read the entire audio file                  

        print("Transcription: " + r.recognize_google(audio))
        
        


