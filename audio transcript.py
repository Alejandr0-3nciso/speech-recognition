# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 20:57:35 2021

@author: Usuario
"""

import speech_recognition as sr

from pydub import AudioSegment
import pydub

import os
import pandas as pd
import numpy as np
from datetime import datetime

os.chdir('C:\\Users\\Usuario\\web development\\Git\\speech-recognition')



# convert mp3 file to wav  

pydub.AudioSegment.ffmpeg = "/absolute/path/to/ffmpeg"                                                   
sound = AudioSegment.from_mp3("Debate presidencial Colombia Noticias Caracol.mp3")

table_times=pd.read_excel("speech times per candidate.xlsx")


table_times["archivo"]=''
table_times["texto"]=''


table_candidate1=pd.DataFrame()
#table_candidate["archivo"]=''
#table_candidate["texto"]=''
'''
for i in set(table_times.candidato):
    table_candidate=table_times[table_times.candidato==i].reset_index()
    print("Candidato: "+i)
    for j in np.arange(table_candidate.shape[0]):
        
        print("Pregunta # ",table_candidate["Pregunta"].iloc[j])
        
        startMin = table_candidate['min_inicio'].iloc[j]
        startSec = table_candidate['seg_inicio'].iloc[j]

        endMin = table_candidate['min_fin'].iloc[j]
        endSec = table_candidate['seg_fin'].iloc[j]

        # Time to miliseconds
        startTime = startMin*60*1000+startSec*1000
        endTime = endMin*60*1000+endSec*1000

        extract = sound[startTime:endTime]
        file_name=i+str(j+1)+'.wav'
        # Saving
        extract.export( file_name, format="wav")
                     

        # use the audio file as the audio source                                        
        #r = sr.Recognizer()
        #with sr.AudioFile(file_name) as source:
        #    audio = r.record(source)  # read the entire audio file                  

        #    print("Transcription: " + r.recognize_google(audio,language="es-ES"))
        #texto= r.recognize_google(audio,language="es-ES") 

        table_candidate["archivo"].iloc[j]=file_name
        #table_candidate["texto"].iloc[j]=texto
    
    table_candidate1=table_candidate1.append(table_candidate)
    
'''    

startTime = datetime.now()
for j in np.arange(table_times.shape[0]):
        
    print("Pregunta # ",table_times["candidato"].iloc[j])
    print("Pregunta # ",table_times["Pregunta"].iloc[j])
        
    startMin = table_times['min_inicio'].iloc[j]
    startSec = table_times['seg_inicio'].iloc[j]

    endMin = table_times['min_fin'].iloc[j]
    endSec = table_times['seg_fin'].iloc[j]

        # Time to miliseconds
    startTime = startMin*60*1000+startSec*1000
    endTime = endMin*60*1000+endSec*1000

    extract = sound[startTime:endTime]
    file_name=table_times["candidato"].iloc[j]+str(j+1)+'.wav'
        # Saving
    extract.export( file_name, format="wav")
                     

        # use the audio file as the audio source                                        
    r = sr.Recognizer()
    with sr.AudioFile(file_name) as source:
        audio = r.record(source)  # read the entire audio file                  
        print("Transcription: " + r.recognize_google(audio,language="es-ES"))
        texto= r.recognize_google(audio,language="es-ES") 

    table_times["archivo"].iloc[j]=file_name
    table_times["texto"].iloc[j]=texto
    

print(datetime.now() - startTime)    
    
    
table_times.to_csv('Tabla_textos_candidatos.csv')    
  
table_times=pd.read_csv('Tabla_textos_candidatos.csv')  

table_times.columns



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
        
        


