# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 20:57:35 2021

@author: Usuario
"""

import speech_recognition as sr
from os import path
from pydub import AudioSegment
import pydub
import os


os.chdir('G:/Shared drives/Proyecto Migracion')

# convert mp3 file to wav  

pydub.AudioSegment.ffmpeg = "/absolute/path/to/ffmpeg"                                                   
sound = AudioSegment.from_mp3("Copia de lizeth donoso.mp3")
sound.export("transcript.wav", format="wav")


# transcribe audio file                                                         
AUDIO_FILE = "transcript.wav"

# use the audio file as the audio source                                        
r = sr.Recognizer()
with sr.AudioFile(AUDIO_FILE) as source:
        audio = r.record(source)  # read the entire audio file                  

        print("Transcription: " + r.recognize_google(audio))
        
        
        
try:
    [1,2,3][4]
except IndexError:
    print("""IndexError raised""")
except: 
    print("""Exception raised""")
else:
    print("""Somethin else happened""")
finally:
    print("""cleanin""")
    
    
    
def f(): 
    f() 
    return 42

class test():
    id=0
    def __init__(self,id):
        self.id=id
        id=2
        
import re


[i for i in ifilter(lamda x:x % 5,islice(count(5),10))]
