#!/usr/bin/env python
# encoding: utf=8
'''
Created on Oct 15, 2012
@author: jordanhawkins
'''
import echonest.audio as audio
import echonest.modify as modify
import os
import plistlib
import shutil
import urllib
import numpy.matlib as matlib
import sys
import matplotlib.pyplot as pyplot

workingDirectory = '/Users/jordanhawkins/Documents/workspace/Automatic DJ/src/root/nested'
lib = plistlib.readPlist('/Users/jordanhawkins/Music/iTunes/iTunes Music Library.xml')
LOUDNESS_THRESH = -8 # per capsule_support module

def flushDirectory():
    for filename in os.listdir(workingDirectory):
        if(filename != '__init__.py') and (filename != 'Main.py') and (filename != 'AutoMashUp.py'):
            os.remove(filename)
                   
def getAudioFiles():
    for count in range(len(lib['Playlists'])):
        if lib['Playlists'][count]['Name'] == 'Automatic MashUp Input':
            playlistItems = lib['Playlists'][count]['Playlist Items']
            trackIDs = [i['Track ID'] for i in playlistItems]
            for i in range(len(trackIDs)):
                location = lib['Tracks'][str(trackIDs[i])]['Location']
                location = urllib.unquote(location)
                try:
                    shutil.copy(location[16:], workingDirectory)
                except:
                    print "exception in getAudioFiles         "
            break

def getInput():
    filenames = []
    for filename in os.listdir(workingDirectory):
        if(filename != '__init__.py') and (filename != 'Main.py') and (filename != 'AutoMashUp.py'):
            filenames.append(filename)
    inputList = []
    for i in range(len(filenames)):
        audiofile = audio.LocalAudioFile(filenames[i])
        inputList.append((audiofile.analysis.tempo.pop('value'), audiofile, filenames[i]))
    inputList.sort()
    tempos = [t[0] for t in inputList]
    localAudioFiles = [t[1] for t in inputList]
    filenames = [t[2] for t in inputList]
    return localAudioFiles, filenames, tempos
 
def equalize_tracks(tracks):   # copied from capsule_support module 
    def db_2_volume(loudness):
        return (1.0 - LOUDNESS_THRESH * (LOUDNESS_THRESH - loudness) / 100.0)   
    for track in tracks:
        loudness = track.analysis.loudness
        track.gain = db_2_volume(loudness)
   
def deleteOldSongs(filenames):
    for filename in os.listdir(workingDirectory):
        if filename in filenames:
            os.remove(filename)

def matchTempoAndKey(localAudioFiles, tempos):
    ratio = (tempos[1]-tempos[0])/tempos[0]
    mod = modify.Modify(sampleRate=44100, numChannels=1, blockSize=10000)
    print "before mod1"
    modify.Modify.shiftTempo(mod, localAudioFiles[0],ratio)
    print "after mod"
    return localAudioFiles

def main(): 
    flushDirectory()
    getAudioFiles()
    localAudioFiles, filenames, tempos = getInput()
    equalize_tracks(localAudioFiles)
    localAudioFiles = matchTempoAndKey(localAudioFiles, tempos)
    align()
    deleteOldSongs(filenames)
    os.system('automator /Users/jordanhawkins/Documents/workspace/Automatic\ DJ/import.workflow/')      
    
if __name__ == '__main__':
    main()

