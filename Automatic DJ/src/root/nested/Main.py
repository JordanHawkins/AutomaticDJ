#!/usr/bin/env python
# encoding: utf=8
'''
Created on Oct 15, 2012
@author: jordanhawkins
'''
import echonest.audio as audio
import os
#import os.path # this is for looping through the Input file folder
import sys
import plistlib
import shutil
import urllib

workingDirectory = '/Users/jordanhawkins/Documents/workspace/Automatic DJ/src/root/nested'
lib = plistlib.readPlist('/Users/jordanhawkins/Music/iTunes/iTunes Music Library.xml')

def flushDirectory():
    for filename in os.listdir(workingDirectory):
        if(filename != '__init__.py') and (filename != 'Main.py'):
            os.remove(filename)

def getAudioFiles():
    for count in range(len(lib['Playlists'])):
        if lib['Playlists'][count]['Name'] == 'Automatic DJ Input':
            playlistItems = lib['Playlists'][count]['Playlist Items']
            trackIDs = [i['Track ID'] for i in playlistItems]
            for i in range(len(trackIDs)):
                location = lib['Tracks'][str(trackIDs[i])]['Location']
                location = urllib.unquote(location)
                shutil.copy(location[16:], workingDirectory)
            break

def getInput():
    filenames = []
    for filename in os.listdir(workingDirectory):
        if(filename != '__init__.py') and (filename != 'Main.py'):
            filenames.append(filename)
    inputList = []
    for i in range(len(filenames)):
        audiofile = audio.LocalAudioFile(filenames[i])
        inputList.append((audiofile.analysis.tempo.pop('value'), audiofile, filenames[i]))
    inputList.sort()
    sections = [t[1].analysis.sections for t in inputList]
    localAudioFiles = [t[1] for t in inputList]
    filenames = [t[2] for t in inputList]
    return localAudioFiles, sections, filenames

def transition(first_song_sections, second_song_sections, laf1, laf2):
    bars1 = laf1.analysis.bars
    while(len(bars1) > 0):
        if(bars1[len(bars1)-1].confidence > .9):
            break
        bars1.pop()
    bars2 = laf2.analysis.bars
    while(len(bars2) > 0):
        if(bars2[0].confidence > .9):
            break
        bars2.pop(0)    
    return bars1,bars2

def transitionOld(first_song_sections, second_song_sections, laf1, laf2):
    first_song_output = audio.AudioQuantumList()
    for section in range(len(first_song_sections) - 1):
        
        if(section == len(first_song_sections)-2):
            bars = first_song_sections[section].children()
            while(len(bars) > 0):
                if(bars[len(bars)-1].confidence > .1):
                    break
                bars.pop()
            first_song_sections[section] = audio.getpieces(laf1, bars)       
        first_song_output.append(first_song_sections[section])
    second_song_output = audio.AudioQuantumList()
    for section in range(len(second_song_sections) - 1):
        second_song_output.append(second_song_sections[section+1])
    return first_song_output,second_song_output

def generateOutput(sections, localAudioFiles, filenames): 
    for count in range(len(sections)):
        out = audio.getpieces(localAudioFiles[count], sections[count])
        out.encode(filenames[count])
    os.system('automator /Users/jordanhawkins/Documents/workspace/Automatic\ DJ/import.workflow/')    
    
def main():
    flushDirectory()
    getAudioFiles()
    localAudioFiles, sections, filenames = getInput()
    for count in range(len(sections)-1):
        sections[count], sections[count + 1] = transition(sections[count],sections[count+1],localAudioFiles[count], localAudioFiles[count+1])
    generateOutput(sections, localAudioFiles, filenames)
    
if __name__ == '__main__':
    main()

    