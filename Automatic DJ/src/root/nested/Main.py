#!/usr/bin/env python
# encoding: utf=8
'''
Created on Oct 15, 2012
@author: jordanhawkins
'''
import echonest.audio as audio
import echonest.action as action
import echonest.selection as selection
import os
import plistlib
import shutil
import urllib
import sys

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
    print "inputList is: ", inputList
    localAudioFiles = [t[1] for t in inputList]
    sections = [t[1].analysis.sections for t in inputList]
    filenames = [t[2] for t in inputList]
    return localAudioFiles, sections, filenames

def modify(sections, localAudioFiles, filenames):
    sections = [s[1:-1] for s in sections]
    beats = [f.analysis.beats for f in localAudioFiles]
    for i in range(len(sections)):
        try:
            while (beats[i][0].start < sections[i][0].start) or (not selection.fall_on_the(1)(beats[i][0])) or (not len(beats[i][0].group()) == 4):
                beats[i].pop(0)
            while (beats[i][-1].start > sections[i][-1].start) or (not selection.fall_on_the(4)(beats[i][-1])):
                beats[i].pop()
        except: 
            print "the problematic song is: ", filenames[i]      
    return beats

def deleteOldSongs(filenames):
    for filename in os.listdir(workingDirectory):
        if filename in filenames:
            os.remove(filename)
       
def generateCrossmatch(localAudioFiles, beats, filenames):
    actions = [action.Crossmatch((localAudioFiles[i], localAudioFiles[i+1]), ([(t.start, t.duration) for t in beats[i][-1].group()],[(t.start, t.duration) for t in beats[i+1][0].group()])) for i in range(len(beats)-1)]
    for i in range(len(beats)): 
        actions.insert(2*i, action.Playback(localAudioFiles[i], beats[i][4].start, sum([beats[i][4+j].duration for j in range(len(beats[i])-8)])))
    for i in range(len(actions)/2):
        action.render([actions[2*i],actions[2*i+1]], str(i) + " " + filenames[i])                       
    action.render([actions[-1]], str(len(filenames)-1) + " " + filenames[-1])
    action.render(actions, "totalCrossmatched.mp3")
    
def main(): 
    flushDirectory()
    getAudioFiles()
    localAudioFiles, sections, filenames = getInput()
    beats = modify(sections, localAudioFiles, filenames)
    generateCrossmatch(localAudioFiles, beats, filenames)
    deleteOldSongs(filenames)
    os.system('automator /Users/jordanhawkins/Documents/workspace/Automatic\ DJ/import.workflow/')      
    
if __name__ == '__main__':
    main()

    