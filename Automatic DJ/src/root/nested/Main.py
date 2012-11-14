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
import matplotlib.pyplot as pyplot

workingDirectory = '/Users/jordanhawkins/Documents/workspace/Automatic DJ/src/root/nested'
lib = plistlib.readPlist('/Users/jordanhawkins/Music/iTunes/iTunes Music Library.xml')
BEAT_CONFIDENCE = .9
BAR_CONFIDENCE = .8

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
                try:
                    shutil.copy(location[16:], workingDirectory)
                except:
                    print "exception in getAudioFiles         "
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
    localAudioFiles = [t[1] for t in inputList]
    sections = [t[1].analysis.sections for t in inputList]
    filenames = [t[2] for t in inputList]
    return localAudioFiles, sections, filenames

def Oldmodify(sections, localAudioFiles, filenames):
    sections = [s[1:-1] for s in sections]
    beats = [f.analysis.beats for f in localAudioFiles]
    bars = [f.analysis.bars for f in localAudioFiles]
    exceptions = []
    for i in range(len(sections)):
        try:
            while (bars[i][0].start < sections[i][0].start) or (bars[i][0].confidence < BAR_CONFIDENCE):
                bars[i].pop(0) 
            print "first bar confidence: ", bars[i][0].confidence
            while (bars[i][-1].start > sections[i][-1].start) or (bars[i][-1].confidence < BAR_CONFIDENCE):
                bars[i].pop()
            print "last bar confidence: ", bars[i][-1].confidence
            while (beats[i][0].start < sections[i][0].start) or (not selection.fall_on_the(1)(beats[i][0])) or (not len(beats[i][0].group()) == 4) or (beats[i][0].confidence < BEAT_CONFIDENCE):
                beats[i].pop(0)
            print "Selection fall on the 1? ", selection.fall_on_the(1)(beats[i][0])
            print "beats[i][3] fall on the 4? ",selection.fall_on_the(4)(beats[i][3])
            print "length of beats.group(): ", len(beats[i][0].group())
            print "confidence of first bar: ", beats[i][0].group().confidence
            while (beats[i][-1].start > sections[i][-1].start) or (not selection.fall_on_the(4)(beats[i][-1])) or (not len(beats[i][-1].group()) == 4) or (beats[i][-1].confidence < BEAT_CONFIDENCE) or (not len(beats[i]) % 4 == 0):
                beats[i].pop()
            print "Selection fall on the 4? ", selection.fall_on_the(4)(beats[i][-1])
            print "beats[i][-4] fall on the 1? ",selection.fall_on_the(1)(beats[i][-4])
            print "length of beats.group(): ", len(beats[i][-1].group())    
            print "length of beats mod 4: ", len(beats[i]) % 4
            print "confidence of last bar: ", beats[i][-1].group().confidence            
        except: 
            print "the problematic song is: ", filenames[i]
            exceptions.append(filenames[i])      
    for i in exceptions:
        index = filenames.index(i)
        filenames.remove(i)
        localAudioFiles.pop(index)
        beats.pop(index)
        bars.pop(index)
    return localAudioFiles, filenames, beats, bars

def modify(sections, localAudioFiles, filenames):
    beats = [f.analysis.beats for f in localAudioFiles]
    bars = [f.analysis.bars for f in localAudioFiles]
    exceptions = []
    for i in range(len(sections)):
        try:
            while bars[i][0].confidence < BAR_CONFIDENCE:
                bars[i].pop(0) 
            while bars[i][-1].confidence < BAR_CONFIDENCE:
                bars[i].pop()
            print "This song actually worked! -> ", filenames[i]
        except: 
            #print "the problematic song is: ", filenames[i]
            exceptions.append(filenames[i])      
    for i in exceptions:
        index = filenames.index(i)
        filenames.remove(i)
        localAudioFiles.pop(index)
        beats.pop(index)
        bars.pop(index)
    return localAudioFiles, filenames, beats, bars

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
    #action.render(actions, "totalCrossmatched.mp3")
    
def OldgenerateHipHopSlam(localAudioFiles, beats, filenames):
    actions = [action.Playback(localAudioFiles[i], beats[i][0].start, sum(beats[i][j].duration for j in range(len(beats[i])))) for i in range(len(beats))]
    for i in range(len(actions)): action.render([actions[i]],str(i) + " " + filenames[i])
    #action.render(actions, "totalHipHopSlam.mp3")
    
def generateHipHopSlam(localAudioFiles, bars, filenames):
    actions = [action.Playback(localAudioFiles[i], bars[i][0].start, sum(bars[i][j].duration for j in range(len(bars[i])))) for i in range(len(bars))]
    for i in range(len(actions)): action.render([actions[i]],str(i) + " " + filenames[i])
    #action.render(actions, "totalHipHopSlam.mp3")
    
def generateBarConfidenceGraphs(bars, filenames):
    for i in range(len(bars)):
        pyplot.figure(i) 
        pyplot.plot([j.confidence for j in bars[i]])
        pyplot.xlabel('Bar Number')
        pyplot.ylabel('Confidence')
        pyplot.title(filenames[i])
    pyplot.show()
    
def generateBeatConfidenceGraphs(beats, filenames):
    for i in range(len(beats)):
        pyplot.figure(i) 
        pyplot.plot([j.confidence for j in beats[i]])
        pyplot.xlabel('Beat Number')
        pyplot.ylabel('Confidence')
        pyplot.title(filenames[i])
    pyplot.show()
    
def generateBarDurationGraphs(bars, filenames):
    for i in range(len(bars)):
        pyplot.figure(i) 
        pyplot.plot([j.duration for j in bars[i]])
        pyplot.xlabel('Bar Number')
        pyplot.ylabel('Duration')
        pyplot.title(filenames[i])
    pyplot.show()
    
def generateBeatDurationGraphs(beats, filenames):
    for i in range(len(beats)):
        pyplot.figure(i) 
        pyplot.plot([j.duration for j in beats[i]])
        pyplot.xlabel('Beat Number')
        pyplot.ylabel('Duration')
        pyplot.title(filenames[i])
    pyplot.show()
    
def generateGraphs(beats, bars, filenames):
    for i in range(len(beats)):
        pyplot.figure(i)
        pyplot.subplot(221)
        pyplot.plot([j.duration for j in beats[i]])
        pyplot.xlabel('Beat Number')
        pyplot.ylabel('Duration')
        pyplot.title(filenames[i])
        pyplot.subplot(222)
        pyplot.plot([j.duration for j in bars[i]])
        pyplot.xlabel('Bar Number')
        pyplot.ylabel('Duration')
        pyplot.subplot(223)
        pyplot.plot([j.confidence for j in beats[i]])
        pyplot.xlabel('Beat Number')
        pyplot.ylabel('Confidence')
        pyplot.subplot(224)
        pyplot.plot([j.confidence for j in bars[i]])
        pyplot.xlabel('Bar Number')
        pyplot.ylabel('Confidence')
    pyplot.show()
    
def main(): 
    flushDirectory()
    getAudioFiles()
    localAudioFiles, sections, filenames = getInput()
    generateGraphs([f.analysis.beats for f in localAudioFiles],[f.analysis.bars for f in localAudioFiles],filenames)
    localAudioFiles, newFilenames, beats, bars = modify(sections, localAudioFiles, filenames)
    generateHipHopSlam(localAudioFiles, bars, newFilenames)
    deleteOldSongs(filenames)
    os.system('automator /Users/jordanhawkins/Documents/workspace/Automatic\ DJ/import.workflow/')      
    
if __name__ == '__main__':
    main()

    