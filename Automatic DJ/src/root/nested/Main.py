#!/usr/bin/env python
# encoding: utf=8
'''
Created on Oct 15, 2012
@author: jordanhawkins
'''
import echonest.audio as audio
import echonest.action as action
import os
import plistlib
import shutil
import urllib
import numpy.matlib as matlib
import sys
import matplotlib.pyplot as pyplot

workingDirectory = '/Users/jordanhawkins/Documents/workspace/Automatic DJ/src/root/nested'
lib = plistlib.readPlist('/Users/jordanhawkins/Music/iTunes/iTunes Music Library.xml')
BEAT_CONFIDENCE = .9
BAR_CONFIDENCE = .8
LOUDNESS_THRESH = -8 # per capsule_support module
STD_WINDOW_LENGTH = 9

def flushDirectory():
    for filename in os.listdir(workingDirectory):
        if(filename != '__init__.py') and (filename != 'Main.py') and (filename != 'AutoMashUp.py'):
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

def generateSegmentGraphs(segments, filenames, segmentMarkers):
    for i in range(len(segments)):
        pyplot.figure(i)
        pyplot.subplot(211)
        pyplot.plot([j.loudness_max for j in segments[i]])
        pyplot.xlabel('Segment Number')
        pyplot.ylabel('Loudness_Max (dB)')
        pyplot.vlines(segmentMarkers[i][0], min([j.loudness_max for j in segments[i]]), max([j.loudness_max for j in segments[i]]))
        pyplot.vlines(segmentMarkers[i][1], min([j.loudness_max for j in segments[i]]), max([j.loudness_max for j in segments[i]]))
        pyplot.title(filenames[i])       
        # now calculate mean average deviation using an arbitrary window length 9
        pyplot.subplot(212)
        mad = [0] * STD_WINDOW_LENGTH
        for k in range(len(segments[i])-STD_WINDOW_LENGTH):
            s = matlib.std(matlib.array([segments[i][k+p].loudness_max for p in range(STD_WINDOW_LENGTH)]))
            mad.insert((len(mad)-(STD_WINDOW_LENGTH/2)),s)            
        pyplot.plot(mad)
        pyplot.xlabel('Segment Number')
        pyplot.ylabel('Standard Deviation, with window length = ' + str(STD_WINDOW_LENGTH))
    pyplot.show()

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
    localAudioFiles = [t[1] for t in inputList]
    filenames = [t[2] for t in inputList]
    return localAudioFiles, filenames
 
def equalize_tracks(tracks):   # copied from capsule_support module 
    def db_2_volume(loudness):
        return (1.0 - LOUDNESS_THRESH * (LOUDNESS_THRESH - loudness) / 100.0)   
    for track in tracks:
        loudness = track.analysis.loudness
        track.gain = db_2_volume(loudness)
   
def modify(sections, localAudioFiles, filenames):
    segments = [f.analysis.segments for f in localAudioFiles]
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
        segments.pop(index)
        bars.pop(index)
    return localAudioFiles, filenames, segments, bars

def deleteOldSongs(filenames):
    for filename in os.listdir(workingDirectory):
        if filename in filenames:
            os.remove(filename)

def runTrainingSet(localAudioFiles, segments, filenames):
    # ordered transition filenames:  ['02 Bangarang.mp3', 'Move Your Feet.mp3', '01 Lights.m4a', '2-20 We Are The People.mp3', '06 Hearts On Fire.mp3', 'Cry (Just A Little) (Original Mix).mp3', '01 Call On Me.mp3', 'Believer.mp3', '2-08 Feel So Close.mp3', '2-21 I Remember.mp3', '01 Til Death (Denzal Park Radio Edit).mp3', 'LE7ELS (Original Mix).mp3', '02 When Love Takes Over (Featuring Kelly Rowland).mp3', '01 Spectrum (feat. Matthew Koma) [Extended Mix].mp3', '11 Pins.mp3', '07 Downforce 1.mp3', 'Sandstorm.m4a', '09 Dreamcatcher.mp3', '04 Somebody Told Me.mp3', '08 What You Know.mp3']
    timeMarkers = [(26.516,131.312),(4.746,172.450),(41.044,201.012),(82.312,175.997),(15.370,46.003),(122.042,213.469),(30.887,122.294),(0.000,272.304),(37.785,195.357),(15.230,195.357),(37.539,172.498),(67.721,157.716),(37.282,125.899),(147.876,325.127),(14.775,192.008),(213.437,298.800),(29.553,86.022),(238.297,294.371),(21.150,193.356),(41.625,138.350)]
    # Locate the closest segment to the specified time markers
    segmentMarkers = []
    for i,segment in enumerate(segments):
        beginSegment = 0
        while(segment[beginSegment].start < timeMarkers[i][0]):
            beginSegment += 1
        #check to make sure the previous segment wasn't closer to the marker than the current one which overshot it...
        if((timeMarkers[i][0] - segment[beginSegment-1].start) < (segment[beginSegment].start - timeMarkers[i][0])):
            beginSegment -= 1 
        endSegment = len(segment)-1
        while(segment[endSegment].start > timeMarkers[i][1]):
            endSegment -= 1
        # again check to make sure the previous start value wasn't closer than this current one
        if((segment[endSegment+1].start - timeMarkers[i][1]) < (timeMarkers[i][1] - segment[endSegment].start)):
            endSegment += 1
        segmentMarkers.append((beginSegment, endSegment))
    generateSegmentGraphs(segments,filenames,segmentMarkers)
    sys.exit()
    # Now check and see if it spits out similar songs to mine...
    out = []
    for marker,segment in zip(segmentMarkers, segments): 
        if marker[0] == -1: marker = (0,marker[1])
        segment = segment[marker[0]:marker[1]]
        out.append(segment)
    segments = out
    actions = [action.Playback(localAudioFiles[i], segments[i][0].start, sum(segments[i][j].duration for j in range(len(segments[i])))) for i in range(len(segments))]
    for i in range(len(actions)): action.render([actions[i]],str(i) + " " + filenames[i])
             
       
def generateCrossmatch(localAudioFiles, segments, filenames):
    actions = [action.Crossmatch((localAudioFiles[i], localAudioFiles[i+1]), ([(t.start, t.duration) for t in segments[i][-1].group()],[(t.start, t.duration) for t in segments[i+1][0].group()])) for i in range(len(segments)-1)]
    for i in range(len(segments)): 
        actions.insert(2*i, action.Playback(localAudioFiles[i], segments[i][4].start, sum([segments[i][4+j].duration for j in range(len(segments[i])-8)])))
    for i in range(len(actions)/2):
        action.render([actions[2*i],actions[2*i+1]], str(i) + " " + filenames[i])                       
    action.render([actions[-1]], str(len(filenames)-1) + " " + filenames[-1])
    #action.render(actions, "totalCrossmatched.mp3")
       
def generateHipHopSlam(localAudioFiles, bars, filenames):
    actions = [action.Playback(localAudioFiles[i], bars[i][0].start, sum(bars[i][j].duration for j in range(len(bars[i])))) for i in range(len(bars))]
    for i in range(len(actions)): action.render([actions[i]],str(i) + " " + filenames[i])
    #action.render(actions, "totalHipHopSlam.mp3")
    
       
def generateBeatBarGraphs(segments, bars, filenames):
    for i in range(len(segments)):
        pyplot.figure(i)
        pyplot.subplot(221)
        pyplot.plot([j.duration for j in segments[i]])
        pyplot.xlabel('Beat Number')
        pyplot.ylabel('Duration')
        pyplot.title(filenames[i])
        pyplot.subplot(222)
        pyplot.plot([j.duration for j in bars[i]])
        pyplot.xlabel('Bar Number')
        pyplot.ylabel('Duration')
        pyplot.subplot(223)
        pyplot.plot([j.confidence for j in segments[i]])
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
    localAudioFiles, filenames = getInput()
    equalize_tracks(localAudioFiles)
    runTrainingSet(localAudioFiles, [f.analysis.segments for f in localAudioFiles], filenames)
    #generateBeatBarGraphs([f.analysis.segments for f in localAudioFiles],[f.analysis.bars for f in localAudioFiles],filenames)
    #localAudioFiles, newFilenames, segments, bars = modify(sections, localAudioFiles, filenames)
    #generateHipHopSlam(localAudioFiles, bars, newFilenames)
    deleteOldSongs(filenames)
    os.system('automator /Users/jordanhawkins/Documents/workspace/Automatic\ DJ/import.workflow/')      
    
if __name__ == '__main__':
    main()

    