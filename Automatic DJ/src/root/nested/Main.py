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
import cPickle as cPickle

workingDirectory = '/Users/jordanhawkins/Documents/workspace/Automatic DJ/src/root/nested'
programFiles = ['__init__.py','Main.py','AutoMashUp.py','LocalAudioFiles.pkl','filenames.pkl', 'segments.pkl', 'tempos.pkl', 'valNames.pkl', 'valSegs.pkl', 'valTempos.pkl']
lib = plistlib.readPlist('/Users/jordanhawkins/Music/iTunes/iTunes Music Library.xml')
BEAT_CONFIDENCE = .9
BAR_CONFIDENCE = .8
LOUDNESS_THRESH = -8 # per capsule_support module
WINDOW_LENGTH_MEAN = 81 # default length of windows for finding mean loudness regions
WINDOW_LENGTH_STD = 9 # default length of windows for finding standard deviations that demarcate song sections

def flushDirectory():
    for filename in os.listdir(workingDirectory):
        if programFiles.count(filename) == 0:
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
                    print "exception in getAudioFiles"
            break

def generateSegmentGraphs(segments, filenames, segmentMarkers, tempos):
    for i in range(len(segments)):
        pyplot.figure(i,(16,9))
        def printLoudnessStats():
            print filenames[i]
            print "mean loudness of song: ", matlib.mean(matlib.array([h.loudness_max for h in segments[i]]))
            print "mean loudness of designated region: ", matlib.mean(matlib.array([h.loudness_max for h in segments[i][segmentMarkers[i][0]:segmentMarkers[i][1]]]))
            print "region/song mean loudness ratio: ", matlib.mean(matlib.array([h.loudness_max for h in segments[i][segmentMarkers[i][0]:segmentMarkers[i][1]]]))/matlib.mean(matlib.array([h.loudness_max for h in segments[i]]))
            print "standard deviation of song: ", matlib.std(matlib.array([h.loudness_max for h in segments[i]]))
            print "standard deviation of designated region: ", matlib.std(matlib.array([h.loudness_max for h in segments[i][segmentMarkers[i][0]:segmentMarkers[i][1]]]))
            print "region/song std loudness ratio: ", matlib.std(matlib.array([h.loudness_max for h in segments[i][segmentMarkers[i][0]:segmentMarkers[i][1]]]))/matlib.std(matlib.array([h.loudness_max for h in segments[i]]))
            print " "
        # window through all segment loudness values to find loudest region...
        #WINDOW_LENGTH_MEAN = 32/int(tempos[i])*60
        WINDOW_LENGTH_MEAN = int((32.0/tempos[i])*60.0/(matlib.mean(matlib.array(segments[i].durations))))
        WINDOW_LENGTH_STD = int((8.0/tempos[i])*60.0/(matlib.mean(matlib.array(segments[i].durations))))
        stdev = [matlib.std(matlib.array([segments[i][k+p].loudness_max for p in range(WINDOW_LENGTH_STD)])) for k in range(len(segments[i])-WINDOW_LENGTH_STD)]
        means = [matlib.mean(matlib.array([segments[i][k+p].loudness_max for p in range(WINDOW_LENGTH_MEAN)])) for k in range(len(segments[i])-WINDOW_LENGTH_MEAN)]
        # piecewise multiply means and standard deviations to find segment marker with greatest magnitude
        combined = [sd*mean for sd,mean in zip(stdev,means)]
        pyplot.plot(segments[i].loudness_max)
        pyplot.xlabel('Segment Number')
        pyplot.ylabel('Loudness (dB)')
        #pyplot.vlines(segmentMarkers[i][0], min(segments[i].loudness_max), max(segments[i].loudness_max))
        #pyplot.vlines(segmentMarkers[i][1], min(segments[i].loudness_max), max(segments[i].loudness_max))
        pyplot.vlines(means.index(max(means)), min(segments[i].loudness_max), max(segments[i].loudness_max), 'r')
        pyplot.vlines(means.index(max(means))+WINDOW_LENGTH_MEAN, min(segments[i].loudness_max), max(segments[i].loudness_max), 'r')
        pyplot.vlines(stdev.index(max(stdev)), min(segments[i].loudness_max), max(segments[i].loudness_max), 'b')
        pyplot.vlines(stdev.index(max(stdev))+WINDOW_LENGTH_STD, min(segments[i].loudness_max), max(segments[i].loudness_max), 'b')
        pyplot.vlines(combined.index(max(combined)), min(segments[i].loudness_max), max(segments[i].loudness_max), 'g')
        print "filename: ", filenames[i]
        print "total duration: ", action.humanize_time(sum(segments[i].durations))
        print "start location: ", action.humanize_time(segments[i][combined.index(max(combined))].start)
        pyplot.title(filenames[i])    
    pyplot.show()

def getInput():
    filenames = []
    for filename in os.listdir(workingDirectory):
        if programFiles.count(filename) == 0:
            filenames.append(filename)
    try:
        #fileTest = filenames.sort() == cPickle.load(open('filenames.pkl')).sort()
        fileTest = 1
    except: fileTest = 0
    if fileTest: 
        #lafs = cPickle.load(open('LocalAudioFiles.pkl'))
        filenames = cPickle.load(open('valNames.pkl'))
        segments = cPickle.load(open('valSegs.pkl'))
        tempos = cPickle.load(open('valTempos.pkl'))
        return segments, filenames, tempos
    inputList = []
    for i in range(len(filenames)):
        audiofile = audio.LocalAudioFile(filenames[i])
        inputList.append((audiofile.analysis.tempo['value'], audiofile, filenames[i]))
    inputList.sort()
    cPickle.dump([t[2] for t in inputList],open('valNames.pkl','wb'))
    cPickle.dump([t[1].analysis.tempo['value'] for t in inputList],open('valTempos.pkl','wb'))
    cPickle.dump([t[1].analysis.segments for t in inputList],open('valSegs.pkl','wb'))
    return [t[1].analysis.segments for t in inputList],[t[2] for t in inputList],[t[1].analysis.tempo['value'] for t in inputList]
    localAudioFiles = [t[1] for t in inputList]
    filenames = [t[2] for t in inputList]
    try:
        os.remove('filenames.pkl')
    except: print "No file named filenames.pkl..."
    try:    
        os.remove('LocalAudioFiles.pkl')
    except: print "No file names LocalAudioFiles.pkl..."
    cPickle.dump(filenames,open('filenames.pkl','wb'))
    cPickle.dump([f.analysis.tempo['value'] for f in localAudioFiles],open('tempos.pkl','wb'))
    cPickle.dump(localAudioFiles,open('LocalAudioFiles.pkl','wb'))
    return localAudioFiles, filenames, [f.analysis.tempo['value'] for f in localAudioFiles]
 
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

def runTrainingSet(segments, filenames, tempos):
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
    generateSegmentGraphs(segments,filenames,segmentMarkers, tempos)
    sys.exit()
    # Now check and see if it spits out similar songs to mine...
    out = []
    for marker,segment in zip(segmentMarkers, segments): 
        if marker[0] == -1: marker = (0,marker[1])
        segment = segment[marker[0]:marker[1]]
        out.append(segment)
    segments = out
    #actions = [action.Playback(localAudioFiles[i], segments[i][0].start, sum(segments[i][j].duration for j in range(len(segments[i])))) for i in range(len(segments))]
    #for i in range(len(actions)): action.render([actions[i]],str(i) + " " + filenames[i])
                   
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
    segments, filenames, tempos = getInput()
    #equalize_tracks(localAudioFiles)
    generateSegmentGraphs(segments,filenames,0,tempos)
    sys.exit()
    runTrainingSet(segments, filenames, tempos)
    #generateBeatBarGraphs([f.analysis.segments for f in localAudioFiles],[f.analysis.bars for f in localAudioFiles],filenames)
    #localAudioFiles, newFilenames, segments, bars = modify(sections, localAudioFiles, filenames)
    #generateHipHopSlam(localAudioFiles, bars, newFilenames)
    deleteOldSongs(filenames)
    os.system('automator /Users/jordanhawkins/Documents/workspace/Automatic\ DJ/import.workflow/')      
    
if __name__ == '__main__':
    main()

    