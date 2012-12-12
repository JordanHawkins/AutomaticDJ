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
import numpy.matlib as matlib
import numpy as np
import scipy.signal as signal
import scipy.stats as stats
import sys
import matplotlib.pyplot as pyplot
import cPickle as cPickle

workingDirectory = '/Users/jordanhawkins/Documents/workspace/Automatic DJ/src/root/nested'
programFiles = ['__init__.py','Main.py','AutoMashUp.py','LocalAudioFiles.pkl','filenames.pkl', 'segments.pkl', 'tempos.pkl', 'valNames.pkl', 'valSegs.pkl', 'valTempos.pkl', 'valBeats.pkl', 'valLocalAudioFiles.pkl']
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

def findLoudestRegion(segments,tempos):
    segmentMarkers = []
    for segs,tempo in zip(segments,tempos):
        w128 = int((128.0/tempo)*60.0/(matlib.mean(matlib.array(segs.durations))))
        w64 = int((64.0/tempo)*60.0/(matlib.mean(matlib.array(segs.durations))))
        w48 = int((48.0/tempo)*60.0/(matlib.mean(matlib.array(segs.durations))))
        w32 = int((32.0/tempo)*60.0/(matlib.mean(matlib.array(segs.durations))))
        w16 = int((16.0/tempo)*60.0/(matlib.mean(matlib.array(segs.durations))))
        w8 = int((8.0/tempo)*60.0/(matlib.mean(matlib.array(segs.durations))))
        w4 = int((4.0/tempo)*60.0/(matlib.mean(matlib.array(segs.durations))))
        lpf128 = signal.lfilter(np.ones(w128)/w128,1,segs.loudness_max) + signal.lfilter(np.ones(w128)/w128,1,segs.loudness_max[::-1])[::-1]
        lpf64 = signal.lfilter(np.ones(w64)/w64,1,segs.loudness_max) + signal.lfilter(np.ones(w64)/w64,1,segs.loudness_max[::-1])[::-1]
        lpf32 = signal.lfilter(np.ones(w32)/w32,1,segs.loudness_max) + signal.lfilter(np.ones(w32)/w32,1,segs.loudness_max[::-1])[::-1]
        lpf16 = signal.lfilter(np.ones(w16)/w16,1,segs.loudness_max) + signal.lfilter(np.ones(w16)/w16,1,segs.loudness_max[::-1])[::-1]
        lpf4 = signal.lfilter(np.ones(w4)/w4,1,segs.loudness_max) + signal.lfilter(np.ones(w4)/w4,1,segs.loudness_max[::-1])[::-1]
        lpf128[:(w128-1)] = min(lpf128)
        lpf128[:(-1*w128-1):-1] = min(lpf128)
        lpf64[:(w64-1)] = min(lpf64)
        lpf64[:(-1*w64-1):-1] = min(lpf64)
        lpf32[:(w32-1)] = min(lpf32)
        lpf32[:(-1*w32-1):-1] = min(lpf32)
        lpf16[:(w16-1)] = min(lpf16)
        lpf16[:(-1*w16-1):-1] = min(lpf16)
        lpf4[:(w4-1)] = min(lpf4)
        lpf4[:(-1*w4-1):-1] = min(lpf4)
        loc128 = lpf128.argmax()
        loc64 = lpf64.argmax()
        loc32 = lpf32.argmax()
        loc16 = lpf16.argmax()
        print "len(segs) is: ", len(segs)
        print "loc128 is: ", loc128
        print "loc64 is: ", loc64
        print "loc32 is: ", loc32
        if(loc64 >= loc128-w32 and loc64 <= loc128+w32):
            print "loc64 is within loc128..."
            if(loc32 >= loc128-w48 and loc32 <= loc128+w48):
                print "...and loc32 is within loc128"
                print "mean loudness of 128 beat region is: ", lpf128[loc128]
                print "min 4 beat mean loudness in the 128 beat region is: ", min(lpf4[(loc128-w64):(loc128+w64)])
                loc = loc128-w64,loc128+w64
            else: 
                print "...but loc32 is not within loc128."
                loc = loc32-w16,loc32+w16
        elif(loc32 >= loc64-w16 and loc32 <= loc64+w16): 
            print "loc64 is not within loc128, but loc32 is within loc64."
            print "mean loudness of 64 beat region is: ", lpf64[loc64]
            print "min 4 beat mean loudness in the 64 beat region is: ", min(lpf4[(loc64-w32):(loc64+w32)])
            loc = loc64-w32,loc64+w32
        elif(loc16 >= loc32-w8 and loc16 <= loc32+w8):
            print "loc32 is not within loc64, but loc16 is within loc32."
            loc = loc32-w16,loc32+w16
        else: 
            print "mean loudness of 32 beat region is: ", lpf32[loc32]
            print "min 4 beat mean loudness in the 64 beat region is: ", min(lpf4[(loc32-w16):(loc32+w16)])
            loc = loc32-w16,loc32+w16
        segmentMarkers.append(loc)
    print segmentMarkers
    return segmentMarkers
                
                

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
        windowLen1 = int((4.0/tempos[i])*60.0/(matlib.mean(matlib.array(segments[i].durations))))
        windowLen2 = int((8.0/tempos[i])*60.0/(matlib.mean(matlib.array(segments[i].durations))))
        windowLen3 = int((16.0/tempos[i])*60.0/(matlib.mean(matlib.array(segments[i].durations))))
        windowLen4 = int((32.0/tempos[i])*60.0/(matlib.mean(matlib.array(segments[i].durations))))
        windowLen5 = int((64.0/tempos[i])*60.0/(matlib.mean(matlib.array(segments[i].durations))))
        windowLen6 = int((128.0/tempos[i])*60.0/(matlib.mean(matlib.array(segments[i].durations))))
        #lpf = signal.filtfilt(np.ones(windowLen)/windowLen,np.ones(windowLen),segments[i].loudness_max)
        lpf1 = signal.lfilter(np.ones(windowLen1)/windowLen1,1,segments[i].loudness_max) + signal.lfilter(np.ones(windowLen1)/windowLen1,1,segments[i].loudness_max[::-1])[::-1]
        std1 = [matlib.std(matlib.array([segments[i][k+p].loudness_max for p in range(windowLen1)])) for k in range(len(segments[i])-windowLen1)]
        lpf2 = signal.lfilter(np.ones(windowLen2)/windowLen2,1,segments[i].loudness_max) + signal.lfilter(np.ones(windowLen2)/windowLen2,1,segments[i].loudness_max[::-1])[::-1]
        lpf3 = signal.lfilter(np.ones(windowLen3)/windowLen3,1,segments[i].loudness_max) + signal.lfilter(np.ones(windowLen3)/windowLen3,1,segments[i].loudness_max[::-1])[::-1]
        lpf4 = signal.lfilter(np.ones(windowLen4)/windowLen4,1,segments[i].loudness_max) + signal.lfilter(np.ones(windowLen4)/windowLen4,1,segments[i].loudness_max[::-1])[::-1]
        lpf5 = signal.lfilter(np.ones(windowLen5)/windowLen5,1,segments[i].loudness_max) + signal.lfilter(np.ones(windowLen5)/windowLen5,1,segments[i].loudness_max[::-1])[::-1]
        lpf6 = signal.lfilter(np.ones(windowLen6)/windowLen6,1,segments[i].loudness_max) + signal.lfilter(np.ones(windowLen6)/windowLen6,1,segments[i].loudness_max[::-1])[::-1]
        WINDOW_LENGTH_MEAN = int((32.0/tempos[i])*60.0/(matlib.mean(matlib.array(segments[i].durations))))
        WINDOW_LENGTH_STD = int((8.0/tempos[i])*60.0/(matlib.mean(matlib.array(segments[i].durations))))
        stdev = [matlib.std(matlib.array([segments[i][k+p].loudness_max for p in range(WINDOW_LENGTH_STD)])) for k in range(len(segments[i])-WINDOW_LENGTH_STD)]
        means = [matlib.mean(matlib.array([segments[i][k+p].loudness_max for p in range(WINDOW_LENGTH_MEAN)])) for k in range(len(segments[i])-WINDOW_LENGTH_MEAN)]
        # piecewise multiply means and standard deviations to find segment marker with greatest magnitude
        combined = [sd*mean for sd,mean in zip(stdev,means)]
        pyplot.plot(segments[i].loudness_max)
        pyplot.plot(lpf1-10)
        pyplot.plot([std-10 for std in std1])
        #pyplot.plot([lpf-10 for lpf in lpf12])
        pyplot.plot(lpf2-15)
        pyplot.plot(lpf3-20)
        pyplot.plot(lpf4-25)
        lpf5[:63] = min(lpf5)
        lpf5[:-65:-1] = min(lpf5)
        pyplot.plot(lpf5-30)
        lpf6[:123] = min(lpf6)
        lpf6[:-125:-1] = min(lpf6)
        pyplot.plot(lpf6-35)
        pyplot.xlabel('Segment Number')
        pyplot.ylabel('Loudness (dB)')
        """
        pyplot.vlines(means.index(max(means)), min(segments[i].loudness_max), max(segments[i].loudness_max), 'r')
        pyplot.vlines(means.index(max(means))+WINDOW_LENGTH_MEAN, min(segments[i].loudness_max), max(segments[i].loudness_max), 'r')
        pyplot.vlines(stdev.index(max(stdev)), min(segments[i].loudness_max), max(segments[i].loudness_max), 'b')
        pyplot.vlines(stdev.index(max(stdev))+WINDOW_LENGTH_STD, min(segments[i].loudness_max), max(segments[i].loudness_max), 'b')
        pyplot.vlines(combined.index(max(combined)), min(segments[i].loudness_max), max(segments[i].loudness_max), 'g')
        """
        #pyplot.vlines(lpf5.argmax(),min(lpf5)-30,max(segments[i].loudness_max))
        #pyplot.vlines(lpf5.argmax()-windowLen5/2,min(lpf5)-30,max(segments[i].loudness_max))
        #pyplot.vlines(lpf5.argmax()+windowLen5/2,min(lpf5)-30,max(segments[i].loudness_max))
        pyplot.vlines(segmentMarkers[i][0], min(lpf5)-30, max(segments[i].loudness_max))
        pyplot.vlines(segmentMarkers[i][1], min(lpf5)-30, max(segments[i].loudness_max))
        print "filename: ", filenames[i]
        print "total duration: ", action.humanize_time(sum(segments[i].durations))
        print "start location: ", action.humanize_time(segments[i][combined.index(max(combined))].start)
        pyplot.title(filenames[i])    
    pyplot.show()

def generateTatumsGraphsConfidence(tatums, filenames, segmentMarkers, tempos):
    print "got into confidence"
    for i in range(len(tatums)):
        pyplot.figure(i,(16,9))
        def printLoudnessStats():
            print filenames[i]
            print "mean loudness of song: ", matlib.mean(matlib.array([h.loudness_max for h in tatums[i]]))
            print "mean loudness of designated region: ", matlib.mean(matlib.array([h.loudness_max for h in tatums[i][segmentMarkers[i][0]:segmentMarkers[i][1]]]))
            print "region/song mean loudness ratio: ", matlib.mean(matlib.array([h.loudness_max for h in tatums[i][segmentMarkers[i][0]:segmentMarkers[i][1]]]))/matlib.mean(matlib.array([h.loudness_max for h in tatums[i]]))
            print "standard deviation of song: ", matlib.std(matlib.array([h.loudness_max for h in tatums[i]]))
            print "standard deviation of designated region: ", matlib.std(matlib.array([h.loudness_max for h in tatums[i][segmentMarkers[i][0]:segmentMarkers[i][1]]]))
            print "region/song std loudness ratio: ", matlib.std(matlib.array([h.loudness_max for h in tatums[i][segmentMarkers[i][0]:segmentMarkers[i][1]]]))/matlib.std(matlib.array([h.loudness_max for h in tatums[i]]))
            print " "
        # window through all segment loudness values to find loudest region...
        windowLen1 = int((4.0/tempos[i])*60.0/(matlib.mean(matlib.array(tatums[i].durations))))
        windowLen2 = int((8.0/tempos[i])*60.0/(matlib.mean(matlib.array(tatums[i].durations))))
        windowLen3 = int((16.0/tempos[i])*60.0/(matlib.mean(matlib.array(tatums[i].durations))))
        windowLen4 = int((32.0/tempos[i])*60.0/(matlib.mean(matlib.array(tatums[i].durations))))
        windowLen5 = int((64.0/tempos[i])*60.0/(matlib.mean(matlib.array(tatums[i].durations))))
        windowLen6 = int((128.0/tempos[i])*60.0/(matlib.mean(matlib.array(tatums[i].durations))))
        #lpf = signal.filtfilt(np.ones(windowLen)/windowLen,np.ones(windowLen),tatums[i].confidence)
        print "just before lpf1"
        lpf1 = signal.lfilter(np.ones(windowLen1)/windowLen1,1,tatums[i].confidence) + signal.lfilter(np.ones(windowLen1)/windowLen1,1,tatums[i].confidence[::-1])[::-1]
        print lpf1
        std1 = [matlib.std(matlib.array([tatums[i][k+p].confidence for p in range(windowLen1)])) for k in range(len(tatums[i])-windowLen1)]
        lpf2 = signal.lfilter(np.ones(windowLen2)/windowLen2,1,tatums[i].confidence) + signal.lfilter(np.ones(windowLen2)/windowLen2,1,tatums[i].confidence[::-1])[::-1]
        lpf3 = signal.lfilter(np.ones(windowLen3)/windowLen3,1,tatums[i].confidence) + signal.lfilter(np.ones(windowLen3)/windowLen3,1,tatums[i].confidence[::-1])[::-1]
        lpf4 = signal.lfilter(np.ones(windowLen4)/windowLen4,1,tatums[i].confidence) + signal.lfilter(np.ones(windowLen4)/windowLen4,1,tatums[i].confidence[::-1])[::-1]
        lpf5 = signal.lfilter(np.ones(windowLen5)/windowLen5,1,tatums[i].confidence) + signal.lfilter(np.ones(windowLen5)/windowLen5,1,tatums[i].confidence[::-1])[::-1]
        lpf6 = signal.lfilter(np.ones(windowLen6)/windowLen6,1,tatums[i].confidence) + signal.lfilter(np.ones(windowLen6)/windowLen6,1,tatums[i].confidence[::-1])[::-1]        
        WINDOW_LENGTH_MEAN = int((32.0/tempos[i])*60.0/(matlib.mean(matlib.array(tatums[i].durations))))
        WINDOW_LENGTH_STD = int((8.0/tempos[i])*60.0/(matlib.mean(matlib.array(tatums[i].durations))))
        stdev = [matlib.std(matlib.array([tatums[i][k+p].confidence for p in range(WINDOW_LENGTH_STD)])) for k in range(len(tatums[i])-WINDOW_LENGTH_STD)]
        means = [matlib.mean(matlib.array([tatums[i][k+p].confidence for p in range(WINDOW_LENGTH_MEAN)])) for k in range(len(tatums[i])-WINDOW_LENGTH_MEAN)]
        # piecewise multiply means and standard deviations to find segment marker with greatest magnitude
        combined = [sd*mean for sd,mean in zip(stdev,means)]
        pyplot.plot(tatums[i].confidence)
        pyplot.plot(lpf1-1)
        pyplot.plot([std-10 for std in std1])
        #pyplot.plot([lpf-10 for lpf in lpf12])
        pyplot.plot(lpf2-2)
        pyplot.plot(lpf3-3)
        pyplot.plot(lpf4-4)
        lpf5[:63] = min(lpf5)
        lpf5[:-65:-1] = min(lpf5)
        pyplot.plot(lpf5-5)
        lpf6[:123] = min(lpf6)
        lpf6[:-125:-1] = min(lpf6)
        pyplot.plot(lpf6-6)
        pyplot.xlabel('Segment Number')
        pyplot.ylabel('Loudness (dB)')
        """
        pyplot.vlines(means.index(max(means)), min(tatums[i].loudness_max), max(tatums[i].loudness_max), 'r')
        pyplot.vlines(means.index(max(means))+WINDOW_LENGTH_MEAN, min(tatums[i].loudness_max), max(tatums[i].loudness_max), 'r')
        pyplot.vlines(stdev.index(max(stdev)), min(tatums[i].loudness_max), max(tatums[i].loudness_max), 'b')
        pyplot.vlines(stdev.index(max(stdev))+WINDOW_LENGTH_STD, min(tatums[i].loudness_max), max(tatums[i].loudness_max), 'b')
        pyplot.vlines(combined.index(max(combined)), min(tatums[i].loudness_max), max(tatums[i].loudness_max), 'g')
        """
        #pyplot.vlines(lpf5.argmax(),min(lpf5)-30,max(tatums[i].loudness_max))
        #pyplot.vlines(lpf5.argmax()-windowLen5/2,min(lpf5)-30,max(tatums[i].loudness_max))
        #pyplot.vlines(lpf5.argmax()+windowLen5/2,min(lpf5)-30,max(tatums[i].loudness_max))
        pyplot.vlines(segmentMarkers[i][0], min(lpf5)-30, max(tatums[i].confidence))
        pyplot.vlines(segmentMarkers[i][1], min(lpf5)-30, max(tatums[i].confidence))
        print "filename: ", filenames[i]
        print "total duration: ", action.humanize_time(sum(tatums[i].durations))
        print "start location: ", action.humanize_time(tatums[i][combined.index(max(combined))].start)
        pyplot.title(filenames[i])    
    pyplot.show()

def generateChromaGraphs(segments, filenames, segmentMarkers, tempos):
    pitches = [s.pitches for s in segments]
    for i in range(len(pitches)):
        pyplot.figure(i,(16,9))
        chords = [p.index(max(p)) for p in pitches[i]]
        pyplot.plot(chords)
        windowLen1 = int((4.0/tempos[i])*60.0/(matlib.mean(matlib.array(segments[i].durations))))
        modes = [stats.mode(matlib.array([chords[k+p] for p in range(windowLen1)])) for k in range(len(chords)-windowLen1)]
        print modes
        pyplot.plot(modes)
        pyplot.xlabel('Segment Number')
        pyplot.ylabel('Loudness (dB)')
        pyplot.title(filenames[i])    
    pyplot.show()

def getInput():
    filenames = []
    for filename in os.listdir(workingDirectory):
        if programFiles.count(filename) == 0:
            filenames.append(filename)
    if 0: 
        localAudioFiles = cPickle.load(open('valLocalAudioFiles.pkl'))
        ones = cPickle.load(open('valBeats.pkl'))
        filenames = cPickle.load(open('valNames.pkl'))
        segments = cPickle.load(open('valSegs.pkl'))
        tempos = cPickle.load(open('valTempos.pkl'))
        return segments, filenames, tempos, ones, localAudioFiles, [f.analysis.beats for f in localAudioFiles] 
    inputList = []
    for i in range(len(filenames)):
        audiofile = audio.LocalAudioFile(filenames[i])
        inputList.append((audiofile.analysis.tempo['value'], audiofile, filenames[i]))
    inputList.sort()
    localAudioFiles = [t[1] for t in inputList]
    filenames = [t[2] for t in inputList]
    """
    cPickle.dump([t[2] for t in inputList],open('valNames.pkl','wb'))
    cPickle.dump([t[1].analysis.tempo['value'] for t in inputList],open('valTempos.pkl','wb'))
    cPickle.dump([t[1].analysis.segments for t in inputList],open('valSegs.pkl','wb'))
    try:
        os.remove('filenames.pkl')
    except: print "No file named filenames.pkl..."
    try:    
        os.remove('LocalAudioFiles.pkl')
    except: print "No file names LocalAudioFiles.pkl..."
    cPickle.dump(filenames,open('filenames.pkl','wb'))
    cPickle.dump([f.analysis.tempo['value'] for f in localAudioFiles],open('tempos.pkl','wb'))
    cPickle.dump(localAudioFiles,open('LocalAudioFiles.pkl','wb'))
    """
    return [f.analysis.segments for f in localAudioFiles], filenames, [f.analysis.tempo['value'] for f in localAudioFiles],[f.analysis.beats.that(selection.fall_on_the(1)) for f in localAudioFiles], localAudioFiles, [f.analysis.beats for f in localAudioFiles]
 
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

def getBeatMarkers(loudnessMarkers,segments,ones,beats):
    print "loudnessMarkers is: ", loudnessMarkers
    loudnessMarkers = [(segment[marker[0]].start,segment[marker[1]].start) for segment,marker in zip(segments,loudnessMarkers)]
    print [o.start for o in ones[0]]
    starts = [[abs(o.start - lm[0]) for o in beat] for beat,lm in zip(ones,loudnessMarkers)]
    ends = [[abs(o.start - lm[1]) for o in beat] for beat,lm in zip(ones,loudnessMarkers)]
    print "starts[0] is: ", starts[0]
    si = [min(s) for s in starts]
    ei = [min(e) for e in ends]
    startIndeces = [start.index(s) for start,s in zip(starts,si)] 
    endIndeces = [end.index(e) for end,e in zip(ends,ei)]
    print [len(beat) for beat in ones]
    print [ei-si for ei,si in zip(endIndeces,startIndeces)]
    closestStarts = [beat[s] for beat,s in zip(ones,startIndeces)]
    closestEnds = [beat[e] for beat,e in zip(ones,endIndeces)]
    print "beats: ", beats
    print "len(beats): ", len(beats)
    for i in range(len(beats)): 
        try:
            print "length of beats mod 4: ", (beats[i].index(closestEnds[i])-beats[i].index(closestStarts[i]))%4
        except: print "problem" 
    return [(cs,ce) for cs,ce in zip(closestStarts,closestEnds)]
       
def generateHipHopSlam(localAudioFiles, beatMarkers, filenames):
    actions = [action.Playback(laf, b[0].start, (b[1].start+b[1].duration-b[0].start)) for laf,b in zip(localAudioFiles,beatMarkers)]
    for i in range(len(actions)): 
        action.render([actions[i]],str(i) + " " + filenames[i])
         
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
    segments, filenames, tempos, ones, localAudioFiles, beats = getInput()
    equalize_tracks(localAudioFiles)
    loudnessMarkers = findLoudestRegion(segments,tempos)
    beatMarkers = getBeatMarkers(loudnessMarkers,segments,ones,beats)
    generateChromaGraphs(segments,filenames,loudnessMarkers,tempos)
    generateTatumsGraphsConfidence([f.analysis.tatums for f in localAudioFiles],filenames,loudnessMarkers,tempos)
    generateSegmentGraphs(segments,filenames,loudnessMarkers,tempos)
    generateHipHopSlam(localAudioFiles,beatMarkers,filenames)
    #runTrainingSet(segments, filenames, tempos)
    #generateBeatBarGraphs([f.analysis.segments for f in localAudioFiles],[f.analysis.bars for f in localAudioFiles],filenames)
    #localAudioFiles, newFilenames, segments, bars = modify(sections, localAudioFiles, filenames)
    #generateHipHopSlam(localAudioFiles, bars, newFilenames)
    deleteOldSongs(filenames)
    os.system('automator /Users/jordanhawkins/Documents/workspace/Automatic\ DJ/import.workflow/')      
    
if __name__ == '__main__':
    main()

    