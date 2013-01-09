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
import sys
import plistlib
import shutil
import urllib
import numpy.matlib as matlib
import numpy as np
import scipy.signal as signal
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
MIX_LENGTH = 5 # defines the length, in beats, of fades between songs
"""
Remove any old audio files from the project directory.
"""
def flushDirectory():
    for filename in os.listdir(workingDirectory):
        if programFiles.count(filename) == 0:
            os.remove(filename)
  
"""
Copy audio files listed in the "Automatic DJ Input" iTunes playlist into the project directory.
"""                 
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


def oldfindLoudestRegion(segments,tempos):
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
        """ the following are 'loc' integers are indices of the middle of the loudest 128 beat region
        of the song, the loudest 64 best region, etc. segmentMarkers is returned as a list of 'loc'
        integer indeces within their respective audio segment lists"""
        loc128 = lpf128.argmax()
        loc64 = lpf64.argmax()
        loc32 = lpf32.argmax()
        loc16 = lpf16.argmax()
        if(loc64 >= loc128-w32 and loc64 <= loc128+w32):
            """ so loc64 is within loc128..."""
            if(loc32 >= loc128-w48 and loc32 <= loc128+w48):
                """...and loc32 is within loc128"""
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
        loc = loc128-w64,loc128+w64 #this is to try...
        segmentMarkers.append(loc)
    return segmentMarkers
    
"""
Find the longest consistently loud region of the song.
window is the approximate number of segments in a 16 beat span.
lpf is a 16-beat-long rectangular window convolved with the loudness values to smoothen them.
the while loop tries to find a loud region of the song that's at least 60 seconds long. If such a region cannot
    be found the first time, the LOUDNESS_FLOOR value is increased to tolerate slightly softer loud regions for
    the sake of a longer song duration.
*To better understand the mathematics involved, note that loudness is measured in negative decibels, 
    so a small negative number is louder than a large negative number.
"""
def findLoudestRegion(segments,tempos):
    segmentMarkers = []
    for segs,tempo in zip(segments,tempos):
        LOUDNESS_CEILING = .8
        LOUDNESS_FLOOR = 1.2
        window = int((16.0/tempo)*60.0/(matlib.mean(matlib.array(segs.durations))))
        lpf = np.convolve(segs.loudness_max,np.ones(window)/window)[window/2:-(window/2)]
        lpf[0:window/2] = lpf[window/2]
        lpf[-(window/2):] = lpf[-(window/2)]
        mean = matlib.mean(matlib.array(lpf))
        marker1 = 0
        finalMarkers = (0,0)
        foundFirstMarker = 0
        while((sum([s.duration for s in segs[finalMarkers[0]:finalMarkers[1]]]) < 60.0) and LOUDNESS_FLOOR < 2.0):
            for i,l in enumerate(lpf):
                if(foundFirstMarker):
                    if l < mean*LOUDNESS_FLOOR or i == len(lpf)-1:
                        foundFirstMarker = 0
                        if((i-marker1) > (finalMarkers[1]-finalMarkers[0])):
                            finalMarkers = (marker1,i)                         
                elif l > mean*LOUDNESS_CEILING: 
                    foundFirstMarker = 1
                    marker1 = i
            # lower the loudness floor and ceiling to allow for a longer region to be chosen, if necessary
            LOUDNESS_FLOOR = LOUDNESS_FLOOR + .05
            LOUDNESS_CEILING = LOUDNESS_CEILING + .05
        segmentMarkers.append(finalMarkers)
    return segmentMarkers         
    
"""
This method was used during development to visualize the data.
timeMarkers contains a tuple of start and end values (in seconds) for each song in my training set.
"""            
def generateSegmentGraphs(segments, filenames, segmentMarkers, tempos):
    # training set timeMarkers = [(26.516,131.312),(4.746,172.450),(41.044,201.012),(82.312,175.997),(15.370,46.003),(122.042,213.469),(30.887,122.294),(0.000,272.304),(37.785,195.357),(15.230,195.357),(37.539,172.498),(67.721,157.716),(37.282,125.899),(147.876,325.127),(14.775,192.008),(213.437,298.800),(29.553,86.022),(238.297,294.371),(21.150,193.356),(41.625,138.350)]
    timeMarkers = [(4.0,141.0),(25.0,177.0),(17.0,188.0),(16.0,129.0),(17.0,177.0),(15.0,136.0),(87.0,149.0),(98.0,173.0),(106.0,212.0),(0.0,104.0)]
    myMarkers = [(j.index(min(j.that(selection.start_during_range(i[0], i[0]+1.0)))),j.index(min(j.that(selection.start_during_range(i[1], i[1]+10.0))))) for j,i in zip(segments,timeMarkers)]    
    for i in range(len(segments)):
        pyplot.figure(i,(16,9))
        windowLen3 = int((16.0/tempos[i])*60.0/(matlib.mean(matlib.array(segments[i].durations))))
        lpf3 = signal.lfilter(np.ones(windowLen3)/windowLen3,1,segments[i].loudness_max) + signal.lfilter(np.ones(windowLen3)/windowLen3,1,segments[i].loudness_max[::-1])[::-1]
        lpf3 = np.convolve(segments[i].loudness_max,np.ones(windowLen3)/windowLen3)[windowLen3/2:-(windowLen3/2)]
        lpf3[0:windowLen3/2] = lpf3[windowLen3/2]
        lpf3[-(windowLen3/2):] = lpf3[-(windowLen3/2)]
        pyplot.plot(lpf3)
        pyplot.xlabel('Segment Number')
        pyplot.ylabel('Loudness (dB)')
        pyplot.vlines(segmentMarkers[i][0], min(lpf3), max(segments[i].loudness_max), 'g')
        pyplot.vlines(segmentMarkers[i][1], min(lpf3), max(segments[i].loudness_max), 'g')
        pyplot.vlines(myMarkers[i][0], min(lpf3), max(segments[i].loudness_max), 'r')
        pyplot.vlines(myMarkers[i][1], min(lpf3), max(segments[i].loudness_max), 'r')
        pyplot.legend(["Loudness", "Autmatically selected start time: " + str(action.humanize_time(segments[i][segmentMarkers[i][0]].start)), 
                            "Automatically selected end time: " + str(action.humanize_time(segments[i][segmentMarkers[i][1]].start)),
                            "Manually selected start time: " + str(action.humanize_time(timeMarkers[i][0])),
                            "Manually selected end time: " + str(action.humanize_time(timeMarkers[i][1]))])
        pyplot.title(filenames[i])    
    pyplot.show()

"""
Collects the names of the audio files in the project directory,
then creates a list of LocalAudioFile objects, which gets returned
along with key analysis objects.
"""
def getInput():
    filenames = []
    for filename in os.listdir(workingDirectory):
        if programFiles.count(filename) == 0:
            filenames.append(filename)  
    inputList = []
    for filename in filenames:
        try:
            inputList.append((audio.LocalAudioFile(filename).analysis.tempo['value'],audio.LocalAudioFile(filename),filename))
        except: print "Exception in getInput for filename: ", filename
    inputList.sort()
    localAudioFiles = [t[1] for t in inputList]
    return [f.analysis.segments for f in localAudioFiles], [t[2] for t in inputList], [f.analysis.tempo['value'] for f in localAudioFiles], localAudioFiles, [f.analysis.beats for f in localAudioFiles]
"""
I copied this method from capsule_support. It equalizes the volume of the input tracks.
"""
def equalize_tracks(tracks):
    def db_2_volume(loudness):
        return (1.0 - LOUDNESS_THRESH * (LOUDNESS_THRESH - loudness) / 100.0)   
    for track in tracks:
        loudness = track.analysis.loudness
        track.gain = db_2_volume(loudness)

"""
This method deletes the original input songs from the directory so they don't get copied into iTunes.
"""   
def deleteOldSongs(filenames):
    for filename in os.listdir(workingDirectory):
        if filename in filenames:
            os.remove(filename)
            
"""
This method generates 4-beat Crossmatch objects, then renders them attached to the end of Playback objects.
"""
def generateCrossmatch(localAudioFiles, beatMarkers, filenames, beats):
    actions = []
    for i in range(len(beatMarkers)-1): 
        #try:
        cm = action.Crossmatch((localAudioFiles[i], localAudioFiles[i+1]), ([(b.start, b.duration) for b in beats[i][beatMarkers[i][1]- MIX_LENGTH:beatMarkers[i][1]]],[(b.start, b.duration) for b in beats[i+1][beatMarkers[i+1][0]:beatMarkers[i+1][0]+MIX_LENGTH]]))
        actions.append(cm)
        #except: 
            #print "exception at: ", filenames[i]
    for i in range(len(beatMarkers)): 
        startBeat = beats[i][beatMarkers[i][0]+MIX_LENGTH]
        endBeat = beats[i][beatMarkers[i][1]-MIX_LENGTH]
        actions.insert(2*i, action.Playback(localAudioFiles[i], startBeat.start, endBeat.start-startBeat.start))
    action.render([action.Fadein(localAudioFiles[0],beats[0][beatMarkers[0][0]].start,beats[0][beatMarkers[0][0]+MIX_LENGTH].start-beats[0][beatMarkers[0][0]].start)],"000 fade in")
    for i in range(len(actions)/2):
        index = str(i+1)
        while(len(index) < 3): index = "0" + index
        try:
            action.render([actions[2*i],actions[2*i+1]], index + " " + filenames[i])
        except: print filenames[i]                        
    index = str(len(filenames))
    while(len(index) < 3): index = "0" + index
    action.render([actions[-1]], index + " " + filenames[-1])
    action.render([action.Fadeout(localAudioFiles[-1],beats[-1][beatMarkers[-1][1]-MIX_LENGTH].start,beats[-1][beatMarkers[-1][1]].start-beats[-1][beatMarkers[-1][1]-MIX_LENGTH].start)], "999 fade out")

"""
This method finds the closest beat to my designated segments for Crossmatching.
I had some trouble with this method, which is why it is so fragmented. I may resolve
this later.
"""
def oldGetBeatMarkers(loudnessMarkers,segments,ones,beats):
    loudnessMarkers = [(segment[marker[0]].start,segment[marker[1]].start) for segment,marker in zip(segments,loudnessMarkers)]
    starts = [[abs(o.start - lm[0]) for o in beat] for beat,lm in zip(ones,loudnessMarkers)]
    ends = [[abs(o.start - lm[1]) for o in beat] for beat,lm in zip(ones,loudnessMarkers)]
    si = [min(s) for s in starts]
    ei = [min(e) for e in ends]
    startIndices = [start.index(s) for start,s in zip(starts,si)]
    endIndices = [end.index(e) for end,e in zip(ends,ei)]
    closestStarts = [beat[s] for beat,s in zip(ones,startIndices)]
    closestEnds = [beat[e] for beat,e in zip(ones,endIndices)]
    return [(cs,ce) for cs,ce in zip(closestStarts,closestEnds)]
    
def getBeatMarkers(loudnessMarkers,segments,beats):
    return [(b.index(b.that(selection.overlap(segments[i][loudnessMarkers[i][0]]))[0]),b.index(b.that(selection.overlap(segments[i][loudnessMarkers[i][1]]))[0]))
            for i,b in enumerate(beats)]
        
"""
This method probably won't be used as much as Crossmatch.
It immediately transitions between songs, without any kind of mix.
"""   
def generateFaderSlam(localAudioFiles, beatMarkers, filenames):
    actions = [action.Playback(laf, b[0].start, (b[1].start-b[0].start)) for laf,b in zip(localAudioFiles,beatMarkers)]
    for i in range(len(actions)): 
        action.render([actions[i]],str(i) + " " + filenames[i])
         
def main():
    """
    segments = cPickle.load(open('valSegs.pkl'))
    tempos = cPickle.load(open('valTempos.pkl'))
    filenames = cPickle.load(open('valNames.pkl'))
    localAudioFiles = cPickle.load(open('valLocalAudioFiles.pkl'))
    beats = [l.analysis.beats for l in localAudioFiles]
    """
    flushDirectory()
    getAudioFiles()
    segments, filenames, tempos, localAudioFiles, beats = getInput()
    equalize_tracks(localAudioFiles)
    loudnessMarkers = findLoudestRegion(segments,tempos)
    beatMarkers = getBeatMarkers(loudnessMarkers,segments,beats)
    #generateSegmentGraphs(segments,filenames,loudnessMarkers,tempos)
    #generateFaderSlam(localAudioFiles,beatMarkers,filenames)
    generateCrossmatch(localAudioFiles,beatMarkers,filenames,beats)
    deleteOldSongs(filenames)
    os.system('automator /Users/jordanhawkins/Documents/workspace/Automatic\ DJ/import.workflow/')      
    
if __name__ == '__main__':
    main()

    