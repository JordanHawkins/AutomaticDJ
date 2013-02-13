#!/usr/bin/env/python
# encoding: utf=8
'''
Created on Oct 15, 2012
@author: jordanhawkins
'''
import echonest.audio as audio
import echonest.modify as modify
import echonest.selection as selection
import echonest.action as action
import os
import plistlib
import shutil
import urllib
import numpy.matlib as matlib
import numpy.linalg as linalg
import numpy as numpy
import sys
import matplotlib.pyplot as pyplot
import matplotlib.image as image
import scipy.ndimage as ndimage
import cPickle as cPickle
from collections import deque
import skimage.filter as filter
import skimage.feature as feature

programFiles = ['__init__.py','Main.py','AutoMashUp.py','LocalAudioFiles.pkl','filenames.pkl', 'segments.pkl', 
                'tempos.pkl', 'valNames.pkl', 'valSegs.pkl', 'valTempos.pkl', 'valBeats.pkl', 'valLocalAudioFiles.pkl',
                'AmuLafs.pkl', "AmuFilenames.pkl", "AmuPitches.pkl", "AmuKeys.pkl", 'AmuInstPitches.pkl', 
                'AmuInstTimbre.pkl', 'AmuInstLoudness.pkl', 'AmuInstSections.pkl', 'AmuInstSimMats.pkl', 'Just_Dance.pkl']
workingDirectory = '/Users/jordanhawkins/Documents/workspace/Automatic DJ/src/root/nested'
lib = plistlib.readPlist('/Users/jordanhawkins/Music/iTunes/iTunes Music Library.xml')
LOUDNESS_THRESH = -8 # per capsule_support module

def flushDirectory():
    for filename in os.listdir(workingDirectory):
        if programFiles.count(filename) == 0:
            os.remove(filename)
                   
def getAudioFiles():
    filenames = []
    for count in range(len(lib['Playlists'])):
        if lib['Playlists'][count]['Name'] == 'Automatic MashUp Input':
            playlistItems = lib['Playlists'][count]['Playlist Items']
            trackIDs = [i['Track ID'] for i in playlistItems]
            for i in range(len(trackIDs)):
                location = lib['Tracks'][str(trackIDs[i])]['Location']
                location = urllib.unquote(location)
                try:
                    shutil.copy(location[16:], workingDirectory)
                    filenames.append(location[16:])
                except:
                    print "exception in getAudioFiles"
            break
    return filenames

def getInput(filenames):
    localAudioFiles = []
    keys = []
    list = []
    for filename in filenames: 
        try:
            laf = audio.LocalAudioFile(filename)
            localAudioFiles.append(laf)
            print filename
            print "key: ", laf.analysis.key['value']
            key = laf.analysis.key['value']
            if(laf.analysis.mode['value'] == 0):
                key = key+3
            keys.append(key)
            print "key confidence: ", laf.analysis.key['confidence']
            print "mode: ", laf.analysis.mode['value']
            print "mode confidence: ", laf.analysis.mode['confidence']
            print "tempo: ", laf.analysis.tempo['value']
            tempo = laf.analysis.tempo['value']
            print "tempo confidence: ", laf.analysis.tempo['confidence']
            print " "
            list.append((key,filename,tempo))
        except: print "problem with filename: ", filename
    list.sort()
    for l in list:
        print "filename: ", l[1]
        print "key: ", l[0]
        print "tempo: ", l[2]
    tempos = [laf.analysis.tempo['value'] for laf in localAudioFiles]
    #cPickle.dump(localAudioFiles, open('AmuLafs.pkl', 'w'))
    #cPickle.dump(filenames, open('AmuLafs.pkl', 'w'))
    return localAudioFiles, filenames, tempos, keys
 
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

def meanPitches(segments): 
    """ 
    Returns a pitch vector that is the mean of the pitch vectors of any segments  
    that overlap this AudioQuantum. 
    Note that this means that some segments will appear in more than one AudioQuantum. 
    """ 
    temp_pitches = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] 
    for segment in segments: 
        for index, pitch in enumerate(segment.pitches): 
            temp_pitches[index] = temp_pitches[index] + pitch 
        mean_pitches = [pitch / len(segments) for pitch in temp_pitches] 
    return mean_pitches 

def meanTimbre(segments): 
    """ 
    Returns a timbre vector that is the mean of the timbre vectors of any segments  
    that overlap this AudioQuantum. 
    Note that this means that some segments will appear in more than one AudioQuantum. 
    """ 
    temp_timbre = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] 
    for segment in segments: 
        for index, timbre in enumerate(segment.timbre): 
            temp_timbre[index] = temp_timbre[index] + timbre 
        mean_timbre = [timbre / len(segments) for timbre in temp_timbre] 
    return mean_timbre 

def matchTempoAndKey(localAudioFiles, tempos, keys):
    keys[2] = keys[0]
    keys[3] = keys[1]
    tempos[2] = tempos[0]
    tempos[3] = tempos[1]
    print "tempos: ", tempos
    midTempo = (max(tempos) + min(tempos))/2.0
    print "midTempo: ", midTempo
    midTempo = round(midTempo)
    print "rounded midTempo: ", midTempo
    mod = modify.Modify()
    out = [mod.shiftTempo(laf, midTempo/tempo) for laf,tempo in zip(localAudioFiles,tempos)]
    if max(keys)-min(keys) > 6:
        midKey = (((12 - max(keys) + min(keys))/2) + max(keys)) % 12 # there are 12 chroma values
    else:
        midKey = (max(keys) + min(keys))/2
    for i in range(len(out)):
        if(abs(midKey-keys[i])>3):
            out[i] = mod.shiftPitchSemiTones(out[i],midKey-keys[i]+12)
        else:
            out[i] = mod.shiftPitchSemiTones(out[i],midKey-keys[i])
    print "midTempo: ", midTempo
    print "midKey: ", midKey
    out[0].encode('0.mp3')
    out[1].encode('1.mp3')
    out[2].encode('2.mp3')
    out[3].encode('3.mp3')

"""
Calculates a matrix of correlation values between
a chroma vector and all other chroma vectors.
"""
def corrMatrix(p1,p2,key1,key2):
    print "p1[0]: ", p1[0]
    temp = []
    for vector in p1:
        deq = deque(vector)
        deq.rotate(key2-key1)
        l = list(deq)
        temp.append(l)
    p1 = temp
    print "p1[0]: ", p1[0]
    corrMatrix = []
    for pitches in p1:
        eucNorm =[linalg.norm(matlib.array(pitches)-matlib.array(p)) for p in p2]
        corrMatrix.append(eucNorm)        
    return corrMatrix

def analyzeChroma():
    """
    selectionMarkers = [[(18,129),(132,340),(452,468)],             # 1. Just Dance
                        [(18,129),(144,352),(352,368)],             # 2. In My Head
                        [(120,152),(182,248),(358,444),(446,632)],  # 3. Paradise
                        [(0,16),(32,64),(60,123),(131,224)],        # 4. Wide Awake               
                        [(80,208),(240,304),(536,600)],             # 5. Paradise
                        [(32,160),(192,256),(32,96)],               # 6. Day n Nite
                        [(16,80),(156,172),(208,364)],              # 7. Good Feeling 
                        [(16,80),(148,164),(208,364)],              # 8. Good Feeling              
                        [(144,208),(284,348),(392,464)],            # 9. Good Feeling
                        [(64,128),(176,240),(240,312)],             # 10. Dynamite           
                        [(64,144),(176,204),(360,392),(440,520)],   # 11. Good Feeling
                        [(16,96),(96,124),(360,392),(440,520)],     # 12. Just Dance
                        [],                                         # 13. Titanium
                        [],                                         # 14. Titanium
                        [(160,192),(352,384),(448,480)],            # 15. Titanium 
                        [(80,112),(112,144),(80,112)],              # 16. Who's That Chick
                        [(256,352),(416,448)],                      # 17. Titanium
                        [(64,160),(64,96)],                         # 18. Disturbia
                        [],[],[],[],[],[],[],[],
                        [(144,176),(272,336),(368,432)],            # 27. Firework 
                        [(80,112),(208,272),(304,368)],             # 28. Paparazzi
                        [(16,80),(176,240),(312,368)],              # 29. Firework
                        [(0,64),(160,224),(296,352)],               # 30. Feel So Close
                        [(77,140),(237,269),(336,464)],             # 31. Firework
                        [(77,140),(237,269),(336,464)],             # 32. Firework
                        [],[],[],[],[],[],
                        [(93,188),(221,316),(349,416)],             # 39. Feel So Close
                        [(77,172),(233,328),(233,300)],             # 40. Last Friday Night
                        [],[],                                      # 41. Feel So Close # 42. Release Me
                        [(0,64),(64,96),(96,160),(160,352),(352,420)],  #43. Feel So Close
                        [(0,64),(64,96),(96,128),(128,320),(322,390)],  #44. Please Don't Go
                        [],[],[],[],[],[]]
    """
    pitches = cPickle.load(open('AmuInstPitches.pkl'))
    loudness = cPickle.load(open('AmuInstLoudness.pkl'))
    sections = cPickle.load(open('AmuInstSections.pkl'))
    filenames = cPickle.load(open('AmuFilenames.pkl'))
    keys = cPickle.load(open('AmuKeys.pkl'))
    for i in range(len(pitches)/2):
        pyplot.figure(4*i,(16,9))
        print "p1 is: ", filenames[2*i]
        print "key: ", keys[2*i]
        print "p2 is: ", filenames[2*i+1]
        print "key: ", keys[2*i+1]
        pcorrMtx = corrMatrix(pitches[2*i], pitches[2*i+1], keys[2*i],keys[2*i+1])
    """
        pyplot.imshow(pcorrMtx, vmin = 0, vmax = 1, cmap = pyplot.get_cmap('gray'), aspect = 'auto', origin = 'lower')
        pyplot.ylabel(os.path.basename(filenames[2*i]) + " (in beats)")
        pyplot.xlabel(os.path.basename(filenames[2*i+1]) + " (in beats)")
        pyplot.title('Pitch Similarity')
        for section in sections[2*i]: pyplot.hlines(section, 0, len(loudness[2*i+1])-1)
        for section in sections[2*i+1]: pyplot.vlines(section, 0, len(loudness[2*i])-1)
        pyplot.figure(4*i+1,(16,9))
        filtpCorrMtx = pcorrMtx
        for t in range(len(pitches[2*i+1])-4):
            for p in range(len(pitches[2*i])-4):
                average = 0.0
                for k in range(4):
                    k = p+k
                    for j in range(4):
                        j = t+j
                        average += numpy.power([pcorrMtx[k][j]],[2])[0]
                filtpCorrMtx[p][t] = average/16.0
        pyplot.imshow(filtpCorrMtx, vmin = 0, vmax = 1, cmap = pyplot.get_cmap('gray'), aspect = 'auto', origin = 'lower')
        pyplot.ylabel(os.path.basename(filenames[2*i]) + " (in beats)")
        pyplot.xlabel(os.path.basename(filenames[2*i+1]) + " (in beats)")
        pyplot.title('Filtered Pitch Similarity')
        for section in sections[2*i]: pyplot.hlines(section, 0, len(loudness[2*i+1])-1)
        for section in sections[2*i+1]: pyplot.vlines(section, 0, len(loudness[2*i])-1)
        pyplot.figure(4*i+2,(16,9))
        pyplot.imshow(ndimage.median_filter(pcorrMtx,4), vmin = 0, vmax = 1, cmap = pyplot.get_cmap('gray'), aspect = 'auto', origin = 'lower')
        pyplot.ylabel(os.path.basename(filenames[2*i]) + " (in beats)")
        pyplot.xlabel(os.path.basename(filenames[2*i+1]) + " (in beats)")
        pyplot.title('Median Filtered Pitch Similarity')
        for section in sections[2*i]: pyplot.hlines(section, 0, len(loudness[2*i+1])-1)
        for section in sections[2*i+1]: pyplot.vlines(section, 0, len(loudness[2*i])-1)
        
        pyplot.figure(4*i+3,(16,9))
        filter.tv_denoise(numpy.array(pcorrMtx,numpy.float64))
        pyplot.imshow(filter.tv_denoise(numpy.array(pcorrMtx,numpy.float64)), vmin = 0, vmax = 1, cmap = pyplot.get_cmap('gray'), aspect = 'auto', origin = 'lower')
        pyplot.ylabel(os.path.basename(filenames[2*i]) + " (in beats)")
        pyplot.xlabel(os.path.basename(filenames[2*i+1]) + " (in beats)")
        pyplot.title('TV_Denoised Pitch Similarity')
        for section in sections[2*i]: pyplot.hlines(section, 0, len(loudness[2*i+1])-1)
        for section in sections[2*i+1]: pyplot.vlines(section, 0, len(loudness[2*i])-1)
        pyplot.show()
        
        pyplot.figure(3*i+1,(16,9))
        tcorrMtx = corrMatrix(timbre[2*i], timbre[2*i+1], keys[2*i],keys[2*i+1])
        pyplot.imshow(tcorrMtx, vmin = 0, vmax = 1, cmap = pyplot.get_cmap('gray'), aspect = 'auto', origin = 'lower')
        pyplot.ylabel(os.path.basename(filenames[2*i]) + " (in beats)")
        pyplot.xlabel(os.path.basename(filenames[2*i+1]) + " (in beats)")
        pyplot.title('Timbre Similarity')
        
        pyplot.figure(3*i+2,(16,9))
        lcorrMtx = []
        for l1 in loudness[2*i]:
            diff =[abs(l1-l2) for l2 in loudness[2*i+1]]
            lcorrMtx.append(diff)
        pyplot.imshow(lcorrMtx, vmin = 0, vmax = 1, cmap = pyplot.get_cmap('gray'), aspect = 'auto', origin = 'lower')
        pyplot.ylabel(os.path.basename(filenames[2*i]) + " (in beats)")
        pyplot.xlabel(os.path.basename(filenames[2*i+1]) + " (in beats)")
        pyplot.title('Loudness Similarity')
        for section in sections[2*i]: pyplot.hlines(section, 0, len(loudness[2*i+1])-1)
        for section in sections[2*i+1]: pyplot.vlines(section, 0, len(loudness[2*i])-1)
        pyplot.figure(3*i+2,(16,9))
        filtlCorrMtx = lcorrMtx
        for t2 in range(len(pitches[2*i+1])-4):
            for p2 in range(len(pitches[2*i])-4):
                average = 0.0
                for k2 in range(4):
                    k2 = p2+k2
                    for j2 in range(4):
                        j2 = t2+j2
                        average += numpy.power([lcorrMtx[k][j]],[2])[0]
                filtlCorrMtx[p2][t2] = average/16.0
        pyplot.imshow(filtlCorrMtx, vmin = 0, vmax = 1, cmap = pyplot.get_cmap('gray'), aspect = 'auto', origin = 'lower')
        pyplot.ylabel(os.path.basename(filenames[2*i]) + " (in beats)")
        pyplot.xlabel(os.path.basename(filenames[2*i+1]) + " (in beats)")
        pyplot.title('Filtered Loudness Similarity')
        for section in sections[2*i]: pyplot.hlines(section, 0, len(loudness[2*i+1])-1)
        for section in sections[2*i+1]: pyplot.vlines(section, 0, len(loudness[2*i])-1)
        pyplot.figure(3*i+2,(16,9))
        combinedMtx = []
        for t3,t4 in zip(pcorrMtx,lcorrMtx):
            combinedVector = [abs(p3-p4) for p3,p4 in zip(t3,t4)]
            combinedMtx.append(combinedVector)
        pyplot.imshow(combinedMtx, vmin = 0, vmax = 1, cmap = pyplot.get_cmap('gray'), aspect = 'auto', origin = 'lower')
        pyplot.ylabel(os.path.basename(filenames[2*i]) + " (in beats)")
        pyplot.xlabel(os.path.basename(filenames[2*i+1]) + " (in beats)")
        pyplot.title('Combined Dynamic and Chromatic Similarity')
        for section in sections[2*i]: pyplot.hlines(section, 0, len(loudness[2*i+1])-1)
        for section in sections[2*i+1]: pyplot.vlines(section, 0, len(loudness[2*i])-1)
        pyplot.show()
        for sels1,sels2 in zip(selectionMarkers[2*i],selectionMarkers[2*i+1]):
            print "sels1: ", sels1
            print "sels2: ", sels2
            pyplot.hlines(sels1[0], sels2[0], sels2[1], 'r')
            pyplot.hlines(sels1[1], sels2[0], sels2[1], 'r')
            pyplot.vlines(sels2[0],sels1[0], sels1[1], 'r')
            pyplot.vlines(sels2[1],sels1[0], sels1[1], 'r')
        """ 
    pyplot.show()
    sys.exit()

def pickleAnalysisData(localAudioFiles):
    beatList = [laf.analysis.beats for laf in localAudioFiles]
    segmentList = [laf.analysis.segments for laf in localAudioFiles]
    beatPitchesList = []
    beatTimbreList = []
    beatLoudnessList = []
    for beats,segments in zip(beatList,segmentList):
        beatPitches = []
        for beat in beats:
            segs = segments.that(selection.overlap(beat))
            beatPitches.append(meanPitches(segs))
        beatPitchesList.append(beatPitches)
    for beats,segments in zip(beatList,segmentList):
        beatTimbre = []
        for beat in beats:
            segs = segments.that(selection.overlap(beat))
            beatTimbre.append(meanTimbre(segs))
        beatTimbreList.append(beatTimbre)
    for beats,segments in zip(beatList,segmentList):
        beatLoudness = []
        for beat in beats:
            segs = segments.that(selection.overlap(beat))
            beatLoudness.append(matlib.mean(segs.loudness_max))
        beatLoudnessList.append(beatLoudness)
    cPickle.dump(beatPitchesList, open('AmuInstPitches.pkl', 'w'))
    cPickle.dump(beatTimbreList, open('AmuInstTimbre.pkl', 'w'))
    cPickle.dump(beatLoudnessList, open('AmuInstLoudness.pkl', 'w'))
    sectionFirstBeats = []
    sections = [laf.analysis.sections for laf in localAudioFiles]
    for sectionsList,bList in zip(sections,beatList): 
        temp = [bList.index(bList.that(selection.overlap(section))[0]) for section in sectionsList]
        sectionFirstBeats.append(temp)
    cPickle.dump(sectionFirstBeats, open('AmuInstSections.pkl', 'w'))
    
def processImages():
    sims = cPickle.load(open('AmuInstSimMats.pkl'))
    for i,sim in enumerate(sims):
        pyplot.figure(0,(16,9))
        pyplot.imshow(sim, vmin = 0, vmax = 1, cmap = pyplot.get_cmap('gray'), aspect = 'auto', origin = 'lower')
        pyplot.title('Unfiltered Sim Matrix ' + str(i))
        pyplot.savefig('Unfiltered Sim Matrix ' + str(i) + '.jpg')
        pyplot.figure(1,(16,9))
        pyplot.imshow(filter.tv_denoise(numpy.array(sim,numpy.float64), weight = 1), vmin = 0, vmax = 1, cmap = pyplot.get_cmap('gray'), aspect = 'auto', origin = 'lower')
        pyplot.title('TV_Denoise ' + str(i))
        pyplot.savefig('TV_Denoise ' + str(i) + '.jpg')
        pyplot.figure(2,(16,9))
        pyplot.imshow(filter.threshold_adaptive(numpy.array(sim,numpy.float64),21), vmin = 0, vmax = 1, cmap = pyplot.get_cmap('gray'), aspect = 'auto', origin = 'lower')
        pyplot.title('Threshold_Adaptive ' + str(i))
        pyplot.savefig('Threshold_Adaptive ' + str(i) + '.jpg')
        pyplot.figure(3,(16,9))
        pyplot.imshow(ndimage.minimum_filter(numpy.array(sim,numpy.float64),size=2), vmin = 0, vmax = 1, cmap = pyplot.get_cmap('gray'), aspect = 'auto', origin = 'lower')
        pyplot.title('Local Minimum_Filter ' + str(i))
        pyplot.savefig('Local Minimum_Filter ' + str(i) + '.jpg')
        pyplot.figure(4,(16,9))
        template = numpy.array([[0,1,1,1,1,1,1,1],[1,0,1,1,1,1,1,1],[1,1,0,1,1,1,1,1],[1,1,1,0,1,1,1,1],
                                [1,1,1,1,0,1,1,1],[1,1,1,1,1,0,1,1],[1,1,1,1,1,1,0,1],[1,1,1,1,1,1,1,0]])
        pyplot.imshow(feature.match_template(numpy.array(sim,numpy.float64),template), vmin = 0, vmax = 1, cmap = pyplot.get_cmap('gray'), aspect = 'auto', origin = 'lower')
        pyplot.title('Match_Template with my own 8x8 beat diagonal template ' + str(i))
        pyplot.savefig('Match_Template with my own 8x8 beat diagonal template ' + str(i) + '.jpg')
    sys.exit()
    
def matchSections():
    pitches = cPickle.load(open('AmuInstPitches.pkl'))
    sections = cPickle.load(open('AmuInstSections.pkl'))
    filenames = cPickle.load(open('AmuFilenames.pkl'))
    keys = cPickle.load(open('AmuKeys.pkl'))
    for i in range(len(pitches)/2):
        pyplot.figure(i,(16,9))
        newp = []
        for vector in pitches[2*i]:
            deq = deque(vector)
            deq.rotate(keys[2*i+1]-keys[2*i])
            l = list(deq)
            newp.append(l)
        pitches[2*i] = newp
        image = numpy.array(pitches[2*i])
        template = numpy.array(pitches[2*i+1][sections[2*i+1][0]:sections[2*i+1][1]])
        im = feature.match_template(image,template,pad_input=True)
        pyplot.vlines(12,0,im.shape[0],'b')
        for j in range(len(sections[2*i+1])-2):    
            template = numpy.array(pitches[2*i+1][sections[2*i+1][j+1]:sections[2*i+1][j+2]])
            temp = feature.match_template(image,template,pad_input=True)
            im = numpy.concatenate((im,temp),axis = 1)
            pyplot.vlines(12*j+12,0,im.shape[0],'b')
        ij = numpy.unravel_index(numpy.argmax(im), im.shape)
        x, y = ij[::-1]
        pyplot.imshow(im, cmap = pyplot.get_cmap('gray'), aspect = 'auto', origin = 'lower')
        pyplot.ylabel(os.path.basename(filenames[2*i]) + " (in beats)")
        pyplot.xlabel(os.path.basename(filenames[2*i+1]) + " (12 Chroma Values Each)")
        pyplot.title('Section Similarity')
        pyplot.plot(x,y,'o',markeredgecolor='r',markerfacecolor='none',markersize=10)
        pyplot.xlim(0,im.shape[1]-1)
        pyplot.ylim(0,im.shape[0])
    pyplot.show()
    sys.exit()
    
""" Determines the best matching section of two songs. The variables x,y mark this
    location as the middle (?) of the template. From x,y the best-matched section
    and its mashed location can be derived as section = x/12 and starting_segment = y."""    
def mashComponents(localAudioFiles):
    instSegments = localAudioFiles[0].analysis.segments #This is the base instrumental
    vocalSegments = localAudioFiles[1].analysis.segments
    instBeats = localAudioFiles[0].analysis.beats
    vocalBeats = localAudioFiles[1].analysis.beats
    pitches = instSegments.pitches
    sections = localAudioFiles[1].analysis.sections #This is the new lead vocal layer
    #pyplot.figure(0,(16,9))
    image = numpy.array(pitches)
    template = numpy.array(vocalSegments.that(selection.overlap(sections[0])).pitches)
    im = feature.match_template(image,template,pad_input=True)
    maxValues = [] #tuples of x coord, y coord, correlation value, and section length (in secs)
    ij = numpy.unravel_index(numpy.argmax(im), im.shape)
    x, y = ij[::-1]
    maxValues.append((numpy.argmax(im),x,y,sections[0].duration))
    for i in range(len(sections)-1):
        template = numpy.array(vocalSegments.that(selection.overlap(sections[i+1])).pitches)
        match = feature.match_template(image,template,pad_input=True)
        ij = numpy.unravel_index(numpy.argmax(match), match.shape)
        x, y = ij[::-1]
        maxValues.append((numpy.argmax(match),12*i+x,y,sections[i+1].duration))
        im = numpy.concatenate((im,match),axis = 1)
    maxValues.sort()
    maxValues.reverse()
    try:
        count = 0
        while(maxValues[count][3] < 15.0):
            count += 1
        x = maxValues[count][1]
        y = maxValues[count][2]
    except:        
        ij = numpy.unravel_index(numpy.argmax(im), im.shape)
        x, y = ij[::-1]
    print "x: ", x
    print "y: ", y
    print "max value: ", numpy.argmax(im)
    print "maxValues: ", maxValues
    #pyplot.imshow(im, cmap = pyplot.get_cmap('gray'), aspect = 'auto')
    #pyplot.plot(x,y,'o',markeredgecolor='r',markerfacecolor='none',markersize=10)
    #pyplot.show()
    sectionLength = len(vocalSegments.that(selection.overlap(sections[x/12])))
    sectionBeats = vocalBeats.that(selection.overlap(sections[x/12]))
    print "sectionLength: ", sectionLength
    matchingSegments = instSegments[(y-sectionLength/2):(y+sectionLength/2)]
    print "matchingSegments: ", matchingSegments
    matchingBeats = instBeats.that(selection.overlap_starts_of(matchingSegments))[-len(sectionBeats):]
    """ Now, I have to make sure sectionBeats and matchingBeats are similarly aligned
        within their group, aka bar of four beats. I will add a beat to the beginning
        of matchingBeats until that condition is met. """
    while(matchingBeats[0].local_context()[0] != sectionBeats[0].local_context()[0]):
        print "matchingBeats[0].local_context()[0]: ", matchingBeats[0].local_context()[0]
        print "sectionBeats[0].local_context()[0]: ", sectionBeats[0].local_context()[0]
        matchingBeats.insert(0,instBeats[matchingBeats[0].absolute_context()[0]-1])
        sectionBeats.append(vocalBeats[sectionBeats[-1].absolute_context()[0]+1])
    print "matchingBeats: ", matchingBeats
    print "sectionBeats: ", sectionBeats
    print "len(matchingBeats): ", len(matchingBeats)
    print "len(sectionBeats): ", len(sectionBeats)
    """ So, matchingSegments contains the instSegments on which vocData will be overlaid.
        Next, the corresponding vocalBeats for those instSegments will be identified. """
    vocData = audio.getpieces(localAudioFiles[3], sectionBeats)
    instData = audio.getpieces(localAudioFiles[2], matchingBeats)
    mix = audio.megamix([instData, vocData])
    mix2 = audio.megamix([vocData, instData])
    mix.encode('Output.mp3')
    mix2.encode('MoreOutput?.mp3')
    vocData.encode('Vocal.mp3')
    instData.encode('Instrumental.mp3')
    os.system('automator /Users/jordanhawkins/Documents/workspace/Automatic\ DJ/import.workflow/')
    sys.exit()
    
def mashFullMixes(localAudioFiles):
    segments = localAudioFiles[1].analysis.segments
    beats1 = localAudioFiles[0].analysis.beats
    beats2 = localAudioFiles[1].analysis.beats
    sections = localAudioFiles[1].analysis.sections
    image = numpy.array(localAudioFiles[0].analysis.segments.pitches)
    template = numpy.array(segments.that(selection.overlap(sections[0])).pitches)
    im = feature.match_template(image,template,pad_input=True)
    for i in range(len(sections)-1):
        template = numpy.array(segments.that(selection.overlap(sections[i+1])).pitches)
        im = numpy.concatenate((im,feature.match_template(image,template,pad_input=True)),axis = 1)
    ij = numpy.unravel_index(numpy.argmax(im), im.shape)
    x, y = ij[::-1]
    data1 = audio.getpieces(localAudioFiles[0], beats1[y:y+len(beats2.that(selection.overlap(sections[x/12])))])
    data2 = audio.getpieces(localAudioFiles[1], beats2.that(selection.overlap(sections[x/12])))
    mix = audio.megamix([data2, data1])
    mix.encode('Full_Mixes_Output.mp3')
    data1.encode('data1.mp3')
    data2.encode('data2.mp3')
    os.system('automator /Users/jordanhawkins/Documents/workspace/Automatic\ DJ/import.workflow/')
    sys.exit()
    
def main():
    #matchSections()
    flushDirectory()
    filenames = getAudioFiles()
    localAudioFiles, filenames, tempos, keys = getInput(filenames)
    matchTempoAndKey(localAudioFiles, tempos, keys)
    localAudioFiles[0] = audio.LocalAudioFile('0.mp3')
    localAudioFiles[1] = audio.LocalAudioFile('1.mp3')
    localAudioFiles[2] = audio.LocalAudioFile('2.mp3')
    localAudioFiles[3] = audio.LocalAudioFile('3.mp3')
    mashComponents(localAudioFiles)
    #mashFullMixes(localAudioFiles)
    equalize_tracks(localAudioFiles)
    pickleAnalysisData(localAudioFiles)
    """
    localAudioFiles = cPickle.load(open('AmuLafs.pkl'))
    filenames = cPickle.load(open('AmuFilenames.pkl'))
    tempos = [laf.analysis.tempo['value'] for laf in localAudioFiles]
    keys = [laf.analysis.key['value'] for laf in localAudioFiles]
    """
    deleteOldSongs(filenames)
    os.system('automator /Users/jordanhawkins/Documents/workspace/Automatic\ DJ/import.workflow/')      
    
if __name__ == '__main__':
    main()

