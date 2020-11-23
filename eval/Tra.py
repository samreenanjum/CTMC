######################
#  Adapted from:
#  https://github.com/CellTrackingChallenge/CTC-FijiPlugins/blob/master/CTC-paper/src/main/java/de/mpicbg/ulman/ctc/workers/TRA.java
#
######################

import sys
import numpy as np
import pandas as pd
import warnings


from TrackUtils import Track, Fork, TemporalLevel
from TrackDataCache import TrackDataCache


class TRA:

    def __init__(self):
        self.n = 3 #noOfDigits
        self.cache = TrackDataCache()
        self.doConsistencyCheck = False
        self.doMatchingReports = False #do report on how matching is being established between the reference and detected (computed) tracking segments (aka tracking nodes).
        self.doLogReports = False #do report list of discrepancies between the reference and computed tracking result
        self.doAOGM= False #This flag, when set to true, changes the default calculation mode, that is the AOGM will be calculated instead of the TRA (which is essentially a normalized AOGM).
        self.penalty = self.PenaltyConfig(5.0, 10.0, 1.0, 1.0, 1.5, 1.0) #default values as provided by CTC
        self.aogm = 0.0
        self.max_split = 1

    class PenaltyConfig:
        def __init__(self, ns, fn, fp, ed, ea, ec):
            self.m_ns = ns #/** The penalty for a splitting operation. */
            self.m_fn = fn #/** The penalty for a false negative node. */
            self.m_fp = fp #/** The penalty for a false positive node. */
            self.m_ed = ed #/** The penalty for a redundant edge. */
            self.m_ea = ea #/** The penalty for a missing edge. */
            self.m_ec = ec #/** The penalty for an edge with wrong semantics. */
            if (ns < 0.0 or fn < 0.0 or fp < 0.0 or ed < 0.0 or ea < 0.0 or ec < 0.0):
                raise ValueError("All weights must be nonnegative numbers!")
	       
            
            
    '''
    TODO: doConsistencyCheck Function
    def doConsistencyCheck (self, levels, tracks, isGTCheck):
    '''

    '''
    Returns index of RES label that matches with given GT lbl,
    or -1 if no such label was found.
    '''
    def getGTMatch(self, level, lbl):
        return (level.m_gt_match[level.gt_findLabel(lbl)] )


    '''
    * Returns collection of indices of GT labels that matches with given RES lbl,
    * or collection with single item (of -1 value) if no such label was found.
    '''
    def getResMatch(self, level, lbl):
        #print("lbl = {}".format(lbl))
        idx = level.res_findLabel(lbl)
        if (idx != -1):
            return (level.m_res_match[idx])
        else:
            tmp = set()
            tmp.add(-1)
            return tmp

    '''
    * Check if there is an edge of a given type between given
    * temporal levels in the reference tracks.
    #levels = list of TemporalLevel objects
    #tracks is dictionary (integer, Tracks)
    #parental = output boolean variable
    
    '''
    def existGTEdge(self, levels, start_level, start_index, end_level, end_index, tracks, parental):
        #//reasonable label indices? existing labels?
        if (start_index != -1 and end_index != -1):
            
            #// get labels at given times at given indices
            start_label = levels[start_level].m_gt_lab[start_index]
            end_label = levels[end_level].m_gt_lab[end_index]
            #//check the type of the edge
            if(start_label==end_label):
                #// the edge connects nodes from the same track,
                #// are the nodes temporal consecutive? is it really an edge?
                if ((start_level + 1) == end_level):
                    parental[0] = False #//same track, can't be a parental link
                    return True
            else:
                #// the edge connects two tracks, get them...
                parent = tracks[start_label]
                child = tracks[end_label]
                #//is the edge correctly connecting two tracks?
                if (parent.m_end == start_level and child.m_begin == end_level and child.m_parent == start_label):
                    parental[0] = True;
                    return True;
        else:
            return False



    '''
    '''
    def existResEdge(self, levels, start_level, start_index, end_level, end_index, tracks):
        #//reasonable label indices? existing labels?
	#//do start and end labels/nodes have 1:1 matching?
        if (start_index != -1 and end_index != -1
	    and len(levels[start_level].m_res_match[start_index]) == 1
	    and len(levels[end_level].m_res_match[end_index]) == 1):
            
            #// get labels at given times at given indices
            start_label = levels[start_level].m_res_lab[start_index]
            end_label = levels[end_level].m_res_lab[end_index]

            #check the type of the edge
            if (start_label == end_label):
                # the edge connects nodes from the same track,
                # are the nodes temporal consecutive? is it really an edge?
                return ((start_level + 1) == end_level)
            else:
                # the edge connects two tracks, get them..
                parent = tracks[start_label]
                child = tracks[end_label]

                #is the edge correctly connecting two tracks?
                return (parent.m_end == start_level and child.m_begin == end_level and child.m_parent == start_label)
        # default
        return False


    '''
    Find edges in the computed tracks that must be removed or altered.
    '''
    def findEDAndECEdges(self, levels, gt_tracks, res_tracks):
        parent = [False]

        #over all tracks/labels present in the result data
        for key in res_tracks:
            # key is the label id
            # shortcut to track data
            res_track = res_tracks[key]
            #A) check the edge between the first node of the current track
            #B) and the last one of the parent track

            #A:
            end_level = int(res_track.m_begin)
            print("key = {}, end_level = {}, len of levels = {}".format(key, end_level, len(levels)))
            end_match = self.getResMatch(levels[end_level], key)

            #`does this track have a parent?
            if(res_track.m_parent > 0):

                #B:
                start_level = (int)(res_tracks[res_track.m_parent].m_end)
                start_match = self.getResMatch(levels[start_level], (int)(res_track.m_parent))

                #*_match contain lists of indices of GT labels that matches
                if (len(start_match) == 1 and len(end_match) == 1):
                    #right number of matches, deal with this RES edge:
                    if(self.existGTEdge(levels, start_level, next(iter(start_match)), end_level, next(iter(end_match)), gt_tracks, parent)):
                        #corresponding edge exists in GT, does it connect two different tracks too?
                        if(parent[0]==False):
                            #it does not connect different tracks, that's an error
                            self.aogm += self.penalty.m_ec
                            print("aogm: = {}, penalty.ec = {}".format(self.aogm, self.penalty.m_ec))
                            ## TODO: add log code
                    else:
                        #there is no corresponding edge in GT, that's an error
                        self.aogm += self.penalty.m_ed
                        print("aogm: = {}, penalty.ed = {}".format(self.aogm, self.penalty.m_ed))
                        ## TODO: add log code

            ## check edges within the current track
            for i in range((int)(res_track.m_begin), (int)((res_track.m_end)-1), 1):
                
                #define temporal consecutive nodes
                start_level = end_level
                start_match = end_match
                end_level = i+1
                #print("end_level = {}, len of levels = {}, res_track.m_end = {}".format(end_level, len(levels), res_track.m_end))
                end_match = self.getResMatch(levels[end_level], key)

                #*_match contain lists of indices of GT labels that matches
                if (len(start_match) == 1 and len(end_match) == 1):
                    #we have a reasonable edge here, deal with this RES edge:
                    if(self.existGTEdge(levels, start_level, next(iter(start_match)), end_level, next(iter(end_match)), gt_tracks, parent)):
                        #corresponding edge exists in GT, should not be parental link however
                        if(parent[0]==True):
                            #it is parental, that's an error
                            self.aogm +=self.penalty.m_ec
                            print("aogm: = {}, penalty.ec = {}".format(self.aogm, self.penalty.m_ec))
                            ## TODO: add log code

                    else:
                        #there is no corresponding edge in GT, that's an error
                        self.aogm += self.penalty.m_ed
                        print("aogm: = {}, penalty.ed = {}".format(self.aogm, self.penalty.m_ed))
                        ## TODO: add log code


    '''
    Find edges in the reference tracks that must be added.
    '''
    def findEAEdges(self, levels, gt_tracks, res_tracks):
        for key in gt_tracks:
            gt_track = gt_tracks[key]
            gt_track_id = key
            #A)check the edge between the first node of the current track
            #B) and the last one of the parent track

            # A:
            end_level = int(gt_track.m_begin)
            end_index = self.getGTMatch(levels[end_level], gt_track_id)

            # does this track have a parent?
            if(gt_track.m_parent > 0):

                # yes, it does
                # B:
                start_level = gt_tracks[gt_track.m_parent].m_end
                start_index = self.getGTMatch(levels[start_level], (int)(gt_track.m_parent))

                #*_index contain indices of RES labels that matches ...
                if( not self.existResEdge(levels, start_level, start_index, end_level, end_index, res_tracks)):
                    #... but there is no edge between them, that's an error
                    self.aogm += self.penalty.m_ea
                    print("aogm: = {}, penalty.ea = {}".format(self.aogm, self.penalty.m_ea))
                    ## TODO: add log code


            ## check edges within the current track
            for i in range((int)(gt_track.m_begin), (int)(gt_track.m_end), 1):
                #define temporal consecutive nodes
                start_level = end_level
                start_index = end_index
                end_level = i + 1
                #print("key/gt = {}, end_level = {}".format(key, end_level))
                #print(levels[end_level])

                end_index = self.getGTMatch(levels[end_level], gt_track_id)

                #*_index contain indices of RES labels that matches ...
                if (not self.existResEdge(levels, start_level, start_index, end_level, end_index, res_tracks)):
                    #... but there is no edge between them, that's an error
                    self.aogm +=self.penalty.m_ea
                    print("aogm: = {}, penalty.ea = {}".format(self.aogm, self.penalty.m_ea))
                    ## TODO: add log code

                    

    '''
    the main TRA calculator
    '''    
    def calculate(self, gtPath, resPath):
        self.cache.n = self.n
        self.cache.calculate(gtPath, resPath) ## compute TrackDataCache

        ## cache data
        gt_tracks = self.cache.gt_tracks
        res_tracks = self.cache.res_tracks
        levels = self.cache.levels

        ## TODO:consistency check
        #if (doConsistencyCheck):
	#    CheckConsistency(levels,  gt_tracks, true);
	#    CheckConsistency(levels, res_tracks, false);

        #checks matching between all nodes discovered in both GT and RES images

        #levels is dictionary {frame:level}
        for key in levels:
            level = levels[key] 
            #sweep over all gt labels
            #print("level - gt_lab: {} ".format(level.m_gt_lab))
            for i in range(len(level.m_gt_lab)):                #check if we have found corresponding res label
                if(level.m_gt_match[i] == -1):
                    #no correspondence -> the gt label represents FN (false negative) case
                    self.aogm += self.penalty.m_fn
                    print("aogm: = {}, penalty.fn = {}".format(self.aogm, self.penalty.m_fn))
                    ## TODO: add log code


            #for every res label, check we have found exactly one corresponding gt label
            for j in range(len(level.m_res_lab)):
                #number of overlapping gt labels
                num = len(level.m_res_match[j])
                #print("num = {}, res_label = {}, len ofgt matches = {}".format(num, level.m_res_lab, len(level.m_res_match[j])))
                if (num==0):
                    #no label -- too few
                    self.aogm += self.penalty.m_fp
                    print("aogm: = {}, penalty.fp = {}".format(self.aogm, self.penalty.m_fp))
                    ## TODO: add log code

                elif (num>1):
                    #too many labels...
                    self.aogm +=(num-1) *self.penalty.m_ns
                    print("aogm: = {}, penalty.ns = {}, num = {}".format(self.aogm, self.penalty.m_ns, num))
                    ## TODO: add log code
                    self.max_split = num if num > self.max_split else self.max_split

                else:
                    continue
                    #print("")
                    #print("num==1, reslabel = {}, num = {}".format(level.m_res_match[j], num))
                    ## TODO: add log code

        #check the minimality condition
        if((self.max_split - 1) * self.penalty.m_ns > (self.penalty.m_fp + self.max_split * self.penalty.m_fn)):
            print("error: minimality condition")
            ## TODO: log code

        self.findEDAndECEdges(levels, gt_tracks, res_tracks)
        self.findEAEdges(levels, gt_tracks, res_tracks)


        ## TODO: add log code


        ## AOGm
        if(self.doAOGM == False):
            #how many parental links to add
            num_par = 0
            #how many track links (edges) to add
            sum_val = 0

            for key in gt_tracks:
                t = gt_tracks[key]
                sum_val +=int(t.m_end) -int(t.m_begin)

                if (t.m_parent>0):
                    num_par =+1

            aogm_empty = self.penalty.m_fn * (sum_val + len(gt_tracks))+ self.penalty.m_ea* (sum_val + num_par) # nodes  + edges

            # 
            self.aogm = aogm_empty if self.aogm > aogm_empty else self.aogm
            print("aogm: = {}".format(self.aogm))
            #normalization
            self.aogm = 1.0-(self.aogm/aogm_empty)
            print("aogm: = {}, aogm_empty".format(self.aogm, aogm_empty))

            ## TODO: add log data

        else:
            print(self.aogm)
            ## TODO: add log data

        return self.aogm
