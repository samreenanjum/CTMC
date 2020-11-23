import sys
import numpy as np
import pandas as pd


##
class Track:
    def __init__(self, id, begin, end, parent ):
        self.m_id = id
        self.m_begin = begin
        self.m_end = end
        self.m_parent = parent


    ## TODO: toString function?
    def __repr__(self):
        return "ID: %d, Begin: %d, End: %d, Parent: %d )" %(self.m_id, self.m_begin, self.m_end, self.m_parent)
'''
For Representation
'''
class Fork:
    def __init__(self, parent_id, child_ids):
        self.m_parent_id = parent_id
        self.m_child_ids = child_ids.copy()


    #returns index of the input GT label, or -1 if label was not found
    def findChildLabel(self, label):
        for i in range(0, len(self.m_child_ids)):
            if(label==self.m_child_ids[i]):
                return i

        return -1
    
class TemporalLevel:
    def __init__(self, level):
        self.m_level = level #Temporal level -- a particular time point. 
        self.m_gt_lab = [] #List of labels (and their sizes) in the reference image
        self.m_gt_size = []
        self.m_res_lab = [] #List of labels (and their sizes) in the computed image.
        self.m_res_size = []
        ##Matching matrix, stored as a plain 1D array.
	#* For every position (j,i) (j-th row and i-th column), it contains
	#* number of voxels in the intersection between m_res_lab[j] label
	#* and m_gt_lab[i] label.
        self.m_match = [] #Matching matrix, stored as a plain 1D array.
        #* Indices of reference vertex matching, i.e., it is of the same length
	#* as m_gt_lab and it holds indices into the m_res_lab.
	#* It is initialized with -1 values. After matching is done,
	#* the value -1 corresponds to a FN vertex.
        self.m_gt_match = []
        #* Sets of indices of computed vertex matching, i.e., it is of the same length
	#* as m_res_lab and it holds sets of indices into the m_gt_lab.
	#* It is initialized with empty sets. After matching is done,
	#* an empty set corresponds to a FP vertex.
        self.m_res_match =[] #HashSet<Integer>[] m_res_match = null;

    ## returns index of the input GT label
    def gt_findLabel(self,label):
        for i in range(0, len(self.m_gt_lab)):
            if(label==self.m_gt_lab[i]):
                return i
        #raise ValueError("Label not found in GT")
        return -1
    ## returns index of the input RES label
    def res_findLabel(self,label):
        for i in range(0, len(self.m_res_lab)):
            if(label==self.m_res_lab[i]):
                return i
        return -1
        #raise ValueError("Label not found in RES, label = {}, list = {}".format(label, self.m_res_lab))

    
