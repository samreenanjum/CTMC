######################
#
#  Adapted from:
#  https://github.com/CellTrackingChallenge/CTC-FijiPlugins/blob/master/CTC-paper/src/main/java/de/mpicbg/ulman/ctc/workers/TrackDataCache.java
#  https://github.com/cheind/py-motmetrics
######################



import sys
import numpy as np
import pandas as pd
import warnings
import progressbar
import time

from TrackUtils import Track, Fork, TemporalLevel

##
class TrackDataCache:


    def __init__(self):
        self.n = 3#noOfDigits = 3
        self.gtPath=''
        self.resPath=''

        self.gt_tracks = {}
        self.res_tracks = {}
        #self.levels = []
        self.levels = {}

        self.gt_forks = []
        self.res_forks = []
    ## read Image functions TODO

    def timing(f):
        def wrap(*args, **kwargs):
            time1 = time.time()
            ret = f(*args, **kwargs)
            time2 = time.time()
            print('{:s} function took {:.3f} ms'.format(f.__name__, (time2-time1)*1000.0))

            return ret
        return wrap
    
    ## TODO: load Track files
    def loadTrackFile(self, fname, track_list):
        ## fill in
        df = pd.read_csv(
            fname,
            sep=" ",
            index_col=[0],
            skipinitialspace=True,
            header=None,
            names=['Id', 'Begin', 'End', 'Parent'],
            engine='python'
        )

        for index, row in df.iterrows():
            track_list[index] = Track(index, row['Begin'], row['End'], row['Parent'])
        ##    return 0
        
        ##
        
    ## calculate
    #@timing
    def calculate (self, gtPath, resPath):

        ## for temp testing
        #self.loadTrackFile("data/", self.gt_tracks)
        #self.loadTrackFile("data/", self.res_tracks)
               
       
        self.loadTrackFile(gtPath+"/TRA/man_track.txt", self.gt_tracks);
	self.loadTrackFile(resPath+"/res_track.txt", self.res_tracks);
        
        time = 1 ## start frame counter

        ## Read the MOT annotation files into dataframes
        df_gt = self.load_txt(gtPath)
        df_res = self.load_txt(resPath)

        
        ## Total number of frames
        allframeids = df_gt.index.union(df_res.index).levels[0]

        #allframeids = allframeids[1:3]
        num_frames = np.max(allframeids)#allframeids.size
        
        print(np.min(allframeids))
        print(np.max(allframeids))
        print(num_frames)
        bar = progressbar.ProgressBar(maxval=num_frames+2, \
                                      widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        
       
        ## for each frame
        distfields = ['X', 'Y', 'Width', 'Height']
        df_gt = df_gt[distfields]
        df_res = df_res[distfields]
        fid_to_fgt = dict(iter(df_gt.groupby('FrameId')))
        fid_to_fdt = dict(iter(df_res.groupby('FrameId')))
        for fid in allframeids: 
            bar.update(fid-1)

            ## get the subset of annotations for the frame
            fgt = fid_to_fgt.get(fid, None)
            fdt = fid_to_fdt.get(fid, None)


            ## TODO: deal with empty images, for now ignore them
            if (fgt is None or fdt is None):
                time = time+1
                continue
            
            self.classifyLabels(fgt, fdt, fid);
            time =time+1

        ## end loop
        #self.detect_forks(self.gt_tracks,  self.gt_forks);
        #self.detect_forks(self.res_tracks,  self.res_forks);
        bar.finish()



    '''
    Adapted from: https://github.com/cheind/py-motmetrics/blob/d261d16cca263125b135571231011ccf9efd082b/motmetrics/io.py
    '''    
    def load_txt(self, fname, **kwargs):
        r"""Load MOT challenge data.
        Params
        ------
        fname : str
        Filename to load data from
        Kwargs
        ------
        sep : str
        Allowed field separators, defaults to '\s+|\t+|,'
        min_confidence : float
        Rows with confidence less than this threshold are removed.
        Defaults to -1. You should set this to 1 when loading
        ground truth MOTChallenge data, so that invalid rectangles in
        the ground truth are not considered during matching.
        Returns
        ------
        df : pandas.DataFrame
        The returned dataframe has the following columns
            'X', 'Y', 'Width', 'Height', 'Confidence', 'ClassId', 'Visibility'
        The dataframe is indexed by ('FrameId', 'Id')
        """
        
        sep = kwargs.pop('sep', r'\s+|\t+|,')
        min_confidence = kwargs.pop('min_confidence', -1)
        df = pd.read_csv(
            fname,
            sep=sep,
            index_col=[0, 1],
            skipinitialspace=True,
            header=None,
            names=['FrameId', 'Id', 'X', 'Y', 'Width', 'Height', 'Confidence', 'ClassId', 'Visibility', 'unused'],
            engine='python'
        )
        
        # Account for matlab convention.
        df[['X', 'Y']] -= (1, 1)
        
        # Removed trailing column
        del df['unused']

        #TODO: check confidence scores of GT data
        # Remove all rows without sufficient confidence
        return df[df['Confidence'] >= min_confidence]
    
    ## classify Labels function for each frame (time)
    #@timing
    def classifyLabels(self, df_gt, df_res, time, overlapRatio=0.5):
        level = TemporalLevel(time)

        ## Not required for MOT -- Read the annotations from file to dataframe
        #df_gt = self.load_txt(gtAnnFile)
        #df_res = self.load_txt(resAnnFile)

        ## get the unique list of gt labels
        level.m_gt_lab =  df_gt.index.get_level_values('Id').unique()
        ## get the unique list of res labels
        level.m_res_lab = df_res.index.get_level_values('Id').unique()
        
        ## init the matches
        level.m_gt_match = [-1]*len(level.m_gt_lab)
        level.m_res_match = [set() for index in range(len(level.m_res_lab))]

        overlap = 0
       
        dists = self.iou_mat_util(df_gt.values, df_res.values)
        matches = np.argwhere(~np.isnan(dists))

        level.m_gt_match = np.array(level.m_gt_match)
        level.m_gt_match[matches[:,0].astype(int)] = list(matches[:,1])
    
        [level.m_res_match[item[1]].add(item[0]) for item in matches]

        ## finally, #save the level
        self.levels[time] = level

    ## TODO: Detect Forks
    def detect_forks(self, tracks, forks):

        ##create a map of families, dict of lists (parent: list of kids)
        families = {}
        for key, track in tracks.items():
            #does the track have a parent
            if(track.m_parent>0):
                #retrieve current list of kids of this track's parent
                kids = families.get(track.m_parent)
                if(kids==None):
                    kids = []
                    families[track.m_parent] = kids#list()#[kids]
                kids.append(key)
                    
        print("families = {}".format(families))
        
        #now that we have (piece-by-piece) collected Fork-like data,
        #fill the output variable finally
        #key == parent,  values == kids 
        for key in families:
            #retrieve final list of kids of this parent
            kids = families[key]
            
            #enough kids for a fork?
            if(len(kids)>1):
                forks.append(Fork(key, kids))
            


        
    ## helper
    def rect_min_max(self, r):
        min_pt = r[..., :2]
        size = r[..., 2:]
        max_pt = min_pt + size
        return min_pt, max_pt
    
    ## compute IOU
    def iou_mat_prep (self, fgt, fdt, fid=None):
        distfields = ['X', 'Y', 'Width', 'Height']
        fgt = fgt[distfields]
        fdt = fdt[distfields]

        fgt = np.asfarray(fgt)
        fdt = np.asfarray(fdt)
        iou = self.boxiou(fgt, fdt)
        return iou

    ##
    def iou_mat_util(self, fgt, fdt, max_iou=0.5):
        #distfields = ['X', 'Y', 'Width', 'Height']
        #fgt = fgt[distfields]
        #fdt = fdt[distfields]

        if np.size(fgt) == 0 or np.size(fdt) == 0:
            return np.empty((0, 0))
        objs = np.asfarray(fgt)
        hyps = np.asfarray(fdt)
        assert objs.shape[1] == 4
        assert hyps.shape[1] == 4
        iou = self.boxiou(objs[:, None], hyps[None, :])
        dist = 1 - iou
        return np.where(dist > max_iou, np.nan, dist)


    ##
    def boxiou(self, a, b):
        """Computes IOU of two rectangles."""

        a_min, a_max = self.rect_min_max(a)
        b_min, b_max = self.rect_min_max(b)
        # Compute intersection.
        i_min = np.maximum(a_min, b_min)
        i_max = np.minimum(a_max, b_max)
        i_size = np.maximum(i_max - i_min, 0)
        i_vol = np.prod(i_size, axis=-1)
        # Get volume of union.
        a_size = np.maximum(a_max - a_min, 0)
        b_size = np.maximum(b_max - b_min, 0)
        a_vol = np.prod(a_size, axis=-1)
        b_vol = np.prod(b_size, axis=-1)
        u_vol = a_vol + b_vol - i_vol
        return np.where(i_vol == 0, np.zeros_like(i_vol, dtype=np.float),
                        self.quiet_divide(i_vol, u_vol))

    
    def quiet_divide(self, a, b):
        """Quiet divide function that does not warn about (0 / 0)."""
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            return np.true_divide(a, b)
'''        
##
class Track:
    def __init__(self, id, begin, end, parent ):
        self.m_id = id
        self.m_begin = begin
        self.m_end = end
        self.m_parent = parent


    ## TODO: toString function?


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
    def gt_findLabel(label):
        for i in range(0, len(self.m_gt_lab)):
            if(label==self.m_gt_lab[i]):
                return i
        raise ValueError("Label not found in GT")

    ## returns index of the input RES label
    def res_findLabel(label):
        for i in range(0, len(self.m_res_lab)):
            if(label==self.m_res_lab[i]):
                return i
        raise ValueError("Label not found in RES")
'''
    
