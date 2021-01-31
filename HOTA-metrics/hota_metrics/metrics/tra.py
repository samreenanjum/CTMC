#####
#
# Adapted from https://github.com/CellTrackingChallenge/measures/blob/master/src/main/java/net/celltrackingchallenge/measures
#
#####


import numpy as np
from scipy.optimize import linear_sum_assignment
from ._base_metric import _BaseMetric
from .. import _timing


class TRA(_BaseMetric):
    """Class which implements the TRA metric"""
    def __init__(self):
        super().__init__()
        main_float_fields = ['TRA']
        self.float_array_fields = main_float_fields
        self.fields = self.float_array_fields
        self.summary_fields = main_float_fields

        self.threshold = 0.5
        self.gt_tracks = {}
        self.res_tracks = {}
        self.levels = {}
        self.doAOGM= False #This flag, when set true, changes the default calc mode, i.e. AOGM will be calculated instead of the TRA (a normalized AOGM).
        self.penalty = self.PenaltyConfig(5.0, 10.0, 1.0, 1.0, 1.5, 1.0) #default values as provided by CTC
        self.aogm = 0.0
        self.max_split = 1
        self.do_mitosis = False

        
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
            return -1
        ## returns index of the input RES label
        def res_findLabel(self,label):
            for i in range(0, len(self.m_res_lab)):
                if(label==self.m_res_lab[i]):
                    return i
            return -1
            
            
    @_timing.time
    def eval_sequence(self, data):
        """Calculates TRA metrics for one sequence"""
        # Initialize results
        res = {}
        for field in self.fields:
            res[field] = 0
        self.aogm =0.0
        self.levels={}

        self.do_mitosis = data['do_mitosis']
        self.gt_tracks = data['gt_tracks']
        self.res_tracks = data['tracker_tracks']

        # Calculate scores for each timestep
        for t, (gt_ids_t, tracker_ids_t) in enumerate(zip(data['gt_ids'], data['tracker_ids'])):

            ## check for empty frames
            if len(gt_ids_t)==0 and len(tracker_ids_t)==0:
                continue
            if len(gt_ids_t) == 0:
                gt_ids_t = []
            if len(tracker_ids_t) == 0:
                tracker_ids_t = []

            level = self.TemporalLevel(t)
            ## get the unique list of gt labels
            level.m_gt_lab =  gt_ids_t
            ## get the unique list of res labels
            level.m_res_lab = tracker_ids_t

            ## init the matches
            level.m_gt_match = [-1]*len(level.m_gt_lab)
            level.m_res_match = [set() for index in range(len(level.m_res_lab))]

            # Calc score matrix
            similarity = data['similarity_scores'][t] ## IoUs
            similarity[similarity < self.threshold] = np.nan
            matches = np.argwhere(~np.isnan(similarity))
        
            level.m_gt_match = np.array(level.m_gt_match)
            level.m_gt_match[matches[:,0].astype(int)] = list(matches[:,1])

            [level.m_res_match[item[1]].add(item[0]) for item in matches]

            ## save the level
            self.levels[t] = level

        ## Compute the TRA metric
        res['TRA'] = self.calculate()
        return res

    
    def combine_sequences(self, all_res):
        """Combines metrics across all sequences"""
        res = {}
        res['TRA'] = [all_res[k]['TRA'] for k in all_res.keys()]
        return res

    
    ## Helper functions
    def getGTMatch(self, level, lbl):
        '''
        Returns index of RES label that matches with given GT lbl, or -1 if no such label was found.
        '''
        return (level.m_gt_match[level.gt_findLabel(lbl)] )


    def getResMatch(self, level, lbl):
        '''
        Returns collection of indices of GT labels that matches with given RES lbl,
        *or collection with single item (of -1 value) if no such label was found.
        '''
        idx = level.res_findLabel(lbl)
        if (idx != -1):
            return (level.m_res_match[idx])
        else:
            tmp = set()
            tmp.add(-1)
            return tmp


    def existGTEdge(self, levels, start_level, start_index, end_level, end_index, tracks, parental):
        '''
        Check if there is an edge of a given type between given temporal levels in the reference tracks.
        levels = list of TemporalLevel objects
        tracks is dictionary (integer, Tracks)
        parental = output boolean variable
        '''
        # reasonable label indices? existing labels?
        if (start_index != -1 and end_index != -1):
            
            #  get labels at given times at given indices
            start_label = levels[start_level].m_gt_lab[start_index]
            end_label = levels[end_level].m_gt_lab[end_index]
            # check the type of the edge
            if(start_label==end_label):
                # the edge connects nodes from the same track,
                # are the nodes temporal consecutive? is it really an edge?
                if ((start_level + 1) == end_level):
                    parental[0] = False # same track, can't be a parental link
                    return True
            else:
                # the edge connects two tracks, get them...
                parent = tracks[start_label]
                child = tracks[end_label]
                # is the edge correctly connecting two tracks?
                if (parent.m_end == start_level and child.m_begin == end_level and child.m_parent == start_label):
                    parental[0] = True;
                    return True;
        else:
            return False

        
    def existResEdge(self, levels, start_level, start_index, end_level, end_index, tracks):
        # reasonable label indices? existing labels?
	# do start and end labels/nodes have 1:1 matching?
        if (start_index != -1 and end_index != -1
	    and len(levels[start_level].m_res_match[start_index]) == 1
	    and len(levels[end_level].m_res_match[end_index]) == 1):
            
            # get labels at given times at given indices
            start_label = levels[start_level].m_res_lab[start_index]
            end_label = levels[end_level].m_res_lab[end_index]

            # check the type of the edge
            if (start_label == end_label):
                # the edge connects nodes from the same track,
                # are the nodes temporal consecutive? is it really an edge?
                return ((start_level + 1) == end_level)
            else:
                # the edge connects two tracks, get them..
                parent = tracks[start_label]
                child = tracks[end_label]

                # is the edge correctly connecting two tracks?
                return (parent.m_end == start_level and child.m_begin == end_level and child.m_parent == start_label)
        # default
        return False

    
    def findEDAndECEdges(self, levels, gt_tracks, res_tracks):
        '''
        Find edges in the computed tracks that must be removed or altered.
        '''
        parent = [False]

        # over all tracks/labels present in the result data
        for key in res_tracks:
            # key is the label id
            # shortcut to track data
            res_track = res_tracks[key]
            #A) check the edge between the first node of the current track
            #B) and the last one of the parent track

            #A:
            end_level = int(res_track.m_begin)
            end_match = self.getResMatch(levels[end_level], key)

            #does this track have a parent?
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
                    else:
                        #there is no corresponding edge in GT, that's an error
                        self.aogm += self.penalty.m_ed
                       
            ## check edges within the current track
            for i in range((int)(res_track.m_begin), (int)((res_track.m_end)-1), 1):
                #define temporal consecutive nodes
                start_level = end_level
                start_match = end_match
                end_match = self.getResMatch(levels[end_level], key)

                #*_match contain lists of indices of GT labels that matches
                if (len(start_match) == 1 and len(end_match) == 1):
                    #we have a reasonable edge here, deal with this RES edge:
                    if(self.existGTEdge(levels, start_level, next(iter(start_match)), end_level, next(iter(end_match)), gt_tracks, parent)):
                        #corresponding edge exists in GT, however should not be parental link
                        if(parent[0]==True):
                            #it is parental, that's an error
                            self.aogm +=self.penalty.m_ec
                    else:
                        #there is no corresponding edge in GT, that's an error
                        self.aogm += self.penalty.m_ed

    
    def findEAEdges(self, levels, gt_tracks, res_tracks):
        '''
        Find edges in the reference tracks that must be added.
        '''

        for key in gt_tracks:
            gt_track = gt_tracks[key]
            gt_track_id = key
            #A)check the edge between the first node of the current track
            #B) and the last one of the parent track

            #case A:
            end_level = int(gt_track.m_begin)
            end_index = self.getGTMatch(levels[end_level], gt_track_id)

            # does this track have a parent?
            if(gt_track.m_parent > 0):
                # yes, it does, case B:
                start_level = gt_tracks[gt_track.m_parent].m_end
                start_index = self.getGTMatch(levels[start_level], (int)(gt_track.m_parent))

                #*_index contain indices of RES labels that matches ...
                if( not self.existResEdge(levels, start_level, start_index, end_level, end_index, res_tracks)):
                    #... but there is no edge between them, that's an error
                    self.aogm += self.penalty.m_ea

            ## check edges within the current track
            for i in range((int)(gt_track.m_begin), (int)(gt_track.m_end), 1):
                #define temporal consecutive nodes
                start_level = end_level
                start_index = end_index
                end_index = self.getGTMatch(levels[end_level], gt_track_id)

                #*_index contain indices of RES labels that matches ...
                if (not self.existResEdge(levels, start_level, start_index, end_level, end_index, res_tracks)):
                    #... but there is no edge between them, that's an error
                    self.aogm +=self.penalty.m_ea

                    
    def calculate(self):
        '''
        Main function that computes the TRA metric
        '''
        ## cache data
        gt_tracks = self.gt_tracks
        res_tracks = self.res_tracks
        levels = self.levels
      
        # Checks matching between all nodes discovered in both GT and RES images
        # levels is a dict {frame: level}     
        for key in levels:
            level = levels[key] 
            #sweep over all gt labels
            #check if we have found corresponding res label
            for i in range(len(level.m_gt_lab)):                
                if(level.m_gt_match[i] == -1):
                    #no correspondence -> the gt label represents FN (false negative) case                  
                    self.aogm += self.penalty.m_fn                    
            #for every res label, check we have found exactly one corresponding gt label
            for j in range(len(level.m_res_lab)):
                #number of overlapping gt labels
                num = len(level.m_res_match[j])
                if (num==0):
                    #no label or too few labels
                    self.aogm += self.penalty.m_fp
                    
                elif (num>1):
                    #too many labels...
                    self.aogm +=(num-1) *self.penalty.m_ns
                    self.max_split = num if num > self.max_split else self.max_split
                else:
                    continue
      
        #check the minimality condition
        if((self.max_split - 1) * self.penalty.m_ns > (self.penalty.m_fp + self.max_split * self.penalty.m_fn)):
            print("Warning: The minimality condition broken! (m*= {} )".format(max_split))
            
        self.findEDAndECEdges(levels, gt_tracks, res_tracks)
        self.findEAEdges(levels, gt_tracks, res_tracks)      
              
        ## AOGM
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
            self.aogm = aogm_empty if self.aogm > aogm_empty else self.aogm
            #normalization
            self.aogm = 1.0-(self.aogm/aogm_empty)

        return self.aogm
