
import os
import csv
import configparser
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from ._base_dataset import _BaseDataset
from .. import utils
from .. import _timing


class CTMC2DBox(_BaseDataset):
    """Dataset class for MOT Challenge 2D bounding box tracking"""

    @staticmethod
    def get_default_dataset_config():
        """Default class config values"""
        code_path = utils.get_code_path()
        default_config = {
            'GT_FOLDER': os.path.join(code_path, 'data/gt/mot_challenge/'),  # Location of GT data
            'TRACKERS_FOLDER': os.path.join(code_path, 'data/trackers/mot_challenge/'),  # Trackers location
            'OUTPUT_FOLDER': None,  # Where to save eval results (if None, same as TRACKERS_FOLDER)
            'TRACKERS_TO_EVAL': None,  # Filenames of trackers to eval (if None, all in folder)
            'CLASSES_TO_EVAL': ['cell'],  # Valid: ['cell']
            'BENCHMARK': 'CTMC',  # Valid: 'CTMC' 
            'SPLIT_TO_EVAL': 'train',  # Valid: 'train', 'test' 
            'INPUT_AS_ZIP': False,  # Whether tracker input files are zipped
            'PRINT_CONFIG': True,  # Whether to print current config
            'TRACKER_SUB_FOLDER': 'data',  # Tracker files are in TRACKER_FOLDER/tracker_name/TRACKER_SUB_FOLDER
            'OUTPUT_SUB_FOLDER': '',  # Output files are saved in OUTPUT_FOLDER/tracker_name/OUTPUT_SUB_FOLDER
            'SEQMAP_FOLDER': None,  # Where seqmaps are found (if None, GT_FOLDER/seqmaps)
            'SEQMAP_FILE': None,  # Directly specify seqmap file (if none use seqmap_folder/benchmark-split_to_eval)
            'SKIP_SPLIT_FOL': False,  # If False, data is in GT_FOLDER/BENCHMARK-SPLIT_TO_EVAL/ and in
                                      # TRACKERS_FOLDER/BENCHMARK-SPLIT_TO_EVAL/tracker/
                                      # If True, then the middle 'benchmark-split' folder is skipped for both.
        }
        return default_config

    def __init__(self, config=None):
        """Initialise dataset, checking that all required files are present"""
        super().__init__()
        # Fill non-given config values with defaults
        self.config = utils.init_config(config, self.get_default_dataset_config(), self.get_name())

        self.benchmark = self.config['BENCHMARK']
        gt_set = self.config['BENCHMARK'] + '-' + self.config['SPLIT_TO_EVAL']
        if not self.config['SKIP_SPLIT_FOL']:
            split_fol = gt_set
        else:
            split_fol = ''
        self.gt_fol = os.path.join(self.config['GT_FOLDER'], split_fol)
        self.tracker_fol = os.path.join(self.config['TRACKERS_FOLDER'], split_fol)
        self.should_classes_combine = False
        self.data_is_zipped = self.config['INPUT_AS_ZIP']

        self.output_fol = self.config['OUTPUT_FOLDER']
        if self.output_fol is None:
            self.output_fol = self.tracker_fol

        self.tracker_sub_fol = self.config['TRACKER_SUB_FOLDER']
        self.output_sub_fol = self.config['OUTPUT_SUB_FOLDER']

        self.exists_tra_file = True
        self.do_mitosis = False
        
        # Get classes to eval
        self.valid_classes = ['cell']
        self.class_list = [cls.lower() if cls.lower() in self.valid_classes else None
                           for cls in self.config['CLASSES_TO_EVAL']]
   
        # Get sequences to eval and check gt files exist
        self.seq_list = []
        self.seq_lengths = {}
        if self.config["SEQMAP_FILE"]:
            seqmap_file = self.config["SEQMAP_FILE"]
        else:
            if self.config["SEQMAP_FOLDER"] is None:
                seqmap_file = os.path.join(self.config['GT_FOLDER'], 'seqmaps', gt_set + '.txt')
            else:
                seqmap_file = os.path.join(self.config["SEQMAP_FOLDER"], gt_set + '.txt')
        if not os.path.isfile(seqmap_file):
            raise Exception('no seqmap found: ' + os.path.basename(seqmap_file))
        with open(seqmap_file) as fp:
            reader = csv.reader(fp)
            for i, row in enumerate(reader):
                if i == 0 or row[0] == '':
                    continue
                seq = row[0]
                self.seq_list.append(seq)
                ini_file = os.path.join(self.gt_fol, seq, 'seqinfo.ini')
                if not os.path.isfile(ini_file):
                    raise Exception('ini file does not exist: ' + seq + '/' + os.path.basename(ini_file))
                ini_data = configparser.ConfigParser()
                ini_data.read(ini_file)
                self.seq_lengths[seq] = int(ini_data['Sequence']['seqLength'])
                if not self.data_is_zipped:
                    curr_file = os.path.join(self.gt_fol, seq, 'gt', 'gt.txt')
                    if not os.path.isfile(curr_file):
                        raise Exception('GT file not found: ' + seq + '/gt/' + os.path.basename(curr_file))
        if self.data_is_zipped:
            curr_file = os.path.join(self.gt_fol, 'data.zip')
            if not os.path.isfile(curr_file):
                raise Exception('GT file not found: ' + os.path.basename(curr_file))

        # Get trackers to eval
        if self.config['TRACKERS_TO_EVAL'] is None:
            self.tracker_list = os.listdir(self.tracker_fol)
        else:
            self.tracker_list = self.config['TRACKERS_TO_EVAL']
        for tracker in self.tracker_list:
            if self.data_is_zipped:
                curr_file = os.path.join(self.tracker_fol, tracker, self.tracker_sub_fol + '.zip')
                if not os.path.isfile(curr_file):
                    raise Exception('Tracker file not found: ' + tracker + '/' + os.path.basename(curr_file))
            else:
                for seq in self.seq_list:
                    curr_file = os.path.join(self.tracker_fol, tracker, self.tracker_sub_fol, seq + '.txt')
                    if not os.path.isfile(curr_file):
                        raise Exception(
                            'Tracker file not found: ' + tracker + '/' + self.tracker_sub_fol + '/' + os.path.basename(
                                curr_file))
                    tra_curr_file = os.path.join(self.tracker_fol, tracker, self.tracker_sub_fol, seq + '_track.txt')
                    if not os.path.isfile(tra_curr_file):
                        self.exists_tra_file = False
                    


    def _load_raw_file(self, tracker, seq, is_gt):
        """Load a file (gt or tracker) in the MOT Challenge 2D box format

        If is_gt, this returns a dict which contains the fields:
        [gt_ids, gt_classes] : list (for each timestep) of 1D NDArrays (for each det).
        [gt_dets, gt_crowd_ignore_regions]: list (for each timestep) of lists of detections.
        [gt_extras] : list (for each timestep) of dicts (for each extra) of 1D NDArrays (for each det).

        if not is_gt, this returns a dict which contains the fields:
        [tracker_ids, tracker_classes, tracker_confidences] : list (for each timestep) of 1D NDArrays (for each det).
        [tracker_dets]: list (for each timestep) of lists of detections.
        """
        # File location
        if self.data_is_zipped:
            if is_gt:
                zip_file = os.path.join(self.gt_fol, 'data.zip')
            else:
                zip_file = os.path.join(self.tracker_fol, tracker, self.tracker_sub_fol + '.zip')
            file = seq + '.txt'
            tra_file = seq + '_track.txt' 
        else:
            zip_file = None
            if is_gt:
                file = os.path.join(self.gt_fol, seq, 'gt', 'gt.txt')
                tra_file = os.path.join(self.gt_fol, seq, 'TRA', 'man_track.txt')
            else:
                file = os.path.join(self.tracker_fol, tracker, self.tracker_sub_fol, seq + '.txt')
                tra_file = os.path.join(self.tracker_fol, tracker, self.tracker_sub_fol, seq + '_track.txt')

        # Load raw data from text file
        read_data, ignore_data = self._load_simple_text_file(file, is_zipped=self.data_is_zipped, zip_file=zip_file)

        # Convert data to required format
        num_timesteps = self.seq_lengths[seq]
        data_keys = ['ids', 'classes', 'dets']
        raw_data = {key: [None] * num_timesteps for key in data_keys}
        for t in range(num_timesteps):
            time_key = str(t+1)

            if time_key in read_data.keys():
                time_data = np.asarray(read_data[time_key], dtype=np.float)
                
                raw_data['dets'][t] = np.atleast_2d(time_data[:, 2:6])
                raw_data['ids'][t] = np.atleast_1d(time_data[:, 1]).astype(int)
                if time_data.shape[1] >= 8:
                    raw_data['classes'][t] = np.atleast_1d(time_data[:, 7]).astype(int)
                else:
                    if not is_gt:
                        raw_data['classes'][t] = np.ones_like(raw_data['ids'][t])
                    else:
                        raise Exception(
                            'GT data is not in a valid format, there is not enough rows in seq %s, timestep %i.' % (
                                seq, t))
            else:
                raw_data['dets'][t] = np.empty((0, 4))
                raw_data['ids'][t] = np.empty(0).astype(int)
                raw_data['classes'][t] = np.empty(0).astype(int)
               
        if is_gt:
            key_map = {'ids': 'gt_ids',
                       'classes': 'gt_classes',
                       'dets': 'gt_dets'}
        else:
            key_map = {'ids': 'tracker_ids',
                       'classes': 'tracker_classes',
                       'dets': 'tracker_dets'}
        for k, v in key_map.items():
            raw_data[v] = raw_data.pop(k)
        raw_data['num_timesteps'] = num_timesteps
        raw_data['seq'] = seq

        if self.exists_tra_file:
            if is_gt:
                raw_data['gt_tracks'] = self.loadTrackFile(is_gt,tra_file)
            else:
                raw_data['tracker_tracks'] = self.loadTrackFile(is_gt, tra_file)

        raw_data['do_mitosis'] = self.do_mitosis    
        return raw_data

    @_timing.time
    def get_preprocessed_seq_data(self, raw_data, cls):
        """ Preprocess data for a single sequence for a single class ready for evaluation.
        Inputs:
             - raw_data is a dict containing the data for the sequence already read in by get_raw_seq_data().
             - cls is the class to be evaluated.
        Outputs:
             - data is a dict containing all of the information that metrics need to perform evaluation.
                It contains the following fields:
                    [num_timesteps, num_gt_ids, num_tracker_ids, num_gt_dets, num_tracker_dets] : integers.
                    [gt_ids, tracker_ids, tracker_confidences]: list (for each timestep) of 1D NDArrays (for each det).
                    [gt_dets, tracker_dets]: list (for each timestep) of lists of detections.
                    [similarity_scores]: list (for each timestep) of 2D NDArrays.
        Notes:
            General preprocessing (preproc) occurs in 4 steps. Some datasets may not use all of these steps.
                1) Extract only detections relevant for the class to be evaluated (including distractor detections).
                2) Match gt dets and tracker dets. Remove tracker dets that are matched to a gt det that is of a
                    distractor class, or otherwise marked as to be removed.
                3) Remove unmatched tracker dets if they fall within a crowd ignore region or don't meet a certain
                    other criteria (e.g. are too small).
                4) Remove gt dets that were only useful for preprocessing and not for actual evaluation.
            After the above preprocessing steps, this function also calculates the number of gt and tracker detections
                and unique track ids. It also relabels gt and tracker ids to be contiguous and checks that ids are
                unique within each timestep.

        MOT Challenge:
            In MOT Challenge, the 4 preproc steps are as follow:
                1) There is only one class (pedestrian) to be evaluated, but all other classes are used for preproc.
                2) Predictions are matched against all gt boxes (regardless of class), those matching with distractor
                    objects are removed.
                3) There is no crowd ignore regions.
                4) All gt dets except pedestrian are removed, also removes pedestrian gt dets marked with zero_marked.
        """
        cls_id = -1 #self.class_name_to_class_id[cls] 

        data_keys = ['gt_ids', 'tracker_ids', 'gt_dets', 'tracker_dets', 'similarity_scores']
        data = {key: [None] * raw_data['num_timesteps'] for key in data_keys}
        unique_gt_ids = []
        unique_tracker_ids = []
        num_gt_dets = 0
        num_tracker_dets = 0
  
        for t in range(raw_data['num_timesteps']):

            # Get all data
            gt_ids = raw_data['gt_ids'][t]
            gt_dets = raw_data['gt_dets'][t]
            gt_classes = raw_data['gt_classes'][t]

            tracker_ids = raw_data['tracker_ids'][t]
            tracker_dets = raw_data['tracker_dets'][t]
            tracker_classes = raw_data['tracker_classes'][t]
            similarity_scores = raw_data['similarity_scores'][t]

            data['tracker_ids'][t] = tracker_ids 
            data['tracker_dets'][t] = tracker_dets 
            data['gt_ids'][t] = gt_ids
                  
            data['gt_dets'][t] = gt_dets
            data['similarity_scores'][t] = similarity_scores

            unique_gt_ids += list(np.unique(data['gt_ids'][t]))
            unique_tracker_ids += list(np.unique(data['tracker_ids'][t]))
            num_tracker_dets += len(data['tracker_ids'][t])
            num_gt_dets += len(data['gt_ids'][t])

        # Re-label IDs such that there are no empty IDs
        if len(unique_gt_ids) > 0:
            unique_gt_ids = np.unique(unique_gt_ids)
            gt_id_map = np.nan * np.ones((np.max(unique_gt_ids) + 1))
            gt_id_map[unique_gt_ids] = np.arange(len(unique_gt_ids))
            for t in range(raw_data['num_timesteps']):
                if len(data['gt_ids'][t]) > 0:
                    data['gt_ids'][t] = gt_id_map[data['gt_ids'][t]].astype(np.int)
        if len(unique_tracker_ids) > 0:
            unique_tracker_ids = np.unique(unique_tracker_ids)
            tracker_id_map = np.nan * np.ones((np.max(unique_tracker_ids) + 1))
            tracker_id_map[unique_tracker_ids] = np.arange(len(unique_tracker_ids))
            for t in range(raw_data['num_timesteps']):
                if len(data['tracker_ids'][t]) > 0:
                    data['tracker_ids'][t] = tracker_id_map[data['tracker_ids'][t]].astype(np.int)

        # Ensure that ids are unique per timestep.
        self._check_unique_ids(data)

        ## Assign the TRA dicts
        #self.exists_tra_file = False
        if self.exists_tra_file:
            data['gt_tracks'] = raw_data['gt_tracks']
            data['tracker_tracks'] = raw_data['tracker_tracks']
            data['do_mitosis'] = True 
        else:
            ## Create the dicts for TRA metric if CTC format file is not provided
            # parent id is marked as 0 for both gt and tracker 
            gt_tracks = {(key-1): [key-1, np.inf, 0, 0] for key in unique_gt_ids}
            tracker_tracks = {(key-1): [key-1, np.inf, 0, 0] for key in unique_tracker_ids}
            gt_tracks_list = {}; tracker_tracks_list = {}
            
            for t in range(raw_data['num_timesteps']):
                ## Create the gt dict for TRA metric
                for id in data['gt_ids'][t]:
                    gt_tracks[id][0] = id
                    gt_tracks[id][1] = min(t, gt_tracks[id][1])
                    gt_tracks[id][2] = max(t, gt_tracks[id][2])
                   
                ## Create the tracker dict for TRA metric
                for id in data['tracker_ids'][t]:
                    tracker_tracks[id][0] = id
                    tracker_tracks[id][1] = min(t, tracker_tracks[id][1])
                    tracker_tracks[id][2] = max(t, tracker_tracks[id][2])
                   
            for key, value in gt_tracks.items():
                gt_tracks_list[key] = self.Track(key, value[1], value[2], value[3])
            for key, value in tracker_tracks.items():
                tracker_tracks_list[key] = self.Track(key, value[1], value[2], value[3])
            data['gt_tracks'] = gt_tracks_list
            data['tracker_tracks'] = tracker_tracks_list
            data['do_mitosis'] = False ## if the TRA file does not exist 
                   
        # Record overview statistics.
        data['num_tracker_dets'] = num_tracker_dets
        data['num_gt_dets'] = num_gt_dets
        data['num_tracker_ids'] = len(unique_tracker_ids)
        data['num_gt_ids'] = len(unique_gt_ids)
        data['num_timesteps'] = raw_data['num_timesteps']
        data['seq'] = raw_data['seq']
       
        return data

    def _calculate_similarities(self, gt_dets_t, tracker_dets_t):
        similarity_scores = self._calculate_box_ious(gt_dets_t, tracker_dets_t, box_format='xywh')
        return similarity_scores


    class Track:
        def __init__(self, id, begin, end, parent ):
            self.m_id = id
            self.m_begin = begin
            self.m_end = end
            self.m_parent = parent
            
            def __repr__(self):
                return "ID: %d, Begin: %d, End: %d, Parent: %d )" %(self.m_id, self.m_begin, self.m_end, self.m_parent)
    
    def loadTrackFile(self, is_gt,file):
        track_list = {}
        df = pd.read_csv(
            file,
            sep=" ",
            index_col=[0],
            skipinitialspace=True,
            header=None,
            names=['Id', 'Begin', 'End', 'Parent'],
            engine='python'
        )
      
        for index, row in df.iterrows():
            track_list[index-1] = self.Track(index-1, row['Begin']-1, row['End']-1, row['Parent']-1)
        return track_list
