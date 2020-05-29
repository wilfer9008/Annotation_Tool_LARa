'''
Created on 22.11.2019

@author: Erik Altermann
@email: Erik.Altermann@tu-dortmund.de

'''
import numpy as np
import os
from torch.utils.data import Dataset
import torch


class Data_processor():
    """Data Processor loads,stores and processes the unlabeled data and labels


    Attributes
    -------
    body_segments : dict
        a mapping of an int i to a str with the i-th bodysegments' name 

    body_segments_reversed : dict
        a mapping of a str bodysegments' name to its index i

    colors : dict
        a mapping of a few str literals to RGBA tupels 

    skeleton_colors : tuple
        a 44-tupel with 2 colors for each of the 22 body segments.    

    classes: list
        a list of strings thats loaded from classes.txt

    attributes : list
        a list of strings thats loaded from attributes.txt

    mocap_data : numpy array
        after initialization: contains the mocap_data normalized to the lowerback

    number_samples : int
        after initialization: the number of samples/frames/rows in the mocap_data

    frames : numpy array
        after initialization: a numpy array with the shape (number_samples,44,3)
	this array contains the coordinates for drawing the mocap skeleton at a certain timeframe

    file_name : str
        after initialization: a string with the name of the unlabeled mocap data file

    windows : list
        after initialization: a list of 4-tuples that store the label information    
    """



    body_segments= {
            -1:'none',
            0:'head',
            1:'head end',
            2:'L collar',       12:'R collar',
            6:'L humerus',      16:'R humerus',
            3:'L elbow',        13:'R elbow',
            9:'L wrist',        19:'R wrist',
            10:'L wrist end',   20:'R wrist end',
            11:'lower back',    
            21:'root',
            4:'L femur',        14:'R femur',
            7:'L tibia',        17:'R tibia',
            5:'L foot',         15:'R foot',
            8:'L toe',          18:'R toe'}
    
    body_segments_reversed = {}
    for k,v in body_segments.items():
            body_segments_reversed[v] = k

    colors = {'r': (1,0,0,1), 'g': (0,1,0,1), 'b': (0,0,1,1), 'y': (1,1,0,1)}

    #each bodysegmentline needs 2 colors because each has a start and end. 
    #different colors on each end result in a gradient
    skeleton_colors = (
        colors['b'],colors['b'], #head
        colors['b'],colors['b'], #head end
        colors['b'],colors['b'], #L collar
        colors['g'],colors['g'], #L elbow
        colors['r'],colors['r'], #L femur
        colors['r'],colors['r'], #L foot
        colors['g'],colors['g'], #L humerus
        colors['r'],colors['r'], #L tibia
        colors['r'],colors['r'], #L toe
        colors['g'],colors['g'], #L wrist
        colors['g'],colors['g'], #L wrist end
        colors['b'],colors['b'], #lower back
        colors['b'],colors['b'], #R collar
        colors['g'],colors['g'], #R elbow
        colors['r'],colors['r'], #R femur
        colors['r'],colors['r'], #R foot
        colors['g'],colors['g'], #R humerus
        colors['r'],colors['r'], #R tibia
        colors['r'],colors['r'], #R toe
        colors['g'],colors['g'], #R wrist
        colors['g'],colors['g'], #R wrist end
        colors['b'],colors['b'], #root
        )

    with open('class.txt', 'r') as f:
        classes = f.read().split(',')

    with open('attrib.txt', 'r') as f:
        attributes = f.read().split(',')
    

    def __init__(self,filePath:str,backupPath:str=None, annotated:bool=False):
        """Initializes Data_proccessor
                
        Loads the motioncapture data,
        Normalizes the data if needed,
        Calculates the frames for drawing the mocap data from the data.

        If a backup path was provided it loads window information from it otherwise
        it opens a new backup file
        
        Arguments: 
        --------
        filePath : str
            path to the unlabeled motion capture data        
        backupPath : str (optional)
            path to a backup.txt with label information
        annotated : bool (optional)
            (default) False if the loaded file is not normalized
            True if the data was normalized.
        """
        
        if not annotated:
            self.mocap_data = self.load_data(filePath,True)
        else: 
            self.mocap_data = self.load_data(filePath,False)
            directory, data_name = os.path.split(filePath)
            #print(data_name)
            window_name_parts = data_name.split('_')[:5]
            #print(window_name_parts)
            window_name_parts.append("windows.txt")
            #print(window_name_parts)
            window_name = window_name_parts[0]
            
            for part in window_name_parts[1:]:
                window_name += "_"+part
            window_path = directory+os.sep+window_name
            self.backup = open(window_path, 'r+t')
            self.windows = self.load_backup()
            
        self.number_samples = self.mocap_data.shape[0]
        self.frames = self.calculate_frames()
        self.windows_1 = None
        self.windows_2 = None
        self.windows_3 = None
        self.attr_windows = None
        self.file_name = os.path.split(filePath)[1] # Removes the path and keeps the file's name.
        #S03_P10_R27.csv
        #or
        #S03_P10_R27_A20_N01_norm_data.csv
        name_fragments = self.file_name.split('_')
        
        raw_data_name = name_fragments[0]
        #for fragment in name_fragments[1:3]: 
        for fragment in name_fragments[1:]: 
            raw_data_name += "_"+fragment
        raw_data_name += ".csv" #S03_P10_R27.csv
        
        self.file_name = raw_data_name
        
        if backupPath is None:
            #If no backup path was provided this takes the data file's name without its extension and 
            #adds '_backup.txt' to name the new backup file which is then created.
            self.backup = open('backups'+os.sep+self.file_name.split('.')[0]+'_backup.txt', 'wt') #TODO: figure out how to change this with the setting
            if annotated:
                self.saveWindows()
            else:
                self.windows = []
            
        else:
            self.backup = open(backupPath, 'r+t')
            self.windows = self.load_backup()
        
        
        
        
    def load_data(self,path:str,normalize:bool=True) -> np.array:
        """loads and normalizes mocap data stored in path
        
        Arguments:
        ---------
        path : str
            path to the unlabeled motion capture data
        normalize : bool (optional)
            (default) True if the file needs to be normalized.
            False if it already is normalized
        ---------
        
        Returns: 
        ---------
        array : numpy.array
            2D array with normalized motioncapture data.
            1st dimension is the time
            2nd dimension is the location and rotation data of each bodysegment
            shape should be (t,132) with t as number of timesteps in the data
        ---------
        """
        if normalize:
            array = np.loadtxt(path, delimiter=',', skiprows=5)
            array = array[:,2:]
            array = self.normalize_data(array)
        else:
            array = np.loadtxt(path, delimiter=',', skiprows=1)
            array = array[:,2:]
            
        return array

    def normalize_data(self,array:np.array) -> np.array:
        """normalizes the mocap data array
        
        The data gets normalized by subtraction of the lower backs data from every bodysegment.
        That way the lowerback is in the origin.
        
        Arguments:
        ---------
        array : numpy.array
            2D array with normalized motioncapture data.
            1st dimension is the time
            2nd dimension is the location and rotation data of each bodysegment
            shape should be (t,132) with t as number of timesteps in the data
        ---------
        
        Returns: 
        ---------
        array : numpy.array
            2D array with normalized motioncapture data.
            1st dimension is the time
            2nd dimension is the location and rotation data of each bodysegment
            shape should be (t,132) with t as number of timesteps in the data
        ---------
        """
        
        normalizing_vector = array[:,66:72]#66:72 are the columns for lowerback
        for _ in range(21):
            normalizing_vector = np.hstack((normalizing_vector,array[:,66:72]))    
        array = np.subtract(array,normalizing_vector) 
        return array
    
    def calculate_skeleton(self,frame_index:int) -> np.array:
        """Calculates the lines indicating positions of bodysegments at a single timestep 
        
        Arguments:
        ---------
        frame_index : int
            an integer between 0 and self.number_samples
            denotes the frame/timestep at which the skeleton should be calculated
        ---------
        
        Returns: 
        ---------
        array : numpy.array
            2D array with shape (44,3).
            Contains 44 3-Tupels or 3D coordinates.
            Each bodysegment gets 2 coordinates, for a start and  an end point.
            There are 22 bodysegments.
        ---------
        
        """
        
        frame = self.mocap_data[frame_index, :] #All the data at the time of frame_index
        
        #Extraction of Translational data for each bodysegment (source)
        tx = []
        ty = []
        tz = []
        for i in range(22):
            tx.append( frame[i*6 +3 ])
            ty.append( frame[i*6 +4 ])
            tz.append( frame[i*6 +5 ])
        
        #Extraction of Translational data for each bodysegment (target)
        tu = [] #corresponds to x coordinates
        tv = [] #corresponds to y coordinates
        tw = [] #corresponds to z coordinates
        offset = 3
        for coords in [tu,tv,tw]:               #    xyz        ->     uvw
            coords.append(frame[ 2*6+offset])   # 0   head      -> l collar/rcollar 
            coords.append(frame[ 0*6+offset])   # 1   head end  -> head
            coords.append(frame[11*6+offset])   # 2 l collar    -> lowerback
            coords.append(frame[ 6*6+offset])   # 3 l elbow     -> l humerus
            coords.append(frame[21*6+offset])   # 4 l femur     -> root
            coords.append(frame[ 7*6+offset])   # 5 l foot      -> l tibia
            coords.append(frame[ 2*6+offset])   # 6 l humerus   -> l collar
            coords.append(frame[ 4*6+offset])   # 7 l tibia     -> l femur
            coords.append(frame[ 5*6+offset])   # 8 l toe       -> l foot
            coords.append(frame[ 3*6+offset])   # 9 l wrist     -> l elbow
            coords.append(frame[ 9*6+offset])   #10 l wrist end -> l wrist
            coords.append(frame[11*6+offset])   #11   lowerback -> lowerback
            coords.append(frame[11*6+offset])   #12 r collar    -> lowerback
            coords.append(frame[16*6+offset])   #13 r elbow     -> r humerus
            coords.append(frame[21*6+offset])   #14 r femur     -> root
            coords.append(frame[17*6+offset])   #15 r foot      -> r tibia
            coords.append(frame[12*6+offset])   #16 r humerus   -> r collar
            coords.append(frame[14*6+offset])   #17 r tibia     -> r femur
            coords.append(frame[15*6+offset])   #18 r toe       -> r foot
            coords.append(frame[13*6+offset])   #19 r wrist     -> r elbow
            coords.append(frame[19*6+offset])   #20 r wrist end -> r wrist
            coords.append(frame[11*6+offset])   #21   root      -> lowerback
            offset+=1
        
        #combine the 3 lists of source coordinates into a 3-tupel list
        txyz = list(zip(tx,ty,tz))
        #combine the 3 lists of target coordinates into a 3-tupel list
        tuvw = list(zip(tu,tv,tw))
        #append the coordinates from source and target alternatingly to a single list
        t_all = []
        for a,b in zip(txyz,tuvw):
            t_all.append(a)
            t_all.append(b)
        
        #convert the list into an array, convert millimeters to meters and return the result
        return np.array(t_all)/1000

    def calculate_frames(self) -> np.array:
        """Calculates the skeletonlines for each frame in the data
        
        Returns: 
        ---------
        frames : numpy.array
            3D array with shape (self.number_samples,44,3).
            Contains self.number_samples frames.
            Each frame contains 44 3-Tupels or 3D coordinates.
            Each bodysegment gets 2 coordinates, for a start and an end point.
            There are 22 bodysegments.
        ---------
        
        """
        frames = np.zeros((self.number_samples,44,3)) #dimensions are( frames, bodysegments, xyz)
    
        for frame_index in range(self.number_samples):
            frame = self.calculate_skeleton(frame_index)
            frames[frame_index,:,:] = frame
        return frames
    
    def saveWindow(self,start:int,end:int,class_index:int,attributes:list):
        """Saves a single window of label data to the backup file
        
        Arguments:
        ---------
        start : int
            frame_index where this label window starts.
        end : int
            frame_index where this label window ends. 
        class_index : int
            index of the class this window has.
        attributes : list
            list of 1's and 0's that show which attributes are present
        ---------
        """
        window = (start,end,class_index,attributes)
        self.windows.append(window)
        self.backup.write(str(window)+'\n')
        self.backup.flush()
    
    def changeWindow(self,window_index:int,start:int=None,end:int=None,class_index:int=None,attributes:list=None,save:bool=True):
        """Changes the labels of the window at window_index
        
        Arguments:
        ---------
        window_index : int
            Integer between 0 and the number of windows. 
            Which window to modify
        start : int
            the new start value of the modified window
        end : int
            the new end value of the modified window
        class_index : int (optional)
            the new class of the modified window
        attributes : list (optional)
            new list of 1's and 0's that show which attributes are present
        save : bool
            If true the method saves the changes directly to the file
            set to False if there is more than 1 window that needs to be modified
            Last modified window should have save = True or call saveWindows after all modifications
        ---------
        """
        
        #Changes the value of the window at window_index
        new_window = list(self.windows[window_index])
        if start is not None:
            new_window[0] = start
        if end is not None:
            new_window[1] = end        
        if class_index is not None:
            new_window[2] = class_index
        if attributes is not None:
            new_window[3] = attributes
        self.windows[window_index] = tuple(new_window)
        if save:
            self.saveWindows()
    
    def insertWindow(self,window_index,start:int,end:int,class_index:int,attributes:list,save:bool=True):
        """Saves a single window of label data to the backup file
        
        Arguments:
        ---------
        window_index : int
            Integer between 0 and the number of windows.
            where to insert the new window.
        start : int
            frame_index where this label window starts.
        end : int
            frame_index where this label window ends. 
        class_index : int
            index of the class this window has.
        attributes : list
            list of 1's and 0's that show which attributes are present
        save : bool
            If true the method saves the changes directly to the file
            set to False if there is more than 1 window that needs to be modified
            Last modified window should have save = True or call saveWindows after all modifications
        ---------
        """
        window = (start,end,class_index,attributes)
        self.windows.insert(window_index, window)
        if save:
            self.saveWindows()
            
    def deleteWindow(self,window_index:int,save:bool=True):
        """Deletes the window with the given index
        
        Arguments:
        ----------
        window_index : int
            Integer between 0 and the number of windows. 
            Which window to delete
        save : bool
            If true the method saves the changes directly to the file
            set to False if there is more than 1 window that needs to be modified
            Last modified window should have save = True or call saveWindows after all modifications
        """
        self.windows.pop(window_index)
        if save:
            self.saveWindows()
        
    def saveWindows(self):
        """Rewrites all windows into the backup
        
        Should be called when at least one already existing window gets changed.
        """
        #rewrites the backup file with the new change
        self.backup.close()
        self.backup = open('backups'+os.sep+self.file_name.split('.')[0]+'_backup.txt', 'wt') #TODO: figure out how to change this with the setting
        for window in self.windows:
            self.backup.write(str(window)+'\n')
            self.backup.flush()
        
    def createBackup(self,directory:str,suffix:str='backup') -> str:
        """Saves all saves windows at the specified directory 
        
        Saves the window labels at the directory in a .txt-file with 
        the same name as the originally opened file with the suffix appended to it.
        
        Arguments:
        ----------
        directory : str
            path to the directory where the backup should be saved
        suffix : str (optional)
            suffix that gets appended to the original file name.
            default : 'backup'
        ----------
        """
        
        backup_name = self.file_name.split('.')[0]+"_"+ suffix+ ".txt"
        path = directory+os.sep+backup_name
        backup = open(path, 'wt')
        for window in self.windows:
            backup.write(str(window)+'\n')
        backup.close()
        return "Backup successfully created!"
        
    
    #TODO: move path to backup into the parameters for this method
    def load_backup(self) -> list: 
        """Loads the windows from an opened file
        
        """
        
        windows = []
        
        lines = self.backup.readlines()
        for line in lines:
            #print (line[:-1])
            window = eval(line[:-1])
            windows.append(window)
        return windows
        
    def saveResults(self,directory:str,annotatorID:int,tries:int):
        """Saves the finished labels, normalized data, and windows in 3 files in the provided directory
        
        Arguments:
        ----------
        directory : str
            path to the directory where the results should be saved.
        annotatorID : int
            ID of the person annonating the dataset.
        tries : int
            number indicating how often the currently worked on file 
            was beeing annotated by the same annotator.
        ----------
        
        """
        annotator_suffix = '_A'+'%0.2d' %(int(annotatorID))+'_N'+'%0.2d' %(int(tries))
        
        #--------------------------------------------------------------------------------------
        #First file with class column followed by attribute columns
        #--------------------------------------------------------------------------------------
        header = "class"
        for attribute in self.attributes:
            header += ","+attribute
        
        data = np.zeros((self.number_samples, self.attributes.__len__()+1)) #All attributes +1
        for start,end,class_index,attributes in self.windows:
            for i in range(start,end):
                data[i,0] = class_index
                for j, attribute in enumerate(attributes):
                    data[i,j+1] = attribute
        
        file_name = self.file_name.split('.')[0] +annotator_suffix +"_labels.csv"
        np.savetxt(directory+os.sep+file_name, data, delimiter=',', header=header, comments='')
        
        #--------------------------------------------------------------------------------------
        #Second file normalized data: sample column, class label column, head rx,head ry, etc.
        #--------------------------------------------------------------------------------------
        header = "sample,classlabel"
        for i in range(22):
            bodysegment = self.body_segments[i]
            for coordinate in ['RX','RY','RZ','TX','TY','TZ']:
                header += "," + bodysegment+'_'+coordinate
        
        data = np.zeros((self.number_samples, 2+self.mocap_data.shape[1])) #samples+class+ datacolumns
        for start,end,class_index,attributes in self.windows:
            for i in range(start,end):
                data[i,1] = class_index
        data[:,0] = range(self.number_samples)
        data[:,2:] = self.mocap_data[:,:]   
                
                
        file_name = self.file_name.split('.')[0] +annotator_suffix+ "_norm_data.csv"
        np.savetxt(directory+os.sep+file_name, data, delimiter=',', header=header, comments='')
        
        #--------------------------------------------------------------------------------------
        #Third file: the backup with the windows.
        #--------------------------------------------------------------------------------------
        
        self.createBackup(directory, annotator_suffix[1:]+'_windows')
        
        
    def close(self):
        self.backup.close()
        
        
class Sliding_window_dataset(Dataset):
    """Segments data using a sliding window approach and provides methods for iteration
    
    Stored segments have 4 Dimensions for quick use in torch models
    """
    def __init__(self,data:np.array,window_length:int,window_step:int):
        """Initializes the dataset
        
        Arguments:
        ----------
        data : numpy array
            The data that needs to be segmented
        window_length : int
            The length of each segment
        window_step : int
            The stride/step between segments
        """
        
        self.data = torch.from_numpy(np.delete(data[np.newaxis,np.newaxis,:,:], range(66,72), 3)).float()
        #print(self.data.shape)
        self.window_length = window_length
        self.window_step = window_step
    
    def __getitem__(self, segment_index:int):
        """Returns the segment with the given segment_index
        
        Arguments:
        ----------
        segment_index : int
            index of the segment that will be returned
        ----------
        
        Returns:
        ----------
        segment : 4D torch array
            segment at provided index
        ----------
        
        """
        
        lowerbound, upperbound = self.__range__(segment_index)
        return self.data[:,:,lowerbound:upperbound,:]            
    
    def __len__(self):
        """Returns the length of the dataset/number of segments. """
        
        return int((self.data.shape[2]-self.window_length)/self.window_step) +1
    
    def __range__(self,segment_index):
        """Returns the range of the segment at segment_index
        
        Arguments:
        ----------
        segment_index : int
            index of the segment, whose range will be returned
        ----------
        
        Returns:
        ----------
        range : 2-tuple
            (lowerbound,upperbound)         
        ----------
        """
        
        #if segment_index < self.__len__():
        lowerbound = self.window_step * segment_index
        upperbound = lowerbound + self.window_length
        #else:
        #    upperbound = self.data.shape[2]
        #    lowerbound = upperbound - self.window_length
        return (lowerbound,upperbound)
    
class Labeled_sliding_window_dataset(Sliding_window_dataset):
    """Expands the Sliding_window_dataset to keep track of label information"""
    
    def __init__(self,data,window_length,window_step):
        """Initializes the dataset
        
        Arguments:
        ----------
        data : numpy array
            The data that needs to be segmented
        window_length : int
            The length of each segment
        window_step : int
            The stride/step between segments
        """
        
        super(Labeled_sliding_window_dataset, self).__init__(data,window_length,window_step)
        
        
        self.labels   = {}
        self.labels_2 = {}
        self.labels_3 = {}
        self.attr_labels = []
        #for i in range(self.data.shape[2]):
        #    self.labels[i] = []    
    
    
    def setlabel(self,segment_index,label,kind,label_place=1):
        """Saves labels for all frames in a single segment
            
        Arguments:
        ----------
        segment_index : int
            index of the segment, to which the labels belong
        label : int,tuple or list
            a single int denoting a class
            a list of 0's and 1's denoting attributes
        kind : string
            tells what type of label is provided
            options are 'class','attributes'
        label_place: int
            a value between 1-3 (inclusive) that shows where in the top3 a prediction is            
        """
        
        no_attributes = [0 for _ in range(Data_processor.attributes.__len__())]
        no_attributes[-1] = 1 #All attributes are 0 except for Error attribute
        none_class = Data_processor.classes.__len__()-1
        
        if label_place == 1:
            labels = self.labels
        elif label_place == 2:
            labels = self.labels_2
        elif label_place == 3:
            labels = self.labels_3
        
        #print(kind)
        
        if kind == 'class':
            class_label = int(label)
            attributes = tuple(no_attributes)
        elif kind == 'attributes':
            class_label = none_class
            attributes = tuple(label)
        elif kind == 'both': 
            class_label = int(label[0])
            attributes = tuple(label[1])
        else:
            raise Exception
            
        lowerbound,upperbound = self.__range__(segment_index)
        
        for j in range(lowerbound,upperbound,self.window_step):
            window_start = j
            window_end = j+self.window_step
            
            
            window = (window_start,window_end,class_label,attributes)
            
            if j not in labels.keys():
                labels[j] = [window]
            else:
                #print(labels[0])
                #labels[0][0]
                #labels[0][0][1]
                
                #labels[key] is a list of all possible labels for window starting at key
                #labels[key][0] is the first window.
                #labels[key][1] is the windows end frame. 
                #    the end is the same for all windows at labels[key] 
                for end in [labels[key][0][1] for key in labels.keys()]: #For every end that lies between the current window
                    if window_start < end and end < window_end:
                        window = (window_start,end,class_label,attributes)
                        labels[j].append(window)
                        window = (end,window_end,class_label,attributes)
                        labels[j].append(window)
                        window = None
                        break
                        
                if window is not None:
                    labels[j].append(window)
                    
        
        unlabeled = self.window_length%self.window_step
        
        if unlabeled>0:
            window_start = upperbound-unlabeled
            window_end = upperbound
            window = (window_start,window_end,class_label,attributes)
            
            if j not in labels.keys():
                labels[j] = [window]
            else:
                labels[j].append(window)
            
    def set_top3_labels(self,segment_index,label,kind,metrics=None):
        """Saves labels for all frames in a single segment
            
        Arguments:
        ----------
        segment_index : int
            index of the segment, to which the labels belong
        label: 
        
        kind : string
            tells what type of label is provided
            options are 'class','attributes'
            
        """
        
        if kind == 'class':
            for i in range(3):
                self.setlabel(segment_index, label[:,i].item(), kind, i+1)
            
            
            
        elif kind == 'attributes':
            #print(label.shape)
            self.attr_labels.append( list(label[0].detach().numpy()))
            
            predictions = metrics.efficient_distance(label)
            predictions = metrics.acc_metric(None,predictions)[1][:,0:3]
            self.set_top3_labels(segment_index, predictions, 'class')
            
        else:
            raise Exception
        
                
    def evaluate(self,function = lambda lst: max(set(lst), key=lst.count)):
        """Evaluates all possible saved labels using function to choose a definitive label
        
        Arguments:
        ----------
        function: function (optional)
            any function that recieves a list and returns a single element from it.
            default: mode of a list        
        ----------
        
        Returns:
        ----------
        windows: list
            this list is formated the same way as the windows list in the Data_processor
        ----------
        """
        
        windows = []
        
        for lst in self.labels.values():
            start,end,class_label,attributes = function(lst)
            window = (start,end,class_label,list(attributes))
            windows.append(window)
            
        return windows
        
    def evaluate_top3(self,function = lambda lst: max(set(lst), key=lst.count)):
        windows_1 = []
        windows_2 = []
        windows_3 = []
        
        for labels, windows in zip([self.labels,self.labels_2,self.labels_3],[windows_1,windows_2,windows_3]):
            for lst in labels.values():
                start,end,class_label,attributes = function(lst)
                window = (start,end,class_label,list(attributes))
                windows.append(window)
            
        return (windows_1,windows_2,windows_3)
        
        
    
