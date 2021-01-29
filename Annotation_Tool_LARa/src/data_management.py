"""
Created on 22.11.2019

@author: Erik Altermann
@email: Erik.Altermann@tu-dortmund.de

"""
import os

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import Dataset

import global_variables as g

NORM_MAX_THRESHOLDS = [392.85, 345.05, 311.295, 460.544, 465.25, 474.5, 392.85,
                       345.05, 311.295, 574.258, 575.08, 589.5, 395.81, 503.798,
                       405.9174, 322.9, 331.81, 338.4, 551.829, 598.326, 490.63,
                       667.5, 673.4, 768.6, 560.07, 324.22, 379.405, 193.69,
                       203.65, 159.297, 474.144, 402.57, 466.863, 828.46, 908.81,
                       99.14, 482.53, 381.34, 386.894, 478.4503, 471.1, 506.8,
                       420.04, 331.56, 406.694, 504.6, 567.22, 269.432, 474.144,
                       402.57, 466.863, 796.426, 863.86, 254.2, 588.38, 464.34,
                       684.77, 804.3, 816.4, 997.4, 588.38, 464.34, 684.77,
                       889.5, 910.6, 1079.7, 392.0247, 448.56, 673.49, 322.9,
                       331.81, 338.4, 528.83, 475.37, 473.09, 679.69, 735.2,
                       767.5, 377.568, 357.569, 350.501, 198.86, 197.66, 114.931,
                       527.08, 412.28, 638.503, 691.08, 666.66, 300.48, 532.11,
                       426.02, 423.84, 467.55, 497.1, 511.9, 424.76, 348.38,
                       396.192, 543.694, 525.3, 440.25, 527.08, 412.28, 638.503,
                       729.995, 612.41, 300.33, 535.94, 516.121, 625.628, 836.13,
                       920.7, 996.8, 535.94, 516.121, 625.628, 916.15, 1009.5,
                       1095.6, 443.305, 301.328, 272.984, 138.75, 151.84, 111.35]

NORM_MIN_THRESHOLDS = [-382.62, -363.81, -315.691, -472.2, -471.4, -152.398,
                       -382.62, -363.81, -315.691, -586.3, -581.46, -213.082,
                       -400.4931, -468.4, -409.871, -336.8, -336.2, -104.739,
                       -404.083, -506.99, -490.27, -643.29, -709.84, -519.774,
                       -463.02, -315.637, -405.5037, -200.59, -196.846, -203.74,
                       -377.15, -423.992, -337.331, -817.74, -739.91, -1089.284,
                       -310.29, -424.74, -383.529, -465.34, -481.5, -218.357,
                       -442.215, -348.157, -295.41, -541.82, -494.74, -644.24,
                       -377.15, -423.992, -337.331, -766.42, -619.98, -1181.528,
                       -521.9, -581.145, -550.187, -860.24, -882.35, -645.613,
                       -521.9, -581.145, -550.187, -936.12, -982.14, -719.986,
                       -606.395, -471.892, -484.5629, -336.8, -336.2, -104.739,
                       -406.6129, -502.94, -481.81, -669.58, -703.12, -508.703,
                       -490.22, -322.88, -322.929, -203.25, -203.721, -201.102,
                       -420.154, -466.13, -450.62, -779.69, -824.456, -1081.284,
                       -341.5005, -396.88, -450.036, -486.2, -486.1, -222.305,
                       -444.08, -353.589, -380.33, -516.3, -503.152, -640.27,
                       -420.154, -466.13, -450.62, -774.03, -798.599, -1178.882,
                       -417.297, -495.1, -565.544, -906.02, -901.77, -731.921,
                       -417.297, -495.1, -565.544, -990.83, -991.36, -803.9,
                       -351.1281, -290.558, -269.311, -159.9403, -153.482, -162.718]


class DataProcessor:
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

    mocap_data : numpy array
        after initialization: contains the mocap_data normalized to the lowerback

    number_samples : int
        after initialization: the number of samples/frames/rows in the mocap_data

    frames : numpy array
        after initialization: a numpy array with the shape (number_samples,44,3)
        this array contains the coordinates for drawing the mocap skeleton at a certain timeframe

    file_name : str
        after initialization: a string with the name of the unlabeled mocap data file

    """

    body_segments = {
        -1: 'none',
        0: 'head',
        1: 'head end',
        2: 'L collar', 12: 'R collar',
        6: 'L humerus', 16: 'R humerus',
        3: 'L elbow', 13: 'R elbow',
        9: 'L wrist', 19: 'R wrist',
        10: 'L wrist end', 20: 'R wrist end',
        11: 'lower back',
        21: 'root',
        4: 'L femur', 14: 'R femur',
        7: 'L tibia', 17: 'R tibia',
        5: 'L foot', 15: 'R foot',
        8: 'L toe', 18: 'R toe'}

    body_segments_reversed = {v: k for k, v in body_segments.items()}

    colors = {'r': (1, 0, 0, 1), 'g': (0, 1, 0, 1), 'b': (0, 0, 1, 1), 'y': (1, 1, 0, 1)}

    # each bodysegmentline needs 2 colors because each has a start and end.
    # different colors on each end result in a gradient
    skeleton_colors = (
        colors['b'], colors['b'],  # head
        colors['b'], colors['b'],  # head end
        colors['b'], colors['b'],  # L collar
        colors['g'], colors['g'],  # L elbow
        colors['r'], colors['r'],  # L femur
        colors['r'], colors['r'],  # L foot
        colors['g'], colors['g'],  # L humerus
        colors['r'], colors['r'],  # L tibia
        colors['r'], colors['r'],  # L toe
        colors['g'], colors['g'],  # L wrist
        colors['g'], colors['g'],  # L wrist end
        colors['b'], colors['b'],  # lower back
        colors['b'], colors['b'],  # R collar
        colors['g'], colors['g'],  # R elbow
        colors['r'], colors['r'],  # R femur
        colors['r'], colors['r'],  # R foot
        colors['g'], colors['g'],  # R humerus
        colors['r'], colors['r'],  # R tibia
        colors['r'], colors['r'],  # R toe
        colors['g'], colors['g'],  # R wrist
        colors['g'], colors['g'],  # R wrist end
        colors['b'], colors['b'],  # root
    )

    def __init__(self, file_path: str, annotated: bool = False):
        """Initializes Data_proccessor
                
        Loads the motioncapture data,
        Normalizes the data if needed,
        Calculates the frames for drawing the mocap data from the data.

        If a backup path was provided it loads window information from it otherwise
        it opens a new backup file
        
        Arguments: 
        --------
        file_path : str
            path to the unlabeled motion capture data
        annotated : bool (optional)
            (default) False if the loaded file is not normalized
            True if the data was normalized.
        """

        self.mocap_data = self.load_data(file_path, not annotated)  # Normalize if data wasn't annotated beforehand
        self.number_samples = self.mocap_data.shape[0]
        self.frames = self.calculate_frames()

        file_name = os.path.split(file_path)[1]  # Removes the path and keeps the file's name.
        # L03_P10_R27.csv or L03_P10_R27_A20_N01_norm_data.csv
        name_fragments = file_name.split('.')[0].split('_')
        raw_data_name = name_fragments[0]
        for fragment in name_fragments[1:3]:
            # for fragment in name_fragments[1:]:
            raw_data_name += "_" + fragment
        # raw_data_name += ".csv" #S03_P10_R27.csv
        self.file_name = raw_data_name

    def load_data(self, path: str, normalize: bool = True) -> np.array:
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
            array = array[:, 2:]
            array = self.normalize_data(array)
        else:
            array = np.loadtxt(path, delimiter=',', skiprows=1)
            array = array[:, 2:]

        return array

    def normalize_data(self, array: np.array) -> np.array:
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

        normalizing_vector = array[:, 66:72]  # 66:72 are the columns for lowerback
        for _ in range(21):
            normalizing_vector = np.hstack((normalizing_vector, array[:, 66:72]))
        array = np.subtract(array, normalizing_vector)
        return array

    def calculate_skeleton(self, frame_index: int) -> np.array:
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

        frame = self.mocap_data[frame_index, :]  # All the data at the time of frame_index

        # Extraction of Translational data for each bodysegment (source)
        tx = []
        ty = []
        tz = []
        for i in range(22):
            tx.append(frame[i * 6 + 3])
            ty.append(frame[i * 6 + 4])
            tz.append(frame[i * 6 + 5])

        # Extraction of Translational data for each bodysegment (target)
        tu = []  # corresponds to x coordinates
        tv = []  # corresponds to y coordinates
        tw = []  # corresponds to z coordinates
        offset = 3
        for coords in [tu, tv, tw]:  # xyz        ->     uvw
            coords.append(frame[2 * 6 + offset])  # 0   head      -> l collar/rcollar
            coords.append(frame[0 * 6 + offset])  # 1   head end  -> head
            coords.append(frame[11 * 6 + offset])  # 2 l collar    -> lowerback
            coords.append(frame[6 * 6 + offset])  # 3 l elbow     -> l humerus
            coords.append(frame[21 * 6 + offset])  # 4 l femur     -> root
            coords.append(frame[7 * 6 + offset])  # 5 l foot      -> l tibia
            coords.append(frame[2 * 6 + offset])  # 6 l humerus   -> l collar
            coords.append(frame[4 * 6 + offset])  # 7 l tibia     -> l femur
            coords.append(frame[5 * 6 + offset])  # 8 l toe       -> l foot
            coords.append(frame[3 * 6 + offset])  # 9 l wrist     -> l elbow
            coords.append(frame[9 * 6 + offset])  # 10 l wrist end -> l wrist
            coords.append(frame[11 * 6 + offset])  # 11   lowerback -> lowerback
            coords.append(frame[11 * 6 + offset])  # 12 r collar    -> lowerback
            coords.append(frame[16 * 6 + offset])  # 13 r elbow     -> r humerus
            coords.append(frame[21 * 6 + offset])  # 14 r femur     -> root
            coords.append(frame[17 * 6 + offset])  # 15 r foot      -> r tibia
            coords.append(frame[12 * 6 + offset])  # 16 r humerus   -> r collar
            coords.append(frame[14 * 6 + offset])  # 17 r tibia     -> r femur
            coords.append(frame[15 * 6 + offset])  # 18 r toe       -> r foot
            coords.append(frame[13 * 6 + offset])  # 19 r wrist     -> r elbow
            coords.append(frame[19 * 6 + offset])  # 20 r wrist end -> r wrist
            coords.append(frame[11 * 6 + offset])  # 21   root      -> lowerback
            offset += 1

        # combine the 3 lists of source coordinates into a 3-tupel list
        txyz = list(zip(tx, ty, tz))
        # combine the 3 lists of target coordinates into a 3-tupel list
        tuvw = list(zip(tu, tv, tw))
        # append the coordinates from source and target alternatingly to a single list
        t_all = []
        for a, b in zip(txyz, tuvw):
            t_all.append(a)
            t_all.append(b)

        # convert the list into an array, convert millimeters to meters and return the result
        return np.array(t_all) / 1000

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
        frames = np.zeros((self.number_samples, 44, 3))  # dimensions are( frames, bodysegments, xyz)

        for frame_index in range(self.number_samples):
            frame = self.calculate_skeleton(frame_index)
            frames[frame_index, :, :] = frame
        return frames


class WindowProcessor:
    def __init__(self, file_path: str, annotated: bool, load_backup: bool = False):
        """Initializes LabelProcessor"""
        self.backup = None
        self.windows = []
        self.windows_1 = None
        self.windows_2 = None
        self.windows_3 = None

        file_name = os.path.split(file_path)[1]  # Removes the path and keeps the file's name.
        # L03_P10_R27.csv or L03_P10_R27_A20_N01_norm_data.csv
        name_fragments = file_name.split('.')[0].split('_')
        raw_data_name = name_fragments[0]
        for fragment in name_fragments[1:3]:
            # for fragment in name_fragments[1:]:
            raw_data_name += "_" + fragment
        # raw_data_name += ".csv" #S03_P10_R27.csv
        self.file_name = raw_data_name
        # print("self.file_name: "+self.file_name)
        backup_path = f'{g.settings["backUpPath"]}{os.sep}{self.file_name.split(".")[0]}_backup.txt'
        if not load_backup:
            if annotated:
                self.load_annotations(file_path)

            else:
                self.windows = []
                self.make_backup()
        else:
            try:
                self.windows = self.load_backup()
            except ValueError as _:
                if annotated:
                    self.load_annotations(file_path)
                else:
                    self.windows = []
                    self.make_backup()

    def load_annotations(self, file_path):
        directory, data_name = os.path.split(file_path)
        window_name_parts = data_name.split('_')[:5]
        window_name_parts.append("windows.txt")
        window_name = window_name_parts[0]

        for part in window_name_parts[1:]:
            window_name += "_" + part
        window_path = directory + os.sep + window_name
        self.windows = self.load_backup(window_path)
        self.make_backup()

    def save_window(self, start: int, end: int, class_index: int, attributes: list):
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
        window = (start, end, class_index, attributes)
        self.windows.append(window)
        self.backup.write(str(window) + '\n')
        self.backup.flush()

    def change_window(self, window_index: int, start: int = None, end: int = None, class_index: int = None,
                      attributes: list = None, save: bool = True):
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
            Last modified window should have save = True or call make_backup after all modifications
        ---------
        """

        # Changes the value of the window at window_index
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
            self.make_backup()

    def insert_window(self, window_index, start: int, end: int, class_index: int, attributes: list, save: bool = True):
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
            Last modified window should have save = True or call make_backup after all modifications
        ---------
        """
        window = (start, end, class_index, attributes)
        self.windows.insert(window_index, window)
        if save:
            self.make_backup()

    def delete_window(self, window_index: int, save: bool = True):
        """Deletes the window with the given index

        Arguments:
        ----------
        window_index : int
            Integer between 0 and the number of windows.
            Which window to delete
        save : bool
            If true the method saves the changes directly to the file
            set to False if there is more than 1 window that needs to be modified
            Last modified window should have save = True or call make_backup after all modifications
        """
        self.windows.pop(window_index)
        if save:
            self.make_backup()

    def make_backup(self):
        """Rewrites all windows into the backup

        Should be called when at least one already existing window gets changed.
        """
        # rewrites the backup file with the new change
        if self.backup is not None:
            self.backup.close()

        backup_path = f'{g.settings["backUpPath"]}{os.sep}{self.file_name.split(".")[0]}_backup.txt'
        self.backup = open(backup_path, 'wt')
        for window in self.windows:
            self.backup.write(str(window) + '\n')
            self.backup.flush()

    def save_windows(self, directory: str, suffix: str = 'windows') -> str:
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

        backup_name = self.file_name.split('.')[0] + "_" + suffix + ".txt"
        path = directory + os.sep + backup_name
        backup = open(path, 'wt')
        for window in self.windows:
            backup.write(str(window) + '\n')
        backup.close()
        return "Backup successfully created!"

    def load_backup(self, backup_path=None) -> list:
        """Loads the windows from an opened file

        """
        if backup_path is None:
            backup_path = f'{g.settings["backUpPath"]}{os.sep}{self.file_name.split(".")[0]}_backup.txt'

        if os.path.exists(backup_path):
            self.backup = open(backup_path, 'r+t')
        else:
            self.backup = open(backup_path, 'wt')

        lines = self.backup.readlines()
        windows = []
        for line in lines:
            # print (line[:-1])
            window = eval(line[:-1])
            windows.append(window)
            if type(window[3]) != type([]):
                raise ValueError

        return windows

    def close(self):
        self.backup.close()

    def load_predictions(self, directory: str, annotator_id: int):
        """Loads the top3 predictions of a network.

        Arguments:
        ----------
        directory : str
            path to the directory where the results are located.
        annotator_id : id under which the predictions are saved
        ----------

        """
        self.windows_1 = self.load_prediction(directory, annotator_id, 0)
        self.windows_2 = self.load_prediction(directory, annotator_id, 1)
        self.windows_3 = self.load_prediction(directory, annotator_id, 2)

    def load_prediction(self, directory: str, annotator_id: int, pred_id: int):
        """Loads the top3 predictions of a network.

        Arguments:
        ----------
        directory : str
            path to the directory where the results are located.
        annotator_id : id under which the predictions are saved
        pred_id : id of the prediction to load
        ----------

        """

        file_name = f"{self.file_name.split('.')[0]}_A{annotator_id:0>2}_N{pred_id:0>2}.txt"
        path = directory + os.sep + file_name
        if not os.path.exists(path):
            return None
        backup = open(path, 'rt')
        lines = backup.readlines()
        backup.close()
        prediction = []
        for line in lines:
            # print (line[:-1])
            window = eval(line[:-1])
            prediction.append(window)
        return prediction

    def save_predictions(self, directory: str, annotator_id: int):
        """Saves the top3 predictions of a network.

        Saves the predictions as different "tries" of the same Annotator
        using the different N## suffix.

        Arguments:
        ----------
        directory : str
            path to the directory where the results should be saved.
        annotator_id : id under which to save the predictions
        ----------

        """

        predictions = [self.windows_1, self.windows_2, self.windows_3]

        for pred_id, prediction in enumerate(predictions):
            file_name = f"{self.file_name.split('.')[0]}_A{annotator_id:0>2}_N{pred_id:0>2}.txt"
            path = directory + os.sep + file_name

            backup = open(path, 'wt')
            for window in prediction:
                backup.write(str(window) + '\n')
            backup.close()

    def save_results(self, directory: str, annotator_id: int, tries: int):
        """Saves the finished labels, normalized data, and windows in 3 files in the provided directory

        Arguments:
        ----------
        directory : str
            path to the directory where the results should be saved.
        annotator_id : int
            ID of the person annonating the dataset.
        tries : int
            number indicating how often the currently worked on file
            was beeing annotated by the same annotator.
        ----------

        """
        annotator_suffix = '_A' + '%0.2d' % (int(annotator_id)) + '_N' + '%0.2d' % (int(tries))

        # --------------------------------------------------------------------------------------
        # First file with class column followed by attribute columns
        # --------------------------------------------------------------------------------------
        header = "class"
        for attribute in g.attributes:
            header += "," + attribute

        data = np.zeros((g.data.number_samples, g.attributes.__len__() + 1))  # All attributes +1
        for start, end, class_index, attributes in self.windows:
            for i in range(start, end):
                data[i, 0] = class_index
                for j, attribute in enumerate(attributes):
                    data[i, j + 1] = attribute

        file_name = self.file_name.split('.')[0] + annotator_suffix + "_labels.csv"
        np.savetxt(directory + os.sep + file_name, data, delimiter=',', header=header, comments='')

        # --------------------------------------------------------------------------------------
        # Second file normalized data: sample column, class label column, head rx,head ry, etc.
        # --------------------------------------------------------------------------------------
        header = "sample,classlabel"
        for i in range(22):
            bodysegment = g.data.body_segments[i]
            for coordinate in ['RX', 'RY', 'RZ', 'TX', 'TY', 'TZ']:
                header += "," + bodysegment + '_' + coordinate

        data = np.zeros((g.data.number_samples, 2 + g.data.mocap_data.shape[1]))  # samples+class+ datacolumns
        for start, end, class_index, attributes in self.windows:
            for i in range(start, end):
                data[i, 1] = class_index
        data[:, 0] = range(g.data.number_samples)
        data[:, 2:] = g.data.mocap_data[:, :]

        file_name = self.file_name.split('.')[0] + annotator_suffix + "_norm_data.csv"
        np.savetxt(directory + os.sep + file_name, data, delimiter=',', header=header, comments='')

        # --------------------------------------------------------------------------------------
        # Third file: the backup with the windows.
        # --------------------------------------------------------------------------------------

        self.save_windows(directory, annotator_suffix[1:] + '_windows')


class WindowProcessorStates(WindowProcessor):
    def __init__(self, file_path, annotated, load_backup=False):
        super(WindowProcessorStates, self).__init__(file_path, annotated, load_backup)

        self.old_classes = []
        self.replace_classes_with_states()
        # del self.windows_1
        # del self.windows_2
        # del self.windows_3
        # del self.load_prediction
        # del self.load_predictions
        # del self.save_predictions
        # del self.save_window
        # del self.insert_window
        # del self.delete_window

    def replace_classes_with_states(self):
        if type(self.windows[0][3]) == type([]):  # If windows have 4 entries they are classes & attributes.

            for i, window in enumerate(self.windows):
                self.windows[i] = (window[0], window[1], 0, window[2])

    def load_backup(self, backup_path=None) -> list:
        """Loads the windows from an opened file

        """
        if backup_path is None:
            backup_path = f'{g.settings["backUpPath"]}{os.sep}{self.file_name.split(".")[0]}_backup.txt'

        if os.path.exists(backup_path):
            self.backup = open(backup_path, 'r+t')
        else:
            self.backup = open(backup_path, 'wt')

        lines = self.backup.readlines()
        windows = []
        for line in lines:
            # print (line[:-1])
            window = eval(line[:-1])
            windows.append(window)

        return windows

    def change_window(self, window_index: int, state_index: int = None, save: bool = True, **_):
        """Changes the labels of the window at window_index

        Arguments:
        ---------
        window_index : int
            Integer between 0 and the number of windows.
            Which window to modify
        state_index : int (optional)
            the new state of the modified window
        save : bool
            If true the method saves the changes directly to the file
            set to False if there is more than 1 window that needs to be modified
            Last modified window should have save = True or call make_backup after all modifications
        ---------
        """

        # Changes the value of the window at window_index

        new_window = list(self.windows[window_index])
        if state_index is not None:
            new_window[2] = state_index
        self.windows[window_index] = tuple(new_window)
        if save:
            self.make_backup()

    def save_results(self, directory: str, annotator_id: int, tries: int):
        """Saves the finished labels, normalized data, and windows in 3 files in the provided directory

        Arguments:
        ----------
        directory : str
            path to the directory where the results should be saved.
        annotator_id : int
            ID of the person annonating the dataset.
        tries : int
            number indicating how often the currently worked on file
            was beeing annotated by the same annotator.
        ----------

        """
        annotator_suffix = '_A' + '%0.2d' % (int(annotator_id)) + '_N' + '%0.2d' % (int(tries))

        # --------------------------------------------------------------------------------------
        # Second file normalized data: sample column, class label column, head rx,head ry, etc.
        # --------------------------------------------------------------------------------------
        header = "sample,statelabel"
        for i in range(22):
            bodysegment = g.data.body_segments[i]
            for coordinate in ['RX', 'RY', 'RZ', 'TX', 'TY', 'TZ']:
                header += "," + bodysegment + '_' + coordinate

        data = np.zeros((g.data.number_samples, 2 + g.data.mocap_data.shape[1]))  # samples+class+ datacolumns
        for start, end, state_index, _ in self.windows:
            for i in range(start, end):
                data[i, 1] = state_index
        data[:, 0] = range(g.data.number_samples)
        data[:, 2:] = g.data.mocap_data[:, :]

        file_name = self.file_name.split('.')[0] + annotator_suffix + "_states.csv"
        np.savetxt(directory + os.sep + file_name, data, delimiter=',', header=header, comments='')

        # --------------------------------------------------------------------------------------
        # Third file: the backup with the windows.
        # --------------------------------------------------------------------------------------

        self.save_windows(directory, annotator_suffix[1:] + '_windows')


class SlidingWindowDataset(Dataset):
    """Segments data using a sliding window approach and provides methods for iteration
    
    Stored segments have 4 Dimensions for quick use in torch models
    """

    def __init__(self, data: np.array, window_length: int, window_step: int):
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

        # self.data = torch.from_numpy(np.delete(data[np.newaxis,np.newaxis,:,:], range(66,72), 3)).float()
        self.data = np.delete(data, range(66, 72), 1)
        self.data = self.normalize(self.data)
        self.data = torch.from_numpy(self.data[np.newaxis, np.newaxis, :, :]).float()

        self.window_length = window_length
        self.window_step = window_step

    def __getitem__(self, segment_index: int):
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
        return self.data[:, :, lowerbound:upperbound, :]

    def __len__(self):
        """Returns the length of the dataset/number of segments. """

        return int((self.data.shape[2] - self.window_length) / self.window_step) + 1

    def __range__(self, segment_index):
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

        # if segment_index < self.__len__():
        lowerbound = self.window_step * segment_index
        upperbound = lowerbound + self.window_length
        # else:
        #    upperbound = self.data.shape[2]
        #    lowerbound = upperbound - self.window_length
        return lowerbound, upperbound

    def normalize(self, data):
        """Normalizes all sensor channels

        :param data: numpy integer matrix
            Sensor data
        :return:
            Normalized sensor data
        """
        try:
            max_list = np.array(NORM_MAX_THRESHOLDS)
            min_list = np.array(NORM_MIN_THRESHOLDS)
            diffs = max_list - min_list
            for i in np.arange(data.shape[1]):
                data[:, i] = (data[:, i] - min_list[i]) / diffs[i]
            #     Checking the boundaries
            data[data > 1] = 0.99
            data[data < 0] = 0.00
        except Exception as e:
            raise e

        return data


class LabeledSlidingWindowDataset(SlidingWindowDataset):
    """Expands the SlidingWindowDataset to keep track of label information"""

    def __init__(self, data, window_length, window_step):
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

        super(LabeledSlidingWindowDataset, self).__init__(data, window_length, window_step)

        self.classes = np.zeros((self.__len__(),
                                 g.data.number_samples,
                                 g.classes.__len__() - 1),  # Classes -1 since network cant output class None
                                dtype=int) - 1
        self.attributes = np.zeros((self.__len__(),
                                    g.data.number_samples,
                                    g.attributes.__len__()),
                                   dtype=float) - 1

    def save_labels(self, index, label, label_kind):
        lower, upper = self.__range__(index)
        if label_kind == 'class':
            self.classes[index, lower:upper, :] = label
        elif label_kind == 'attributes':

            self.attributes[index, lower:upper, :] = label

    def evaluate_labels(self, label_kind, average=None, metric=None):
        if label_kind == 'attributes':
            if average:
                self.average_attributes()
                # self.binarize_attributes() # moved to make_windows method
                # so that it happens after class prediction
            else:
                self.binarize_attributes()
                self.label_mode(label_kind)
            if metric is not None:
                self.predict_classes_from_attributes(metric)

        elif label_kind == 'class':

            self.label_mode(label_kind)
            self.generate_blank_attributes()

    def binarize_attributes(self):
        self.attributes = self.attributes.round().astype(np.int)

    def average_attributes(self):
        result = np.zeros((g.data.number_samples, 19))
        for i in range(g.data.number_samples):
            frame = self.attributes[:, i, :]
            for j in range(g.attributes.__len__()):
                attribute = frame[:, j][frame[:, j] > -1]

                if not list(attribute):
                    result[i] = np.zeros((g.attributes.__len__()))
                    result[i, -1] = 1
                else:
                    result[i, j] = np.mean(attribute, 0, dtype=np.float)
        self.attributes = result

    def label_mode(self, label_kind):  # Mode of labels
        if label_kind == 'class':
            labels = self.classes  # n * samples * c
            results = 3

            result = np.zeros((g.data.number_samples, results), dtype=int) + 7
            for i in range(g.data.number_samples):
                segments = labels[:, i]  # n * c
                segments = segments[segments > -1].reshape((-1, segments.shape[1]))  # remove all n without labels
                if segments.size == 0:
                    continue
                # take top1 3 times each time removing the top1 from the possibilities
                for j in range(3):
                    vector = segments[:, 0]  # all segments first class
                    top_j = self.sorted_classes_vector(vector)[-1]  # most common class in vector without -1
                    result[i, j] = top_j
                    # remove top_j
                    segments_shape = segments.shape
                    segments_shape = (segments_shape[0], segments_shape[1] - 1)  # one class is removed
                    segments = segments[segments != top_j].reshape(segments_shape)

            self.classes = result

        elif label_kind == 'attributes':

            labels = self.attributes  # n * samples * attributes

            result = np.zeros((g.data.number_samples, g.attributes.__len__()), dtype=int) - 1
            for i in range(g.data.number_samples):
                frame = labels[:, i, :]  # n * attributes
                for j in range(g.attributes.__len__()):
                    attribute = frame[:, j]
                    attribute = attribute[attribute > -1]
                    if not list(attribute):
                        result[i] = np.zeros((g.attributes.__len__()))
                        result[i, -1] = 1
                    else:
                        result[i, j] = self.sorted_classes_vector(attribute)[-1]  # n

            self.attributes = result

    def sorted_classes_vector(self, arr):
        frame_values, frame_counts = np.unique(arr[arr > -1], return_counts=True)
        sorted_vector = frame_values[frame_counts.argsort()]  # Sort the classes by count
        return sorted_vector

    def predict_classes_from_attributes(self, att_rep):
        attributes = np.array(att_rep[:, 1:])
        # attributes = attributes/np.linalg.norm(attributes,axis=1,keepdims=True)
        # distances = distance_matrix(self.attributes,attributes)
        distances = 1 - cosine_similarity(self.attributes, attributes)

        sorted_distances = np.argsort(distances, 1)
        # print("distances shape ", distances.shape)
        # print("sorted distances shape", sorted_distances.shape)
        # self.top3_distances = distances[sorted_distances[:,:3]]
        self.top3_distances = np.zeros((g.data.number_samples, 3))
        self.classes = np.zeros((g.data.number_samples, 3), dtype=np.int)

        for i in range(g.data.number_samples):
            self.top3_distances[i] = distances[i, sorted_distances[i, :3]]
            sorted_classes = att_rep[sorted_distances[i], 0]
            # First occurance (index) of each class. np.unique is sorted by class not by index.
            indexes = np.unique(sorted_classes, return_index=True)[1]
            # The classes were already sorted by distance.
            # Sorting the indexes again will restore that order
            sorted_classes = [sorted_classes[index] for index in sorted(indexes)]
            self.classes[i] = sorted_classes[:3]

    def generate_blank_attributes(self):
        self.attributes = np.zeros((g.data.number_samples, g.attributes.__len__()), dtype=int)
        # self.attributes[:,-1] = np.ones((g.data.number_samples), dtype=int)
        self.attributes[:, -1] = 1

    def make_windows(self, label_kind, average, metric, deep_rep=None):
        self.evaluate_labels(label_kind, average, metric)
        if average:
            self.binarize_attributes()

        if deep_rep is not None:
            self.choose_3_from_6(deep_rep, False)  # TODO: make this toggleable

        windows_top3 = [[], [], []]

        window_start = 0

        for i in range(1, g.data.number_samples - 1):
            if not self.check_same_labels(i, i + 1):

                window_end = i + 1
                # print("self.classes.shape ",self.classes.shape)
                class_label = self.classes[i, :]
                attributes = list(self.attributes[i])

                for j in range(3):
                    window = (window_start, window_end, class_label[j], attributes)
                    windows_top3[j].append(window)

                window_start = i + 1

        i = g.data.number_samples - 1
        window_end = i + 1
        class_label = self.classes[i, :]
        attributes = list(self.attributes[i])

        for j in range(3):
            window = (window_start, window_end, class_label[j], attributes)
            windows_top3[j].append(window)

        if metric is not None:
            self.check_existing_attrib_vector(windows_top3, metric)

        return windows_top3

    def check_same_labels(self, frame_a, frame_b):
        # The labels are the same, if neither class prediction
        # nor attribute changes between frame_a and frame_b
        # return False

        xnor_classes = [not a ^ b for a, b in zip(self.classes[frame_a, :], self.classes[frame_b, :])]
        xnor_attributes = [not a ^ b for a, b in zip(self.attributes[frame_a], self.attributes[frame_b])]
        return sum(xnor_classes) == 3 and sum(xnor_attributes) == g.attributes.__len__()

    def check_existing_attrib_vector(self, windows_top3, metric):
        """Checks whether the Attribute vector is valid by looking if its in the metrics array"""

        metric_attributes = metric[:, 1:]
        # np.savetxt("metric", metric_attributes)
        for i in range(windows_top3[0].__len__()):
            attributes = np.array(windows_top3[0][i][3])
            # print(attributes)
            # print("att shape", attributes.shape)
            # print("metric shape", metric_attributes.shape)
            if not (attributes == metric_attributes).all(1).any():
                attributes[-1] = 1
                for j in range(3):
                    window = list(windows_top3[j][i])
                    window[3] = list(attributes)
                    windows_top3[j][i] = tuple(window)

    def choose_3_from_6(self, deep_rep, use_threshold=False):
        for i in range(g.data.number_samples):
            # distances index 0-2 are self.top3distances, 3-5 are deep_rep.top3distances
            distances = np.array([self.top3_distances[i], deep_rep.top3_distances[i]]).reshape(-1)
            # print("distances ", distances)

            sorted_distances = np.argsort(distances)
            if sorted_distances[0] >= 3 \
                    or (use_threshold and (distances[3] <= g.settings['deep_threshold'])):
                self.attributes[i] = deep_rep.top1_attributes[i]
            for j in range(3):
                if use_threshold and (distances[3 + j] <= g.settings['deep_threshold']):
                    self.classes[i, j] = deep_rep.classes[i, j]

                elif sorted_distances[j] < 3:
                    continue
                else:
                    self.classes[i, j - 3] = deep_rep.classes[i, j - 3]
                    # print(f"took top{j+1} from deep_rep in frame {i}")


class DeepRepresentationDataset(SlidingWindowDataset):
    def __init__(self, data, window_length, window_step, file_name, windows, network):
        """Initializes the dataset
        
        Arguments:
        ----------
        data : numpy array
            The data that needs to be segmented
        window_length : int
            The length of each segment
        window_step : int
            The stride/step between segments
        windows: list
            The groundtrouth for the data
        network: torch.nn.model
            Neural network that gives the deep representation for the data.
        """
        super(DeepRepresentationDataset, self).__init__(data, window_length, window_step)

        self.file_names = [file_name]
        self.deep_labels = self.windows_to_segment_labels(windows)
        self.deep_fc2 = self.data_to_deep_rep(network)

        size = network.fc4.out_features  # TODO: think about fully convolutional fc4
        self.fc2 = np.zeros((g.data.number_samples, size))

        # self.__getitem__= lambda i: self.data[i]
        # self.__len__ = lambda : self.data.shape[0]
        # self.__range__ = lambda i: (i,i+1)
        del self.data
        self.__len__ = lambda: int((g.data.number_samples - self.window_length) / self.window_step) + 1
        self.__getitem__ = lambda _: None

    def data_to_deep_rep(self, network):
        size = network.fc4.out_features  # TODO: think about fully convolutional fc4
        deep_rep = np.zeros((self.__len__(), size))

        # network.online = True
        for i in range(self.__len__()):
            _, deep_rep[i] = network(self.__getitem__(i))

        # network.online = False
        return deep_rep

    def windows_to_segment_labels(self, windows):
        last_window = 0
        labels = np.zeros((self.__len__(), 1 + g.attributes.__len__()))
        for i in range(self.__len__()):
            lower, upper = self.__range__(i)
            best_window_index = last_window
            best_iou = self.intersection_over_union(lower, upper,
                                                    windows[last_window][0],
                                                    windows[last_window][1])

            for j in range(last_window + 1, windows.__len__()):
                start, end, _, _ = windows[j]
                iou = self.intersection_over_union(lower, upper, start, end)

                if iou == 0 and start < lower:
                    # if the intervals don't intersect and
                    # window interval is earlier than range interval
                    # then check next window
                    continue
                elif iou == 0:
                    last_window = j - 1  # even if current window isn't in range anymore,
                    # previous window could be in next range
                    break
                if iou > best_iou:
                    best_window_index = j
                    best_iou = iou

            _, _, class_, attributes = windows[best_window_index]
            label = np.array([class_] + attributes)
            # print(label)
            labels[i] = label

        # print(labels)
        return labels

    def intersection_over_union(self, a1, b1, a2, b2):
        """Calculates the IoU between 2 intervals [a1,b1] and [a2,b2]"""
        intersection = min(b1, b2) - max(a1, a2)
        if intersection < 0:
            return 0  # if less then zero the 2 intervalls don't intersect. intersection is empty
        union = max(b1, b2) - min(a1, a2)

        return intersection / union

    def add_deep_rep_data(self, file_name, data, windows, network):
        if file_name not in self.file_names:
            deep_rep = DeepRepresentationDataset(data, self.window_length,
                                                 self.window_step, file_name, windows, network)
            # add to files list
            self.file_names.append(file_name)

            # stack data onto self.data
            self.data = np.vstack((self.deep_fc2, deep_rep.deep_fc2))

            # stack labels onto self.deep_labels
            self.deep_labels = np.vstack((self.deep_labels, deep_rep.deep_labels))

    def save_fc2(self, index, fc2):
        lower, upper = self.__range__(index)
        self.fc2[lower:upper, :] += fc2.detach().numpy()

    def predict_labels_from_fc2(self):
        distances = 1 - cosine_similarity(self.fc2, self.deep_fc2)

        sorted_distances = np.argsort(distances, 1)
        # print("distances shape ", distances.shape)
        # print("sorted distances shape", sorted_distances.shape)
        # self.top3_distances = distances[sorted_distances[:, :3]]
        self.top3_distances = np.zeros((g.data.number_samples, 3))

        self.classes = np.zeros((g.data.number_samples, 3), dtype=np.int)
        self.top1_attributes = np.zeros((g.data.number_samples,
                                         g.attributes.__len__()),
                                        dtype=np.int) - 1
        for i in range(g.data.number_samples):
            self.top3_distances[i] = distances[i, sorted_distances[i, :3]]

            self.top1_attributes[i] = self.deep_labels[sorted_distances[i, 0], 1:]
            sorted_classes = self.deep_labels[sorted_distances[i], 0]
            # First occurance (index) of each class. np.unique is sorted by class not by index.
            indexes = np.unique(sorted_classes, return_index=True)[1]
            # The classes were already sorted by distance.
            # Sorting the indexes again will restore that order
            sorted_classes = [sorted_classes[index] for index in sorted(indexes)]
            self.classes[i] = sorted_classes[:3]
