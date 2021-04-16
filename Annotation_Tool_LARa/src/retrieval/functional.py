import os

from PyQt5 import QtWidgets
import dill
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

from network import Network
from dialogs import PlotDialog
import global_variables as g
from data_management import SlidingWindowDataset, DeepRepresentationDataset


class DenseSlidingWindowDataset(SlidingWindowDataset):
    def __init__(self, data: np.array, window_length: int):
        super(DenseSlidingWindowDataset, self).__init__(data, window_length, window_step=100)

        self.classes = np.zeros((len(self), len(g.classes)), dtype=float) - 1
        self.attributes = np.zeros((len(self), len(g.attributes)), dtype=float) - 1

        self.ground_truth = self.make_ground_truth()

    def save_labels(self, index, label, label_kind):
        if label_kind == 'attributes':
            self.attributes[index, :] = label
        else:
            raise ValueError

    def make_ground_truth(self):
        ground_truth = np.zeros((self.__len__(),), dtype=int) - 1

        # labels = expand window classes to array
        labels = np.array([])
        for start, end, class_, _ in g.windows.windows:
            labels = np.hstack((labels, np.repeat(class_, end - start)))

        print(labels.shape)
        # use mode to assign ground truth for each segment
        for i in range(len(self)):
            lower, upper = self.__range__(i)
            values, counts = np.unique(labels[lower:upper], return_counts=True)
            order = np.argsort(counts)
            ground_truth[i] = values[order[-1]]

        return ground_truth

    def make_heatmap(self, class_index):
        return self.classes[:, class_index]

    def predict_classes_from_attributes(self, att_rep):
        attributes = np.array(att_rep[:, 1:])
        # attributes = attributes/np.linalg.norm(attributes,axis=1,keepdims=True)
        # distances = distance_matrix(self.attributes,attributes)
        distances = 1 - cosine_similarity(self.attributes, attributes)

        sorted_distances_indexes = np.argsort(distances, 1)
        # print("distances shape ", distances.shape)
        # print("sorted distances shape", sorted_distances.shape)
        # self.top3_distances = np.zeros((g.data.number_samples, 3))

        for i in range(len(self)):
            print(f"Evaluation: {i+1}/{len(self)}")
            # self.top3_distances[i] = distances[i, sorted_distances[i, :3]]

            sorted_classes = att_rep[sorted_distances_indexes[i], 0]
            sorted_distances = distances[i, sorted_distances_indexes[i, :]]
            #print("sorted_classes.len", len(sorted_classes))
            #print("sorted_distances.len", len(sorted_distances))
            #print(sorted_classes[:90])
            #print(sorted_distances[:90])

            # First occurrence (index) of each class. np.unique is sorted by class not by index.
            indexes = np.unique(sorted_classes, return_index=True)[1]
            #print(indexes)

            # The classes were already sorted by distance.
            # Sorting the indexes again will restore that order
            sorted_classes = [sorted_classes[index] for index in sorted(indexes)]
            sorted_distances = [sorted_distances[index] for index in sorted(indexes)]

            #print(sorted_classes)
            #print(sorted_distances)

            #print("indexes.shape", indexes.shape)
            #print("sorted_classes.len", len(sorted_classes))
            #print("sorted_distances.len", len(sorted_distances))
            #print("g.classes.len", len(g.classes))
            #print("self.classes.shape", self.classes.shape)
            self.classes[i] = np.array(sorted_distances)
            #print(self.classes[i])
            #break
        self.classes = 1 - self.classes

class Annotator:
    def __init__(self, selected_network, deep_rep=False, paths=None):
        super(Annotator, self).__init__()
        if paths is None:
            paths = []

        self.deep_rep = deep_rep
        self.paths = paths

        self.selected_network = selected_network
        self.window_step = g.settings['segmentationWindowStride']
        self.device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")
        print("Device: ", self.device)
        # Load network
        print("Loading Network:", self.selected_network)
        self.network, self.config, self.att_rep = self.load_network(self.selected_network)
        if self.network is None:
            return
        self.network.deep_rep = self.deep_rep
        print("Loaded Network:", self.selected_network, "\n")

    def load_network(self, index):
        """Loads the selected network"""
        try:
            checkpoint = torch.load(g.networks_path + g.networks[index]['file_name'],
                                    map_location=self.device)
            self.network = None

            state_dict = checkpoint['state_dict']
            config = checkpoint['network_config']
            if 'att_rep' in checkpoint.keys():
                att_rep = checkpoint['att_rep']
            else:
                att_rep = None
            network = Network(config)
            network.load_state_dict(state_dict)
            network.eval()
            return network, config, att_rep

        except KeyError as e:
            self.network = None
            print("Something went wrong sorry.")
            raise e
        except FileNotFoundError:
            self.network = None
            print("Could not find the " + g.networks[index]["name"]
                  + " at " + g.networks_path + g.networks[index]['file_name'])
            return None, None, None
            # raise e

    def run(self):
        # Segment Data
        print(f"Segmenting data: 0/{1 + len(self.paths)}")

        window_length = self.config['sliding_window_length']
        dataset = DenseSlidingWindowDataset(g.data.mocap_data, window_length)
        dataset_len = len(dataset)
        print(f"Segmenting data: 1/{1 + len(self.paths)}")

        # Making deep representation
        if self.deep_rep:
            self.network.deep_rep = True
            deep_rep = self.get_deep_representations(self.paths, self.config, self.network)
        else:
            deep_rep = None
        print(f"Segmented data\n")

        # Forward through network
        print(f"Annotating. Total samples: {dataset_len}")
        label_kind = self.config['labeltype']
        for i in range(dataset_len):
            if self.deep_rep:
                label, fc2 = self.network(dataset.__getitem__(i))
                deep_rep.save_fc2(i, fc2)
            else:
                label = self.network(dataset.__getitem__(i))

            if label_kind == 'class':
                label = torch.argsort(label, descending=True)[0]
                dataset.save_labels(i, label, label_kind)
            elif label_kind == 'attributes':
                label = label.detach()
                dataset.save_labels(i, label[0], label_kind)
            else:
                raise Exception

            print(f"Annotating {i + 1}/{dataset_len}")
        print("Annotated\n")

        # Evaluate results
        print("Evaluating")
        if self.deep_rep:
            deep_rep.predict_labels_from_fc2()
        if self.att_rep is not None:
            # metrics = Metrics(config, self.device, att_rep)
            metrics = self.att_rep
        else:
            metrics = None

        graphs = []

        dataset.predict_classes_from_attributes(metrics)
        for i in range(len(g.classes)):
            heatmap_data = dataset.make_heatmap(i)
            dlg = PlotDialog(None)
            dlg.setWindowTitle("Graph")
            plot = dlg.graph_widget()
            plot.setTitle(f'<font size="6"><b>{g.classes[i]}</b></font>')
            #legend = plot.addLegend(offset=(-10, 15), labelTextSize='20pt')
            plot.getAxis('left').setLabel("Cosine Similarity", **{'font-size': '14pt'})
            plot.plot(heatmap_data)
            _ = dlg.show()
            graphs.append(dlg)
        return graphs

    def get_deep_representations(self, paths, config, network):
        current_file_name = g.windows.file_name
        name_parts = current_file_name.split('_')
        subject_id = [s for s in name_parts if 'S' in s][0]
        pickled_deep_rep_path = f"{g.settings['saveFinishedPath']}{os.sep}{subject_id}.p"
        if os.path.exists(pickled_deep_rep_path):
            deep_rep = dill.load(open(pickled_deep_rep_path, "rb"))
            existing_files = deep_rep.file_names
            new_files = [os.path.split(path)[1] for path in paths]

            if [file for file in existing_files if file not in new_files] != []:
                # The deep_rep has files that are not needed. Better make new deep_rep
                # print("making new deep_rep. unneeded files")
                deep_rep = None
            elif [file for file in new_files if file not in existing_files] != []:
                # There are new files that need to be added to deep_rep.
                # It will be updated in the following for-loop
                # print("updating deep_rep. too few files")
                pass
            else:
                # existing and new files are identical.
                # print("returning old deep_rep. identical file list")
                return deep_rep
        else:
            deep_rep = None

        for i, path in enumerate(paths):

            # getting the data
            data = np.loadtxt(path, delimiter=',', skiprows=1)
            data = data[:, 2:]

            # Getting windows file path
            directory, data_name = os.path.split(path)
            window_name_parts = data_name.split('_')[:5]
            window_name_parts.append("windows.txt")
            window_name = window_name_parts[0]
            for part in window_name_parts[1:]:
                window_name += "_" + part
            window_path = directory + os.sep + window_name

            # reading the windows_file
            windows = []
            with open(window_path, 'r+t') as windows_file:
                lines = windows_file.readlines()
                for line in lines:
                    window = eval(line[:-1])
                    windows.append(window)

            if deep_rep is None:
                window_length = config['sliding_window_length']
                deep_rep = DeepRepresentationDataset(data, window_length,
                                                     self.window_step, data_name,
                                                     windows, network)
            else:
                deep_rep.add_deep_rep_data(data_name, data, windows, network)

            print(f"Segmenting data: {1 + i}/{1 + len(self.paths)}")

        dill.dump(deep_rep, open(pickled_deep_rep_path, "wb"))
        return deep_rep


def browse_files(caption="caption"):
    paths = QtWidgets.QFileDialog.getOpenFileNames(
        parent=None,
        caption=caption,
        directory=g.settings['saveFinishedPath'],
        filter=f'CSV Files (*norm_data.csv)',
        initialFilter='')[0]
    return paths
