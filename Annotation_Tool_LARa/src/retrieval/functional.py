import math
import os

import dill
import numpy as np
import torch
from PyQt5 import QtWidgets
from sklearn.metrics.pairwise import cosine_similarity

import global_variables as g
from data_management import SlidingWindowDataset, DeepRepresentationDataset
from dialogs import PlotDialog
from network import Network
from controllers.controller import Graph


class DenseSlidingWindowDataset(SlidingWindowDataset):
    def __init__(self, data: np.array, window_length: int, window_step: int):
        super(DenseSlidingWindowDataset, self).__init__(data, window_length, window_step)
        length = super(DenseSlidingWindowDataset, self).__len__()

        self.classes = np.zeros((length, len(g.classes)), dtype=float) - 1
        self.attributes = np.zeros((length, len(g.attributes)), dtype=float) - 1

        self.attribute_rep_queries = None

        self.ground_truth = self.make_ground_truth()
        self.attr_ground_truth = self.make_attr_ground_truth()

    def __len__(self):
        return self.classes.shape[0]

    def save_labels(self, index, label, label_kind):
        if label_kind == 'attributes':
            self.attributes[index, :] = label
        else:
            raise ValueError

    def make_ground_truth(self) -> np.array:
        ground_truth = np.zeros((len(self),), dtype=int) - 1

        # labels = expand window classes to array
        labels = np.array([])
        for start, end, class_, _ in g.windows.windows:
            labels = np.hstack((labels, np.repeat(class_, end - start)))

        # print("ground truth labels shape", labels.shape)
        # use mode to assign ground truth for each segment
        for i in range(len(self)):
            lower, upper = self.__range__(i)
            values, counts = np.unique(labels[lower:upper], return_counts=True)
            order = np.argsort(counts)
            ground_truth[i] = values[order[-1]]

        return ground_truth

    def make_attr_ground_truth(self) -> np.array:
        ground_truth = np.zeros((len(self), len(g.attributes)), dtype=int) - 1

        # labels = expand window classes to array
        labels = None
        for start, end, _, attr in g.windows.windows:
            if labels is None:
                labels = np.tile(attr, (end - start, 1))
                # print(labels.shape)
            else:
                labels = np.vstack((labels, np.tile(attr, (end - start, 1))))

        # print("attr ground thruth labels shape", labels.shape)
        # use mode to assign ground truth for each segment
        for i in range(len(self)):
            lower, upper = self.__range__(i)
            values, counts = np.unique(labels[lower:upper], return_counts=True, axis=0)
            order = np.argsort(counts)
            ground_truth[i] = values[order[-1]]

            # print("values", values)
            # print("counts", counts)
            # print("order", order)
            # print("ground_truth[i]", ground_truth[i], "\n")

        return ground_truth

    def make_heatmap(self, class_index, normalize=False):
        if not normalize:
            return self.classes[:, class_index]
        heatmap = self.classes[:, class_index]
        min_ = min(heatmap)
        max_ = max(heatmap)
        heatmap = (heatmap - min_) / (max_ - min_)
        return heatmap

    def make_attr_rep_heatmap(self, attr_rep_index, normalize=False):
        if self.attribute_rep_queries is None:
            raise RuntimeError("self.attribute_rep_queries is None. Please run predict_attribute_reps first.")
        if not normalize:
            return self.attribute_rep_queries[:, attr_rep_index]
        heatmap = self.attribute_rep_queries[:, attr_rep_index]
        min_ = min(heatmap)
        max_ = max(heatmap)
        heatmap = (heatmap - min_) / (max_ - min_)
        return heatmap

    def make_attr_heatmap(self, attr_index, normalize=False):
        if not normalize:
            return self.attributes[:, attr_index]
        heatmap = self.attributes[:, attr_index]
        min_ = min(heatmap)
        max_ = max(heatmap)
        heatmap = (heatmap - min_) / (max_ - min_)
        return heatmap

    def predict_classes_from_attributes(self, distance="cosine"):
        attribute_rep = g.attribute_rep[:, 1:]

        distances = None

        if distance == "cosine":
            distances = cosine_similarity(self.attributes, attribute_rep)
        elif distance == "bce":
            attributes = torch.from_numpy(self.attributes)
            attribute_rep = torch.from_numpy(attribute_rep)
            bceloss = torch.nn.BCELoss()
            distances = np.ones((attributes.shape[0], attribute_rep.shape[0]))
            for i in range(attributes.shape[0]):
                for j in range(attribute_rep.shape[0]):
                    distances[i, j] -= bceloss(attributes[i], attribute_rep[j]).item()/100
        else:
            ValueError(f"Supported distances are 'cosine' and 'bce'. But '{distance}' was given.")

        sorted_distances_indexes = np.argsort(1 - distances, 1)

        for i in range(len(self)):
            # print(f"Evaluation: {i + 1}/{len(self)}")

            sorted_classes = g.attribute_rep[sorted_distances_indexes[i], 0]
            sorted_distances = distances[i, sorted_distances_indexes[i, :]]

            # First occurrence (index) of each class. np.unique is sorted by class not by index.
            indexes = np.unique(sorted_classes, return_index=True)[1]
            # Get the distance
            self.classes[i] = np.array(sorted_distances[indexes])

    def predict_attribute_reps(self, distance="cosine"):
        attribute_rep = g.attribute_rep[:, 1:]

        if distance == "cosine":
            self.attribute_rep_queries = cosine_similarity(self.attributes, attribute_rep)

        elif distance == "bce":
            attributes = torch.from_numpy(self.attributes)
            attribute_rep = torch.from_numpy(attribute_rep)
            bceloss = torch.nn.BCELoss()
            self.attribute_rep_queries = np.zeros((attributes.shape[0], attribute_rep.shape[0]))
            for i in range(attributes.shape[0]):
                for j in range(attribute_rep.shape[0]):
                    self.attribute_rep_queries[i, j] = bceloss(attributes[i], attribute_rep[j]).item() / 100
            self.attribute_rep_queries = 1 - self.attribute_rep_queries
        else:
            ValueError(f"Supported distances are 'cosine' and 'bce'. But '{distance}' was given.")

    def retrieve_list(self, class_index, length=None):
        retrieval_list = []
        heatmap = self.make_heatmap(class_index, False)

        indexes = np.argsort(1 - heatmap)  # 1- Because np.argsort sorts in ascending order
        if length is not None and length < len(indexes):
            indexes = indexes[:length]
        for i in indexes:
            retrieval_list.append({"range": self.__range__(i), "index": i, "value": heatmap[i]})
        return retrieval_list

    def retrieve_attr_rep_list(self, attr_index, length=None):
        retrieval_list = []
        heatmap = self.make_attr_rep_heatmap(attr_index, False)

        indexes = np.argsort(1 - heatmap)  # 1- Because np.argsort sorts in ascending order
        if length is not None and length < len(indexes):
            indexes = indexes[:length]
        for i in indexes:
            retrieval_list.append({"range": self.__range__(i), "index": i, "value": heatmap[i]})
        return retrieval_list

    def retrieve_attr_list(self, attr_index, length=None):
        retrieval_list = []
        heatmap = self.make_attr_heatmap(attr_index, False)

        indexes = np.argsort(1 - heatmap)  # 1- Because np.argsort sorts in ascending order
        if length is not None and length < len(indexes):
            indexes = indexes[:length]
        for i in indexes:
            retrieval_list.append({"range": self.__range__(i), "index": i, "value": heatmap[i]})
        return retrieval_list

    def mean_average_precision(self, type="classes"):
        avep_sum = 0
        nan_results = 0
        if type == "classes":
            for i in range(len(g.classes)):
                avep = self.average_precision(class_index=i)
                if not math.isnan(avep):
                    avep_sum += avep
                else:
                    nan_results += 1
                    #print(f"Warning! Class {g.classes[i]} was not included in Mean Average Precision")
            return avep_sum / (len(g.classes) - nan_results)
        elif type == "attr_rep":
            for i in range(g.attribute_rep.shape[0]):
                avep = self.average_precision(attr_rep_index=i)
                if not math.isnan(avep):
                    avep_sum += avep
                else:
                    nan_results += 1
                    #print(f"Warning! Attribute Vector {i} was not included in Mean Average Precision")
            if nan_results!= g.attribute_rep.shape[0]:
                return avep_sum / (g.attribute_rep.shape[0] - nan_results)
            else:
                return 0
        elif type == "attr":
            for i in range(len(g.attributes)):
                avep = self.average_precision(attr_index=i)
                if not math.isnan(avep):
                    avep_sum += avep
                else:
                    nan_results += 1
                    #print(f"Warning! Attribute Vector {i} was not included in Mean Average Precision")
            return avep_sum / (len(g.attributes) - nan_results)
        else:
            raise ValueError(f'Permitted types are "classes", "attr_rep" and "attr". "{type}" was given.' )

    def average_precision(self, class_index=None, attr_rep_index=None, attr_index=None):
        if sum([0 if i is None else 1 for i in [class_index,attr_index,attr_rep_index]]) != 1:
            given_parameters = sum([0 if i is None else 1 for i in [class_index, attr_index, attr_rep_index]])
            raise ValueError(f"Exactly 1 of the parameters must be not None. {given_parameters} parameters were not None.")

        if class_index is not None:
            retrieval_list = self.retrieve_list(class_index)
            relevant_windows: np.array = self.ground_truth == class_index
        elif attr_rep_index is not None:
            retrieval_list = self.retrieve_attr_rep_list(attr_rep_index)
            relevant_windows: np.array = np.zeros((len(self),))
            for i in range(len(self)):
                query = g.attribute_rep[attr_rep_index, 1:]
                # if abs(sum(query == self.attr_ground_truth[i]) - len(g.attributes)) <= 0:
                if all(query == self.attr_ground_truth[i]):
                    relevant_windows[i] = 1
        else:
            retrieval_list = self.retrieve_attr_list(attr_index)
            relevant_windows: np.array = self.attr_ground_truth[:, attr_index]

        total_relevant_windows = sum(relevant_windows)  # False negatives + True Positives
        # print("relevant windows", total_relevant_windows)
        if total_relevant_windows == 0:
            return math.nan

        retrieved_windows = 0  # True Positives + False Positives
        true_positives = 0

        precision_list = []
        for i in range(len(self)):
            retrieved_windows += 1

            # print("i", i, "/", len(self),"len(retrieval_list)", len(retrieval_list))
            retrieved_window_index = retrieval_list[i]['index']

            if relevant_windows[retrieved_window_index] == 1:
                true_positives += 1
            precision = true_positives / retrieved_windows
            precision_list.append(precision)
        precision_list = np.array(precision_list)

        # print("precision_list", precision_list)
        # print("relevant window", relevant_windows)
        # print([(d['index'], relevant_windows[d['index']]) for d in retrieval_list])

        sorted_relevant_windows = relevant_windows[[d['index'] for d in retrieval_list]]

        average_precision = sum(precision_list * sorted_relevant_windows) / total_relevant_windows
        return average_precision

    def add_dataset(self, dataset):
        if type(self) != type(dataset):
            raise TypeError(f"Datasets have to be the same type. {type(self)} and {type(dataset)}")
        if self.window_length != dataset.window_length:
            raise ValueError(f"window_lengths don't match. {self.window_length} and {dataset.window_length}")
        if self.window_step != dataset.window_step:
            raise ValueError(f"window_steps don't match. {self.window_step} and {dataset.window_step}")

        self.data = torch.cat((self.data, dataset.data), dim=2)
        self.classes = np.vstack((self.classes, dataset.classes))
        self.ground_truth = np.hstack((self.ground_truth, dataset.ground_truth))
        self.attributes = np.vstack((self.attributes, dataset.attributes))
        self.attr_ground_truth = np.vstack((self.attr_ground_truth, dataset.attr_ground_truth))
        self.attribute_rep_queries = np.vstack((self.attribute_rep_queries, dataset.attribute_rep_queries))

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
        self.network, self.config = self.load_network(self.selected_network)
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
            network = Network(config)
            network.load_state_dict(state_dict)
            network.eval()
            return network, config

        except KeyError as e:
            self.network = None
            print("Something went wrong sorry.")
            raise e
        except FileNotFoundError:
            self.network = None
            print("Could not find the " + g.networks[index]["name"]
                  + " at " + g.networks_path + g.networks[index]['file_name'])
            return None, None
            # raise e

    def run(self):
        # Segment Data
        #print(f"Segmenting data: 0/{1 + len(self.paths)}")

        window_length = self.config['sliding_window_length']
        dataset = DenseSlidingWindowDataset(g.data.mocap_data, window_length, window_step=window_length)
        #print(f"Segmenting data: 1/{1 + len(self.paths)}")

        # Making deep representation
        if self.deep_rep:
            self.network.deep_rep = True
            deep_rep = self.get_deep_representations(self.paths, self.config, self.network)
        else:
            deep_rep = None
        #print(f"Segmented data\n")

        # Forward through network

        #print(f"Annotating. Total samples: {len(dataset)}")
        label_kind = self.config['labeltype']
        percentiles = range(0, len(dataset), len(dataset)//10)
        for i in range(len(dataset)):
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
                raise ValueError(f"labeltype of the network has to be either 'class' or 'attributes'. "
                                 f"It was '{label_kind}'")
            if i in percentiles:
                # print(f"Annotating {i + 1}/{len(dataset)}")
                pass

        # dataset.attributes = dataset.ground_truth
        # print("Skipped Annotation using ground truth")
        #print("Annotated\n")

        # Evaluate results
        #print("Evaluating")
        if self.deep_rep:
            deep_rep.predict_labels_from_fc2()

        graphs = None

        #metric = "bce"
        metric = "cosine"
        dataset.predict_classes_from_attributes(metric)
        dataset.predict_attribute_reps(metric)
        #print("Evaluated\n")


        """
        graphs = self.show_graphs(dataset, "Cosine Similarity")

        metric = "bce"
        dataset.predict_classes_from_attributes(metric)
        dataset.predict_attribute_reps(metric)

        graphs.extend(self.show_graphs(dataset, "BCE"))
        """
        return dataset, graphs

    def show_graphs(self, dataset, y_axis):
        graphs = []
        dlg = PlotDialog(None, 9)
        dlg.setWindowTitle("Graph")
        plots = dlg.graph_widgets()

        class_graph = Graph(plots[0], "class", interval_lines=False)
        class_graph.setup()
        class_graph.reload_classes(g.windows.windows)
        graphs.append(class_graph)

        for i in range(0, len(g.classes)):
            heatmap_data = dataset.make_heatmap(i, True)
            plots[i + 1].setTitle(f'<font size="6"><b>{g.classes[i]}</b></font>')
            # plots[i + 1].setYRange(0, 1)
            # legend = plot.addLegend(offset=(-10, 15), labelTextSize='20pt')
            plots[i + 1].getAxis('left').setLabel(y_axis)
            plots[i + 1].plot(heatmap_data)
            dlg.showMaximized()
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

            if [file for file in existing_files if file not in new_files]:
                # The deep_rep has files that are not needed. Better make new deep_rep
                # print("making new deep_rep. unneeded files")
                deep_rep = None
            elif [file for file in new_files if file not in existing_files]:
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

def average_att_rep():
    attr_rep = np.zeros((len(g.classes),len(g.attributes)+1))
    for i in range(len(g.classes)):
        class_i_attributes = g.attribute_rep[g.attribute_rep[:, 0] == i]
        #print(class_i_attributes)
        averaged_attr = np.mean(class_i_attributes,axis=0)
        #print(averaged_attr,"\n")
        attr_rep[i] = averaged_attr
    g.attribute_rep = attr_rep
    #print(g.attribute_rep)