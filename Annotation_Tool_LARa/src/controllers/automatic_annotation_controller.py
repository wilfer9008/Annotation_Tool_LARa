"""
Created on 09.07.2020

@author: Erik Altermann
@email: Erik.Altermann@tu-dortmund.de
"""
from PyQt5 import QtWidgets
from os import sep
from .controller import Controller
from dialogs import ProgressDialog, PlotDialog
from PyQt5.QtCore import QThread, pyqtSignal
import torch
import os
import numpy as np
from network import Network
from data_management import LabeledSlidingWindowDataset, \
    DeepRepresentationDataset, RetrievalData
import time

import global_variables as g
import pyqtgraph as pg
from controllers.controller import Graph
import dill


class AutomaticAnnotationController(Controller):
    def __init__(self, gui):
        super(AutomaticAnnotationController, self).__init__(gui)

        self.window_step = g.settings['segmentationWindowStride']

        self.selected_network = 0  # TODO: Save last selected in settings
        self.current_window = -1
        self.deep_rep_files = []
        self.setup_widgets()

    def setup_widgets(self):
        self.load_tab(f'..{sep}ui{sep}automatic_annotation_mode.ui', "Automatic Annotation")
        # ComboBoxes
        self.network_comboBox = self.widget.findChild(QtWidgets.QComboBox, "aa_network_comboBox")
        self.network_comboBox.currentIndexChanged.connect(self.select_network)
        for k in sorted(g.networks.keys()):
            self.network_comboBox.addItem(g.networks[k]['name'])

        self.post_processing_comboBox = self.widget.findChild(QtWidgets.QComboBox, "aa_post_processing_comboBox")

        # Buttons
        self.annotate_button = self.widget.findChild(QtWidgets.QPushButton, "aa_annotate_button")
        self.annotate_button.clicked.connect(lambda _: self.gui.pause())
        self.annotate_button.clicked.connect(lambda _: self.annotate())

        self.load_predictions_button = self.widget.findChild(QtWidgets.QPushButton, "aa_load_prediction_button")
        self.load_predictions_button.clicked.connect(lambda _: self.load_predictions())

        # Graphs
        self.class_graph_1 = self.widget.findChild(pg.PlotWidget, 'aa_classGraph')
        self.class_graph_2 = self.widget.findChild(pg.PlotWidget, 'aa_classGraph_2')
        self.class_graph_3 = self.widget.findChild(pg.PlotWidget, 'aa_classGraph_3')
        self.attribute_graph = self.widget.findChild(pg.PlotWidget, 'aa_attributeGraph')

        # Status window
        self.status_window = self.widget.findChild(QtWidgets.QTextEdit, 'aa_statusWindow')
        self.add_status_message("This mode is for using a Neural Network to annotate Data.")

        # deep rep functions
        self.deep_rep_checkBox = self.widget.findChild(QtWidgets.QCheckBox, "aa_deep_rep_checkBox")

        self.deep_rep_button = self.widget.findChild(QtWidgets.QPushButton, "aa_deep_rep_browse_button")
        self.deep_rep_button.clicked.connect(lambda _: self.browse_deep_rep_files())

    def enable_widgets(self):
        if not self.enabled:
            self.class_graph_1 = Graph(self.class_graph_1, 'class',
                                       interval_lines=False, label='Classes #1')
            self.class_graph_2 = Graph(self.class_graph_2, 'class',
                                       interval_lines=False, label='Classes #2')
            self.class_graph_3 = Graph(self.class_graph_3, 'class',
                                       interval_lines=False, label='Classes #3')
            self.attribute_graph = Graph(self.attribute_graph, 'attribute', interval_lines=False)

            # self.deep_rep_button.setEnabled(True)
            self.network_comboBox.setEnabled(True)

            self.enabled = True

        self.class_graph_1.setup()
        self.class_graph_2.setup()
        self.class_graph_3.setup()
        self.attribute_graph.setup()

        self.reload()
        self.enable_annotate_button()
        self.enable_load_button()

        self.deep_rep_checkBox.setEnabled(False)
        self.deep_rep_checkBox.setChecked(False)
        self.deep_rep_files = []

    def enable_annotate_button(self):
        if self.selected_network > 0 \
                and self.enabled \
                and not self.fixed_window_mode_enabled:
            self.annotate_button.setEnabled(True)
        else:
            self.annotate_button.setEnabled(False)
            # self.annotate_folder_button.setEnabled(False)

    def enable_load_button(self):
        if self.selected_network > 0 \
                and self.enabled \
                and not self.fixed_window_mode_enabled:
            directory = g.settings['saveFinishedPath']
            annotator_id = g.networks[self.selected_network]['annotator_id']

            files_present = True
            for pred_id in range(3):
                file_name = f"{g.windows.file_name.split('.')[0]}_A{annotator_id:0>2}_N{pred_id:0>2}.txt"
                path = directory + os.sep + file_name
                if not os.path.exists(path):
                    files_present = False

            if files_present:
                self.load_predictions_button.setEnabled(True)
            else:
                self.load_predictions_button.setEnabled(False)
        else:
            self.load_predictions_button.setEnabled(False)

    def reload(self):
        frame = self.gui.get_current_frame()
        graphs = [self.class_graph_1, self.class_graph_2, self.class_graph_3]
        for graph in graphs:
            graph.update_frame_lines(play=frame)

        if g.windows is not None \
                and g.windows.windows_1 is not None \
                and len(g.windows.windows_1) > 0:

            windows = [g.windows.windows_1, g.windows.windows_2, g.windows.windows_3]
            for graph, window in zip(graphs, windows):
                graph.reload_classes(window)

            self.select_window_by_frame(frame)
            self.selectWindow(self.current_window)
            self.highlight_class_bar(self.current_window)

    def select_network(self, index):
        """Saves the selected network and tries to activate annotation if one was selected"""
        self.selected_network = index
        if index > 0:
            attributes = g.networks[index]['attributes']
        else:
            attributes = False
        self.deep_rep_button.setEnabled(attributes)
        if not attributes:
            self.deep_rep_checkBox.setChecked(False)
            self.deep_rep_checkBox.setEnabled(False)
        elif self.deep_rep_files:
            self.deep_rep_checkBox.setEnabled(True)

        self.enable_annotate_button()
        self.enable_load_button()

    def annotate(self):
        self.annotate_start_time = time.time()

        self.progress = ProgressDialog(self.gui, "annotating", 6)

        self.annotator = Annotator(self.gui, self.selected_network, self.deep_rep_checkBox.isChecked(),
                                   self.deep_rep_files)
        self.annotator.progress.connect(self.progress.set_step)
        self.annotator.progress_add.connect(self.progress.advance_step)
        self.annotator.nextstep.connect(self.progress.new_step)
        self.annotator.cancel.connect(lambda _: self.cancel_annotation())
        self.annotator.done.connect(lambda _: self.finish_annotation())

        self.attribute_graph.update_attributes(None)
        # for i in range(len(g.attributes)):
        #    self.gui.graphics_controller.attr_bars[i].setOpts(y1=0)

        self.progress.show()
        self.annotator.start()

    def finish_annotation(self):
        self.reload()
        # print("windows_1: ", g.windows.windows_1)
        # print("windows_2: ", g.windows.windows_2)
        # print("windows_3: ", g.windows.windows_3)
        self.time_annotation()

        del self.annotator

    def cancel_annotation(self):
        self.progress.close()
        # self.time_annotation()

    def time_annotation(self):
        annotate_end_time = time.time()
        time_elapsed = int(annotate_end_time - self.annotate_start_time)
        seconds = time_elapsed % 60
        minutes = (time_elapsed // 60) % 60
        hours = time_elapsed // 3600
        # print(time_elapsed)
        self.add_status_message("The annotation took {}:{}:{}".format(hours, minutes, seconds))

    def load_predictions(self):
        g.windows.load_predictions(g.settings['saveFinishedPath'],
                                   g.networks[self.selected_network]['annotator_id'])

        directory = g.settings['saveFinishedPath']
        annotator_id = g.networks[self.selected_network]['annotator_id']
        g.retrieval = RetrievalData.load_retrieval(directory, annotator_id)

        self.reload()
        # print("windows_1: ", g.windows.windows_1)
        # print("windows_2: ", g.windows.windows_2)
        # print("windows_3: ", g.windows.windows_3)

    def new_frame(self, frame):

        classgraphs = [self.class_graph_1, self.class_graph_2, self.class_graph_3]

        for graph in classgraphs:
            graph.update_frame_lines(play=frame)

        if g.windows is not None \
                and g.windows.windows_1 is not None \
                and len(g.windows.windows_1) > 0:

            window_index = self.class_window_index(frame)
            if self.current_window != window_index:
                self.current_window = window_index
                self.selectWindow(self.current_window)
                self.highlight_class_bar(self.current_window)

    def class_window_index(self, frame):
        if frame is None:
            frame = self.gui.get_current_frame()
        for i, window in enumerate(g.windows.windows_1):
            if window[0] <= frame < window[1]:
                return i
        return None

    def select_window_by_frame(self, frame=None):
        """Selects the Window around based on the current Frame shown
        
        """
        if frame is None:
            frame = self.gui.get_current_frame()
        window_index = self.class_window_index(frame)
        if window_index is None:
            window_index = -1
        # if the old and new index is the same do nothing.
        if self.current_window != window_index:
            self.current_window = window_index
            self.selectWindow(window_index)
        else:
            self.current_window = window_index

    def selectWindow(self, window_index: int):
        """Selects the window at window_index"""

        if window_index >= 0:
            self.current_window = window_index

            # needs to update shown attributes and start-, end-lines for top3 graphs
            # start end and attributes are the same in each prediction

            # classgraphs = [self.class_graph_1, self.class_graph_2, self.class_graph_3]
            # for graph in classgraphs:
            # graph.update_frame_lines(start, end)

            _, _, _, attributes = g.windows.windows_1[self.current_window]
            self.attribute_graph.update_attributes(attributes)

    def highlight_class_bar(self, bar_index):

        normal_color = 0.5  # gray
        error_color = 200, 100, 100  # gray-ish red
        selected_color = 'y'  # yellow
        selected_error_color = 255, 200, 50  # orange

        num_windows = len(g.windows.windows_1)

        colors = []
        for i in range(num_windows):
            if g.windows.windows_1[i][3][-1] == 0:
                colors.append(normal_color)
            else:
                colors.append(error_color)

        if bar_index is not None:
            if g.windows.windows_1[bar_index][3][-1] == 0:
                colors[bar_index] = selected_color
            else:
                colors[bar_index] = selected_error_color

        self.class_graph_1.color_class_bars(colors)
        self.class_graph_2.color_class_bars(colors)
        self.class_graph_3.color_class_bars(colors)

    def fixed_windows_mode(self, enable: bool):
        # Controller.fixed_windows_mode(self, enable)
        self.fixed_window_mode_enabled = enable

        self.enable_annotate_button()
        self.enable_load_button()

    def get_start_frame(self) -> int:
        """returns the start of the current window"""
        if g.windows.windows_1 is not None and len(g.windows.windows_1) > 0:
            return g.windows.windows_1[self.current_window][0] + 1
        return self.gui.get_current_frame()

    def browse_deep_rep_files(self):
        current_file_name = g.windows.file_name
        name_parts = current_file_name.split('_')
        subject_id = [s for s in name_parts if 'S' in s][0]
        # print(subject_id)
        paths = QtWidgets.QFileDialog.getOpenFileNames(
            parent=self.gui,
            caption='Please choose annotated files from the same Subject as the current file.',
            directory=g.settings['saveFinishedPath'],
            filter=f'CSV Files (*{subject_id}*norm_data.csv)',
            initialFilter='')[0]
        # print(paths)

        self.deep_rep_files = paths

        if paths:
            file_names = [os.path.split(path)[1][:-14] for path in paths]
            # print(file_names)
            msg = "Selected files for Deep Representation learning:"
            for file in file_names:
                msg += f"\n- {file}"
            self.add_status_message(msg)

            self.deep_rep_checkBox.setEnabled(True)
            self.deep_rep_checkBox.setChecked(True)
        else:
            self.add_status_message("No files selected for Deep Representation learning")
            self.deep_rep_checkBox.setEnabled(False)
            self.deep_rep_checkBox.setChecked(False)


class Annotator(QThread):
    progress = pyqtSignal(int)
    progress_add = pyqtSignal(int)
    nextstep = pyqtSignal(str, int)
    done = pyqtSignal(int)
    cancel = pyqtSignal(int)

    def __init__(self, gui, selected_network, deep_rep=False, paths=None):
        super(Annotator, self).__init__()
        if paths is None:
            paths = []
        self.gui = gui

        self.deep_rep = deep_rep
        self.paths = paths

        self.selected_network = selected_network
        self.network = None
        # self.window_step = g.settings['segmentationWindowStride']
        self.device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")

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
            self.cancel.emit(0)
            self.gui.add_status_message("Something went wrong sorry.")
            raise e
        except FileNotFoundError as e:
            self.network = None
            self.cancel.emit(0)
            self.gui.add_status_message("Could not find the " + g.networks[index]["name"]
                                        + " at " + g.networks_path + g.networks[index]['file_name'])
            return None, None, None
            # raise e

    def run(self):
        self.nextstep.emit("loading network", 1)
        # Load network
        network, config, att_rep = self.load_network(self.selected_network)
        if network is None:
            return
        network.deep_rep = self.deep_rep

        self.nextstep.emit("segmenting", 1 + len(self.paths))
        # Segment Data
        window_length = config['sliding_window_length']

        dataset = LabeledSlidingWindowDataset(g.data.mocap_data, window_length, window_step=window_length)
        if config['labeltype'] == 'attributes':
            g.retrieval = RetrievalData(g.data.mocap_data, window_length, window_step=window_length)
        self.progress_add.emit(1)
        # Making deep representation
        if self.deep_rep:
            network.deep_rep = True
            deep_rep = self.get_deep_representations(self.paths, config, network)
        else:
            deep_rep = None

        # Forward through network
        self.nextstep.emit("annotating", len(dataset))
        label_kind = config['labeltype']
        for i in range(len(dataset)):
            if self.deep_rep:
                label, fc2 = network(dataset.__getitem__(i))
                deep_rep.save_fc2(i, fc2)
            else:
                label = network(dataset.__getitem__(i))
            if label_kind == 'class':
                label = torch.argsort(label, descending=True)[0]
                dataset.save_labels(i, label, label_kind)
            elif label_kind == 'attributes':
                label = label.detach()
                dataset.save_labels(i, label[0], label_kind)
                g.retrieval.save_labels(i, label[0], label_kind)
            else:
                raise Exception
            self.progress.emit(i)

        # Evaluate results
        self.nextstep.emit("evaluating", 1)
        if self.deep_rep:
            deep_rep.predict_labels_from_fc2()
        if att_rep is not None:
            # metrics = Metrics(config, self.device, att_rep)
            metrics = att_rep
        else:
            metrics = None

        windows_1, windows_2, windows_3 = dataset.make_windows(label_kind, metrics, deep_rep)

        # Save windows
        self.nextstep.emit("saving", 1)
        g.windows.windows_1 = windows_1
        g.windows.windows_2 = windows_2
        g.windows.windows_3 = windows_3
        g.windows.save_predictions(g.settings['saveFinishedPath'],
                                   g.networks[self.selected_network]['annotator_id'])
        g.retrieval.save_retrieval(g.settings['saveFinishedPath'],
                                   g.networks[self.selected_network]['annotator_id'])

        self.nextstep.emit("done", 0)
        self.done.emit(0)

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

        for path in paths:

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

            self.progress_add.emit(1)

        dill.dump(deep_rep, open(pickled_deep_rep_path, "wb"))
        return deep_rep
