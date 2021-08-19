"""
Created on 09.07.2020

@author: Erik Altermann
@email: Erik.Altermann@tu-dortmund.de
"""

from PyQt5 import QtWidgets, QtGui
from os import sep
import pyqtgraph as pg

from dialogs import SaveClassesDialog, SaveAttributesDialog

from .controller import Controller, Graph

import global_variables as g


class ManualAnnotationController(Controller):
    def __init__(self, gui):
        super(ManualAnnotationController, self).__init__(gui)
        self.current_window = -1
        self.setup_widgets()

    def setup_widgets(self):
        self.load_tab(f'..{sep}ui{sep}manual_annotation_mode.ui', "Manual Annotation")

        self.start_line_edit = self.widget.findChild(QtWidgets.QLineEdit, 'startLineEdit')
        self.end_line_edit = self.widget.findChild(QtWidgets.QLineEdit, 'endLineEdit')

        self.set_current_frame_button = self.widget.findChild(QtWidgets.QPushButton, 'setCurrentFrameButton')
        self.set_current_frame_button.clicked.connect(lambda _: self.updateEndLineEdit(None))
        self.end_line_edit.returnPressed.connect(lambda: self.updateEndLineEdit(self.end_line_edit.text()))

        self.save_labels_button = self.widget.findChild(QtWidgets.QPushButton, 'saveLabelsButton')
        self.save_labels_button.clicked.connect(lambda _: self.gui.pause())
        self.save_labels_button.clicked.connect(lambda _: self.saveLabel())

        self.verify_labels_button = self.widget.findChild(QtWidgets.QPushButton, 'verifyLabelsButton')
        self.verify_labels_button.clicked.connect(lambda _: self.gui.pause())
        self.verify_labels_button.clicked.connect(lambda _: self.verify_labels())

        self.combobox = self.widget.findChild(QtWidgets.QComboBox, 'jointSelectionBox')

        joint_graph_names = ['jointGraphRX', 'jointGraphRY', 'jointGraphRZ', 'jointGraphTX', 'jointGraphTY',
                             'jointGraphTZ']
        self.joint_graphs = []
        for graph in joint_graph_names:
            self.joint_graphs.append(self.widget.findChild(pg.PlotWidget, graph))

        self.class_graph = self.widget.findChild(pg.PlotWidget, 'classGraph')

        self.status_window = self.widget.findChild(QtWidgets.QTextEdit, 'ma_statusWindow')
        self.add_status_message("Please read the Annotation Guidelines before beginning.")
        self.add_status_message("If you already did start by opening a new unlabeled file or loading your progress.")

    def enable_widgets(self):
        if self.enabled is False:
            # self.startLineEdit.setEnabled(True) #Stays disabled since start is always next unlabeled frame.
            self.end_line_edit.setEnabled(True)
            self.set_current_frame_button.setEnabled(True)
            self.save_labels_button.setEnabled(True)
            self.verify_labels_button.setEnabled(True)
            self.combobox.setEnabled(True)
            self.combobox.addItems(g.data.body_segments.values())
            self.combobox.currentTextChanged.connect(self.update_joint_graphs)

            self.class_graph = Graph(self.class_graph, 'class',
                                     interval_lines=True)

            # Data graphs need kwargs: 'label','unit','AutoSIPrefix','number_samples'
            label = ['RX', 'RY', 'RZ', 'TX', 'TY', 'TZ']
            unit = ['deg', 'deg', 'deg', 'mm', 'mm', 'mm']
            AutoSIPrefix = [False, False, False, True, True, True]
            for i in range(6):
                self.joint_graphs[i] = Graph(self.joint_graphs[i], 'data',
                                             label=label[i],
                                             unit=unit[i],
                                             AutoSIPrefix=AutoSIPrefix[i],
                                             interval_lines=True)

            self.enabled = True

        self.class_graph.setup()
        for i in range(6):
            self.joint_graphs[i].setup()

        self.reload()

        self.update_joint_graphs(self.combobox.currentText())

        if len(g.windows.windows) > 0:
            _, end, _, _ = g.windows.windows[-1]
            end += 1
        else:
            end = 1
        self.start_line_edit.setText(str(end))  # the start of a windows is the end of the last window.

        self.end_line_edit.setText(str(end))
        self.end_line_edit.setValidator(QtGui.QIntValidator(0, g.data.number_samples))

    def reload(self):
        """reloads start line edit.
        
        called when switching to manual annotation mode
        """

        self.class_graph.reload_classes(g.windows.windows)

        if self.enabled and len(g.windows.windows) > 0:
            start = g.windows.windows[-1][1] + 1
            self.start_line_edit.setText(str(start))
            self.end_line_edit.setText(str(start))
            self.update_frame_lines(start, start, self.gui.get_current_frame())

    def update_joint_graphs(self, joint):
        joint_index = g.data.body_segments_reversed[joint]
        if joint_index > -1:
            for i, graph in enumerate(self.joint_graphs):
                graph.update_plot(g.data.mocap_data[:, joint_index * 6 + i])
        else:
            # print([type(x) for x in self.joint_graphs])
            for graph in self.joint_graphs:
                graph.update_plot(None)

    def new_frame(self, frame):
        self.update_frame_lines(play=frame)

        window_index = self.class_window_index(frame)
        if self.current_window != window_index:
            self.current_window = window_index
            self.highlight_class_bar(window_index)

    def highlight_class_bar(self, bar_index):
        colors = Controller.highlight_class_bar(self, bar_index)

        self.class_graph.color_class_bars(colors)

    def update_frame_lines(self, start=None, end=None, play=None):
        self.class_graph.update_frame_lines(start, end, play)
        for graph in self.joint_graphs:
            graph.update_frame_lines(start, end, play)

    def saveLabel(self):
        # print("save")
        start = int(self.start_line_edit.text())
        end = int(self.end_line_edit.text())
        if start + 50 < end or end == g.data.number_samples:
            dlg = SaveClassesDialog(self.gui)
            class_index = dlg.exec_()
            if class_index > -1:
                dlg = SaveAttributesDialog(self.gui)
                attribute_int = dlg.exec_()
                if attribute_int > -1:
                    format_string = '{0:0' + str(len(g.attributes)) + 'b}'
                    attributes = [(x == '1') + 0 for x in list(format_string.format(attribute_int))]
                    g.windows.save_window(start - 1, end, class_index,
                                       attributes)  # Subtracting 1 from start to index windows from 0.
                    # End is the same because indexing a:b is equivalent to [a,b[.
                    self.start_line_edit.setText(str(end + 1))  # End+1 because next window will save as start-1=end.
                    self.end_line_edit.setText(str(end + 1))

                    self.class_graph.add_class(start, end, class_index, attributes)
                    self.update_frame_lines(end, end, self.gui.get_current_frame())

        else:
            self.add_status_message(
                "Please make sure that the Labelwindow is bigger than 50 Frames."
                + "A size of at least ~100 Frames is recommended")

    def verify_labels(self):
        self.add_status_message("Verifying labels")
        verified = True
        last_window_end = 0
        failed_test = []
        for window in g.windows.windows:
            window_start = window[0]
            window_end = window[1]
            window_class = window[2]
            window_attributes = window[3]

            if not last_window_end == window_start:
                failed_test.append(1)
                verified = False
            if window_start >= window_end:
                failed_test.append(2)
                verified = False
            if not (0 <= window_class < len(g.classes)):
                failed_test.append(3)
                verified = False
            if len(window_attributes) is not len(g.attributes):
                failed_test.append(4)
                verified = False

            last_window_end = window_end
        if not (last_window_end == g.data.number_samples):
            failed_test.append(5)
            verified = False

        failed_test = list(set(failed_test))

        if verified:
            self.add_status_message(
                "Labels are verified! You can now save your labels. Please choose the folder for labeled sequences"
                + "when saving.")
            self.gui.activate_save_button()
        else:
            if 1 in failed_test:
                self.add_status_message("Error: There is a gap between windows.\nPlease contact someone for help.")
            if 2 in failed_test:
                self.add_status_message(
                    "Error: One of the windows ends before it begins.\nPlease contact someone for help.")
            if 3 in failed_test:
                self.add_status_message(
                    "Error: There is a window with an invalid class.\nPlease contact someone for help.")
            if 4 in failed_test:
                self.add_status_message(
                    "Error: Some windows have the wrong number of Attributes.\nPlease contact someone for help.")
            if 5 in failed_test:
                self.add_status_message("Please Finish annotating before verifying")

    def updateEndLineEdit(self, end=None):
        current_frame = self.gui.get_current_frame() + 1
        if end is None:
            current_frame = self.gui.get_current_frame() + 1
            self.end_line_edit.setText(str(current_frame))
            # self.gui.updateFrameLines(current_frame)
            self.update_frame_lines(end=current_frame)

        else:
            self.end_line_edit.setText(str(end))
            # self.gui.updateFrameLines(int(end))
            self.update_frame_lines(end=current_frame)

    def get_start_frame(self):
        return int(self.start_line_edit.text())

    def fixed_windows_mode(self, enable: bool):
        # Controller.fixed_windows_mode(self, enable)
        self.revision_mode_enabled = enable
        self.save_labels_button.setEnabled(not enable)
