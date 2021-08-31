"""
Created on 27.07.2021

@author: Erik Altermann
@email: Erik.Altermann@tu-dortmund.de
"""
import numpy as np
from PyQt5 import QtWidgets, QtGui
from os import sep
from .controller import Controller
import global_variables as g
from _functools import reduce

import pyqtgraph as pg
from controllers.controller import Graph


class RetrievalController(Controller):
    def __init__(self, gui):
        super(RetrievalController, self).__init__(gui)

        self.retrieved_list = []
        self.not_filtered_range = (0, 1)  # Used for the distance histogram

        self.setup_widgets()

    def setup_widgets(self):
        self.load_tab(f'..{sep}ui{sep}retrieval_mode.ui', "Retrieval")

        # ----Retrieval preparation----
        self.query_comboBox: QtWidgets.QComboBox = self.widget.findChild(QtWidgets.QComboBox, "qbcComboBox")
        self.query_comboBox.addItems(g.classes)
        self.query_comboBox.currentIndexChanged.connect(self.change_query)

        self.metric_comboBox = self.widget.findChild(QtWidgets.QComboBox, "distanceComboBox")
        self.metric_comboBox.addItems(["cosine", "bce"])
        # self.metric_comboBox.currentTextChanged.connect()

        self.retrieve_button = self.widget.findChild(QtWidgets.QPushButton, "retrievePushButton")
        self.retrieve_button.clicked.connect(lambda _: self.retrieve_list())

        self.none_button = self.widget.findChild(QtWidgets.QPushButton, "nonePushButton")
        self.none_button.clicked.connect(lambda _: self.set_to_none())

        # ----Retrieval video settings----
        # TODO: add these settings to the settings file to save/load them
        self.loop_checkBox = self.widget.findChild(QtWidgets.QCheckBox, "loopCheckBox")
        self.before_lineEdit = self.widget.findChild(QtWidgets.QLineEdit, "beforeLineEdit")
        self.after_lineEdit = self.widget.findChild(QtWidgets.QLineEdit, "afterLineEdit")

        # ----Attribute buttons----
        self.attribute_buttons = [QtWidgets.QCheckBox(text) for text in g.attributes]
        layout2 = self.widget.findChild(QtWidgets.QGroupBox, "attributesGroupBox").layout()

        for button in self.attribute_buttons:
            button.setEnabled(False)
            button.toggled.connect(lambda _: self.move_buttons(layout2, self.attribute_buttons))
            button.clicked.connect(lambda _: self.change_attributes())
        self.move_buttons(layout2, self.attribute_buttons)

        # ----Retrieval Buttons----
        self.accept_button = self.widget.findChild(QtWidgets.QPushButton, "acceptPushButton")
        self.accept_button.clicked.connect(lambda _: self.accept_suggestion())

        self.reject_button = self.widget.findChild(QtWidgets.QPushButton, "rejectPushButton")
        self.reject_button.clicked.connect(lambda _: self.reject_suggestion())

        self.reject_all_button = self.widget.findChild(QtWidgets.QPushButton, "rejectAllPushButton")
        self.reject_all_button.clicked.connect(lambda _: self.reject_all_suggestions())

        # ----Classgraph-----------
        self.class_graph = self.widget.findChild(pg.PlotWidget, 'classGraph')
        self.distance_graph = self.widget.findChild(pg.PlotWidget, 'distanceGraph')
        self.distance_histogram = self.widget.findChild(pg.PlotWidget, 'distanceHistogram')

        # ----Status windows-------
        self.status_window = self.widget.findChild(QtWidgets.QTextEdit, 'statusWindow')
        self.add_status_message("New retrieval mode. This is a WIP")

    def enable_widgets(self):
        self.before_lineEdit.setValidator(QtGui.QIntValidator(0, g.data.number_samples))
        self.after_lineEdit.setValidator(QtGui.QIntValidator(0, g.data.number_samples))

        self.class_graph = Graph(plot_widget=self.class_graph, graph_type="class", label="Classes", interval_lines=True)
        self.class_graph.setup()

        self.distance_graph = Graph(self.distance_graph, 'data', label="score", interval_lines=True,
                                    unit="", AutoSIPrefix=False, y_range=(0, 1))
        self.distance_graph.setup()

        self.distance_histogram = Graph(self.distance_histogram, "histogram", label="histogram", play_line=True)
        self.distance_histogram.setup()

        self.reload()

    def reload(self):
        self.class_graph.reload_classes(g.windows.windows)
        self.class_graph.update_frame_lines(-1000, -1000, self.gui.get_current_frame())
        self.distance_graph.update_frame_lines(-1000, -1000, self.gui.get_current_frame())
        self.distance_graph.update_plot(None)
        self.distance_histogram.update_histogram(None, None)

        if g.retrieval is not None \
                and (self.fixed_window_mode_enabled is None
                     or self.fixed_window_mode_enabled in ["none", "retrieval"]):

            self.none_button.setEnabled(True)

            if self.fixed_window_mode_enabled == "retrieval":
                self.retrieve_button.setEnabled(True)
                self.metric_comboBox.setEnabled(True)
            else:
                self.retrieve_button.setEnabled(False)
                self.metric_comboBox.setEnabled(False)

            if self.retrieved_list:
                self.query_comboBox.setEnabled(True)
                self.loop_checkBox.setEnabled(True)
                self.before_lineEdit.setEnabled(True)
                self.after_lineEdit.setEnabled(True)

                self.accept_button.setEnabled(True)
                self.reject_button.setEnabled(True)
                self.reject_all_button.setEnabled(True)

                # ---- Attribute Buttons ----
                window = g.windows.windows_1[self.get_annotation_index()]
                for button, checked in zip(self.attribute_buttons, window[3]):
                    button.setChecked(checked)
                    button.setEnabled(True)

                # ---- Class Graph ----
                suggestion = self.retrieved_list[0]
                index = suggestion["index"]
                s, e = suggestion["range"]

                self.class_graph.update_frame_lines(s, e)
                self.highlight_class_bar(index)

                # ---- Distance Graph ----
                distances = np.zeros((g.data.number_samples,))
                for suggestion in self.retrieved_list:
                    s_, e_ = suggestion["range"]
                    distances[s_:e_] = suggestion["value"]
                self.distance_graph.update_plot(distances)

                self.distance_graph.update_frame_lines(s, e)

                # ---- Distance Histogram ----
                distances = [item["value"] for item in self.retrieved_list]
                min_x = min(distances)
                max_x = max(distances)
                y_values, x_values = np.histogram(distances, bins=1000, range=(0, 1))

                discard_min = sum([1 for x_value in x_values if x_value < min_x]) - 1
                discard_max = sum([1 for x_value in x_values if x_value > max_x]) - 1
                if discard_max == 0:
                    x_values = x_values[discard_min:]
                    y_values = y_values[discard_min:]
                else:
                    x_values = x_values[discard_min:-discard_max]
                    y_values = y_values[discard_min:-discard_max]

                self.distance_histogram.update_histogram(x_values, y_values, self.not_filtered_range)
                self.distance_histogram.update_frame_lines(play=self.retrieved_list[0]["value"])
            else:

                # if one retrieved list becomes empty other may still have windows left
                # self.query_comboBox.setEnabled(False)
                self.loop_checkBox.setEnabled(False)
                self.before_lineEdit.setEnabled(False)
                self.after_lineEdit.setEnabled(False)

                self.accept_button.setEnabled(False)
                self.reject_button.setEnabled(False)
                self.reject_all_button.setEnabled(False)

                for button in self.attribute_buttons:
                    button.setEnabled(False)
        else:
            self.metric_comboBox.setEnabled(False)
            self.none_button.setEnabled(False)
            self.retrieve_button.setEnabled(False)

            self.query_comboBox.setEnabled(False)
            self.loop_checkBox.setEnabled(False)
            self.before_lineEdit.setEnabled(False)
            self.after_lineEdit.setEnabled(False)

            self.accept_button.setEnabled(False)
            self.reject_button.setEnabled(False)
            self.reject_all_button.setEnabled(False)

            for button in self.attribute_buttons:
                button.setEnabled(False)

    # def setEnabled(self, enable: bool):
    #    print("overwrite setEnabled(self,enable:bool) in", type(self))

    def new_frame(self, frame):
        if self.retrieved_list and self.loop_checkBox.isChecked():
            start, end = self.retrieved_list[0]["range"]
            start = max(0, start - int(self.before_lineEdit.text()))
            end = min(g.data.number_samples - 1, end + int(self.after_lineEdit.text()))
            if not (start <= frame < end):
                frame = start
                self.gui.playback_controller.set_start_frame()
        self.distance_graph.update_frame_lines(play=frame)
        self.class_graph.update_frame_lines(play=frame)

    def get_start_frame(self):
        if self.retrieved_list:
            s, _ = self.retrieved_list[0]["range"]
            return max(0, s - int(self.before_lineEdit.text()))
        else:
            return 0

    def move_buttons(self, layout: QtWidgets.QGridLayout, buttons: list):
        """Moves all the buttons in a layout

        Checked radio/checkbox buttons get moved to the left
        Unchecked buttons get moved to the right

        Arguments:
        ----------
        layout : QGridLayout
            the layout on which the buttons should be
        buttons : list
            a list of QRadioButtons or QCheckBox buttons, that should be moved in the layout
        """

        for i, button in enumerate(buttons):
            if button.isChecked():
                layout.addWidget(button, i + 1, 0)
            else:
                layout.addWidget(button, i + 1, 2)

    def change_attributes(self):
        """Looks which Attribute buttons are checked and saves that to the current window"""

        attributes = []
        for button in self.attribute_buttons:
            if button.isChecked():
                attributes.append(1)
            else:
                attributes.append(0)
        g.windows.change_window(self.retrieved_list[0]["index"], attributes=attributes, save=True)
        self.class_graph.reload_classes(g.windows.windows)
        self.highlight_class_bar(self.retrieved_list[0]["index"])

    def highlight_class_bar(self, bar_index):
        colors = Controller.highlight_class_bar(self, bar_index)
        self.class_graph.color_class_bars(colors)

    def set_to_none(self):

        message = \
            (f"This action will enable fixed-window-size mode.\n"
             f"Any labeling done up to this point will be discarded "
             f"and all windows will be reset to {g.classes[-1]}.\n"
             f"Some features in other modes will be disabled until fixed-window-size mode is disabled again.\n"
             f"In this mode you can choose to accept or reject suggestions for a query class.")

        revision_mode_warning = QtWidgets.QMessageBox.question(
            self.gui, 'Start fixed-window-size mode?', message,
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel,
            QtWidgets.QMessageBox.Cancel)

        if revision_mode_warning == QtWidgets.QMessageBox.Yes:
            error_attr = [0 for _ in range(len(g.attributes))]
            error_attr[-1] = 1
            intervals = [g.retrieval.__range__(i) for i in range(len(g.retrieval))]
            g.windows.windows = [(s, e, len(g.classes) - 1, error_attr) for (s, e) in intervals]
            g.windows.make_backup()
            self.gui.fixed_windows_mode("retrieval")

    def disable_fixed_window_mode(self):
        message = ("This will disable fixed-window-size mode.\n"
                   "Make sure you are finished everything you need to do in this mode.\n"
                   "Next time you activate revision mode your unsaved progress will be lost")

        revision_mode_warning = QtWidgets.QMessageBox.question(
            self.gui, 'Stop revision mode?', message,
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel,
            QtWidgets.QMessageBox.Cancel)

        if revision_mode_warning == QtWidgets.QMessageBox.Yes:
            self.gui.fixed_windows_mode("none")

    def fixed_windows_mode(self, mode: str):
        self.fixed_window_mode_enabled = mode

        if mode == "retrieval":
            self.none_button.clicked.disconnect()
            self.none_button.clicked.connect(lambda _: self.disable_fixed_window_mode())
            self.none_button.setText("Stop retrieval mode")
        elif mode is None or mode == "none":
            self.none_button.clicked.disconnect()
            self.none_button.clicked.connect(lambda _: self.set_to_none())
            self.none_button.setText("Set all to None")
            self.none_button.setEnabled(True)

            self.retrieved_list = []
        else:
            self.none_button.setEnabled(False)

        self.reload()

    def retrieve_list(self):
        distance = self.metric_comboBox.currentText()

        g.retrieval.predict_classes_from_attributes(distance)
        g.retrieval.predict_attribute_reps(distance)
        g.retrieval.reset_filter()
        self.change_query(self.query_comboBox.currentIndex())
        self.reload()

    def change_query(self, class_index: int):
        self.retrieved_list = g.retrieval.retrieve_list(class_index)
        values = sorted([item["value"] for item in self.retrieved_list])
        self.not_filtered_range = (values[0], values[-1])
        self.retrieved_list = g.retrieval.filter_not_none_class(self.retrieved_list, class_index)
        self.reload()

    def accept_suggestion(self):
        _, _, _, a = g.windows.windows_1[self.get_annotation_index(0)]
        index = self.retrieved_list[0]["index"]
        s, e = self.retrieved_list[0]["range"]
        c = self.query_comboBox.currentIndex()
        g.windows.windows[index] = (s, e, c, a)
        g.retrieval.remove_suggestion(self.retrieved_list[0], None)
        self.change_attributes()  # Change attributes as seen on gui
        self.retrieved_list.pop(0)
        self.retrieved_list = g.retrieval.prioritize_neighbors(self.retrieved_list, index)
        self.reload()

    def reject_suggestion(self):
        class_index = self.query_comboBox.currentIndex()
        g.retrieval.remove_suggestion(self.retrieved_list[0], class_index)
        self.retrieved_list.pop(0)
        self.reload()

    def reject_all_suggestions(self):
        class_index = self.query_comboBox.currentIndex()
        g.retrieval.remove_suggestion(None, class_index)
        self.retrieved_list = []
        self.reload()

    def get_annotation_index(self, retrieval_index=0):
        s, e = self.retrieved_list[retrieval_index]["range"]
        m = (s + e) / 2
        for i in range(len(g.windows.windows_1)):
            window = g.windows.windows_1[i]
            if window[0] <= m < window[1]:
                return i
