"""
Created on 20.01.2021

@author: Erik Altermann
@email: Erik.Altermann@tu-dortmund.de
"""
from os import sep

import pyqtgraph as pg
from PyQt5 import QtWidgets

import global_variables as g
from .controller import Controller, Graph


class StateCorrectionController(Controller):
    def __init__(self, gui):
        super(StateCorrectionController, self).__init__(gui)

        self.current_window = -1

        self.setup_widgets()
        self.gui.activate_save_button()

    def setup_widgets(self):
        self.load_tab(f'..{sep}ui{sep}state_correction_mode.ui', "State Correction")

        # ----Labels----
        self.current_window_label = self.widget.findChild(QtWidgets.QLabel, "sc_current_window_label")
        self.current_class_label = self.widget.findChild(QtWidgets.QLabel, "sc_current_class_label")

        # ----Scrollbars----
        self.scrollBar = self.widget.findChild(QtWidgets.QScrollBar, "sc_scrollBar")
        self.scrollBar.valueChanged.connect(self.selectWindow)

        # ----LineEdits----
        # self. = self.widget.get_widget(QtWidgets.QLineEdit,"")
        self.start_lineEdit = self.widget.findChild(QtWidgets.QLineEdit, "sc_start_lineEdit")
        self.end_lineEdit = self.widget.findChild(QtWidgets.QLineEdit, "sc_end_lineEdit")

        # ----Buttons----
        self.window_by_frame_button = self.widget.findChild(QtWidgets.QPushButton, "sc_window_by_frame_button")
        self.window_by_frame_button.clicked.connect(lambda _: self.select_window_by_frame())

        # ----State buttons----
        self.state_buttons = [QtWidgets.QRadioButton(text) for text in g.states]
        self.state_layout = self.widget.findChild(QtWidgets.QGroupBox, "sc_statesGroupBox").layout()

        for button in self.state_buttons:
            button.setEnabled(False)
            button.toggled.connect(lambda _: self.move_buttons(self.state_layout, self.state_buttons))
            button.clicked.connect(lambda _: self.change_state())
        self.move_buttons(self.state_layout, self.state_buttons)

        # ----State graph-----------
        self.state_graph = self.widget.findChild(pg.PlotWidget, 'sc_classGraph')

        # ----Status windows-------
        self.status_window = self.widget.findChild(QtWidgets.QTextEdit, 'sc_statusWindow')
        self.add_status_message("Here you can correct wrong Labels.")

    def enable_widgets(self):
        """"""

        if not self.enabled:
            self.state_graph = Graph(self.state_graph, 'state', interval_lines=True)
            self.enabled = True

            for button in self.state_buttons:
                button.setEnabled(True)

            self.window_by_frame_button.setEnabled(True)
            self.scrollBar.setEnabled(True)

        self.state_graph.setup()
        self.state_graph.reload_states(g.windows.windows)

        self.reload()

    def reload(self):
        """reloads all window information

                called when switching to label correction mode
                """
        # print("reloading LCC")
        self.state_graph.reload_states(g.windows.windows)

        self.state_graph.update_frame_lines(play=self.gui.get_current_frame())

        if self.enabled and len(g.windows.windows) > 0:
            self.select_window_by_frame()
            self.selectWindow(self.current_window)

    def new_frame(self, frame):
        self.state_graph.update_frame_lines(play=frame)
        window_index = self.class_window_index(frame)
        if self.current_window != window_index:
            self.current_window = window_index
            self.highlight_class_bar(window_index)
            self.selectWindow(window_index)

    def get_start_frame(self) -> int:
        """returns the start of the current window"""
        if len(g.windows.windows) > 0:
            return g.windows.windows[self.current_window][0] + 1
        return 1

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

    def change_state(self):
        for i, button in enumerate(self.state_buttons):
            if button.isChecked():

                g.windows.change_window(self.current_window, i, True)
        self.state_graph.reload_states(g.windows.windows)
        self.selectWindow(self.current_window)
        self.state_graph.update_frame_lines(play=self.gui.get_current_frame())
        self.highlight_class_bar(self.current_window)

    def selectWindow(self, window_index: int):
        """Selects the window at window_index"""
        if window_index >= 0:
            self.current_window = window_index
        else:
            self.current_window = len(g.windows.windows) + window_index

        self.scrollBar.setRange(0, len(g.windows.windows) - 1)
        self.scrollBar.setValue(self.current_window)

        window = g.windows.windows[self.current_window]
        self.current_window_label.setText(f"Current Window: {self.current_window + 1}/{len(g.windows.windows)}")
        self.current_class_label.setText(f"Current Class: {g.classes[window[3]]}")
        self.start_lineEdit.setText(str(window[0] + 1))
        self.end_lineEdit.setText(str(window[1] + 1))

        self.state_buttons[window[2]].setChecked(True)

        self.highlight_class_bar(window_index)
        self.state_graph.update_frame_lines(start=window[0], end=window[1])

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

    def highlight_class_bar(self, bar_index):
        # colors = Controller.highlight_class_bar(self, bar_index)
        # self.state_graph.color_class_bars(colors)
        pass