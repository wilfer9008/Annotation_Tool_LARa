"""
Created on 09.07.2020

@author: Erik Altermann
@email: Erik.Altermann@tu-dortmund.de
"""
from PyQt5.QtWidgets import QWidget
from PyQt5 import uic
import pyqtgraph as pg
# import pyqtgraph.opengl as gl
from pyqtgraph.functions import mkPen

import global_variables as g


class Controller:
    def __init__(self, gui):
        self.gui = gui
        self.widget = None
        self.enabled = False
        self.fixed_window_mode_enabled = False
        self.status_window = None
        # self.setup_widgets()

    def load_tab(self, ui_path: str, name: str):
        self.widget = QWidget()
        self.gui.tab_widget.addTab(uic.loadUi(ui_path, self.widget), name)

    def setup_widgets(self):
        print("overwrite setup_widgets(self) in", type(self))

    def enable_widgets(self):
        print("overwrite enable_widgets(self) in", type(self))

    def reload(self):
        print("overwrite reload(self) in", type(self))

    def setEnabled(self, enable: bool):
        print("overwrite setEnabled(self,enable:bool) in", type(self))

    def new_frame(self, frame):
        print("overwrite new_frame(self,frame) in", type(self))

    def fixed_windows_mode(self, enable: bool):
        print("overwrite fixed_windows_mode(self,enable) in", type(self))

    def get_start_frame(self):
        print("overwrite fixed_windows_mode(self,enable) in", type(self))

    def add_status_message(self, msg):
        self.status_window.append(msg + '\n')

    def class_window_index(self, frame) -> [int, None]:
        if frame is None:
            frame = self.gui.get_current_frame()
        for i, window in enumerate(g.windows.windows):
            if window[0] <= frame < window[1]:
                return i
        return -1

    def highlight_class_bar(self, bar_index):
        """Generates a list of colors for use in class_graphs
        
        When using this method in subclasses of controller: 
        use super to get the colors and call color_class_bars(colors) on needed graphs
        """
        normal_color = 0.5  # gray
        error_color = 200, 100, 100  # gray-ish red
        selected_color = 'y'  # yellow
        selected_error_color = 255, 200, 50  # orange

        num_windows = len(g.windows.windows)

        colors = []
        for i in range(num_windows):
            if g.windows.windows[i][3][-1] == 0:
                colors.append(normal_color)
            else:
                colors.append(error_color)

        if bar_index is not None:
            if g.windows.windows[bar_index][3][-1] == 0:
                colors[bar_index] = selected_color
            else:
                colors[bar_index] = selected_error_color
        return colors


class ControlledWidget(QWidget):
    def __init__(self, controller, ui_path):
        super(ControlledWidget, self).__init__()
        uic.loadUi(ui_path, self)
        self.controller = controller

    def new_frame(self, frame):
        self.controller.new_frame(frame)

    def add_status_message(self, msg):
        self.controller.add_status_message(msg)

    def get_start_start_frame(self):
        return self.controller.get_start_frame()

    def reload(self):
        self.controller.reload()

    def revision_mode(self, enable: bool):
        self.controller.fixed_windows_mode(enable)


class Graph:
    def __init__(self, plot_widget, graph_type, **kwargs):
        """
        
        graph_type: 'data','class','attribute'
        
        kwargs = {'label','unit','AutoSIPrefix',interval_lines,
                  'number_classes'}
        
        """

        if graph_type not in ['data', 'class', 'attribute', 'state', 'histogram']:  # , 'heatmap']:
            raise ValueError(f"Received unknown graph_type: {graph_type}. Accepted graph_types are: "
                             "'data', 'class', 'attribute' and 'state'.")

        self.graph = plot_widget
        self.graph_type = graph_type
        self.kwargs = kwargs

        self.graph.setMouseEnabled(False, False)

        self.class_bars = []
        self.start_line = None
        self.end_line = None
        self.play_line = None

    def setup(self):  # TODO add heatmap and its methods

        self.graph.clear()
        self.graph.plot([])

        self.class_bars = []

        # Graphs that need the play_line or interval lines
        if self.graph_type in ['class', 'data', 'state']:
            if self.kwargs['interval_lines']:
                self.start_line = pg.InfiniteLine(0, label='start', labelOpts={'anchors': [(1, 1.5), (0, 1.5)]})
                self.graph.addItem(self.start_line)

                self.end_line = pg.InfiniteLine(0, label='end', labelOpts={'anchors': [(0, 1.5), (1, 1.5)]})
                self.graph.addItem(self.end_line)

            self.play_line = pg.InfiniteLine(0, pen=mkPen(0, 255, 0, 127))
            self.graph.addItem(self.play_line)

            self.graph.setXRange(0, g.data.number_samples, padding=0.02)

        elif self.graph_type == 'attribute':
            if 'label' in self.kwargs.keys():
                self.graph.getAxis('left').setLabel(text=self.kwargs['label'], units='')
            else:
                self.graph.getAxis('left').setLabel(text='Attributes', units='')
            for i, attribute in enumerate(g.attributes):
                bar = pg.BarGraphItem(x0=[i], x1=i + 1, y0=0, y1=0, name=attribute)
                self.class_bars.append(bar)
                self.graph.addItem(bar)
                self.graph.setYRange(0, 1, padding=0.1)
                label = pg.TextItem(text=attribute, color='b', anchor=(0, 0), border=None, fill=None, angle=-90,
                                    rotateAxis=None)
                label.setPos(i + 1, 1)

                self.graph.addItem(label)
            self.graph.setXRange(0, len(g.attributes), padding=0.02)

        if self.graph_type == 'data':
            self.graph.getAxis('left').setLabel(text=self.kwargs['label'],
                                                units=self.kwargs['unit'])
            self.graph.getAxis('left').enableAutoSIPrefix(self.kwargs['AutoSIPrefix'])

        if self.graph_type == 'class':
            if 'label' in self.kwargs.keys():
                self.graph.getAxis('left').setLabel(text=self.kwargs['label'], units='')
            else:
                self.graph.getAxis('left').setLabel(text='Classes', units='')
            self.graph.setYRange(0, len(g.classes) + 1, padding=0)

        if self.graph_type == 'state':
            if 'label' in self.kwargs.keys():
                self.graph.getAxis('left').setLabel(text=self.kwargs['label'], units='')
            else:
                self.graph.getAxis('left').setLabel(text='States', units='')
            self.graph.setYRange(0, len(g.states) + 1, padding=0)

        if self.graph_type == 'histogram':
            if 'label' in self.kwargs.keys():
                self.graph.getAxis('left').setLabel(text=self.kwargs['label'], units='')
            # self.graph.setXRange(0, g.data.number_samples, padding=0.02)
            # self.graph.plot([0, 1], [0],  # pen=(127, 127, 127, 255), brush=(200, 200, 200, 100),
            #         stepMode=True, fillLevel=0, fillOutline=True)

    def update_frame_lines(self, start=None, end=None, play=None):
        """Updates the framelines of the graph
        
        Cannot be used with attribute graphs
        """
        if self.kwargs['interval_lines']:
            if start is not None:
                self.start_line.setValue(start)
            if end is not None:
                self.end_line.setValue(end)
        if play is not None:
            self.play_line.setValue(play)

    def update_plot(self, plot_data):
        """Updates Data for plotting on the graph.
        
        Should only be used with data graphs
        """

        if plot_data is not None:
            self.graph.listDataItems()[0].setData(plot_data)
        else:
            self.graph.listDataItems()[0].setData([], [])

    def update_histogram(self, x, y):
        """Updates Histogram for plotting on the graph.

        Should only be used with histogram graphs
        """
        self.setup()
        if x is not None and y is not None:
            self.graph.plot(x, y,  # pen=(127, 127, 127, 255), brush=(200, 200, 200, 100),
                            stepMode=True, fillLevel=0, fillOutline=True)

    def add_class(self, start, end, class_index, attributes):
        """Adds a class window to the graph.
        
        Should only be used with class graphs
        """

        bar = pg.BarGraphItem(x0=[start], x1=end, y0=0, y1=class_index + 1)
        self.class_bars.append(bar)
        if attributes[-1] == 1:
            bar.setOpts(brush=pg.mkBrush(200, 100, 100))
        self.graph.addItem(bar)

    def reload_classes(self, windows):
        """Reloads all class windows in the graph.
        
        Should only be used with class graphs
        """
        self.setup()

        for start, end, class_index, attributes in windows:
            self.add_class(start, end, class_index, attributes)

    def add_state(self, start, end, state_index=None):
        """Adds a state window to the graph.

           Should only be used with state graphs
        """
        if state_index is None:
            state_index = 0
        bar = pg.BarGraphItem(x0=[start], x1=end, y0=0, y1=state_index)
        self.class_bars.append(bar)
        #    bar.setOpts(brush=pg.mkBrush(200, 100, 100))
        self.graph.addItem(bar)

    def reload_states(self, state_windows):
        """Reloads a state windows in the graph.

        Should only be used with state graphs
        """
        self.setup()

        for start, end, state_index, _ in state_windows:
            self.add_state(start, end, state_index)

    def update_attributes(self, attributes=None):
        """Updates attribute bars in graph
        
        Should only be used with attribute graphs
        """
        if attributes is not None:
            for i in range(len(self.class_bars)):
                self.class_bars[i].setOpts(y1=attributes[i])
        else:
            for i in range(len(self.class_bars)):
                self.class_bars[i].setOpts(y1=0)

    def color_class_bars(self, colors=None):
        """Colors each classbar with corresponding color in colors
        
        Can be used with attribute graph, but should only be used with class graphs
        """
        for bar, color in zip(self.class_bars, colors):
            bar.setOpts(brush=pg.mkBrush(color))

    def update_kwargs(self, **kwargs):
        """
        for kwargs explanation see __init__()
        """
        for k, v in kwargs.items():
            self.kwargs[k] = v
        self.setup()
