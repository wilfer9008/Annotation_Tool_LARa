'''
Created on 09.07.2020

@author: Erik Altermann
@email: Erik.Altermann@tu-dortmund.de
'''

from PyQt5 import QtWidgets, QtGui

from .controller import Controller
from data_management import Data_processor
import global_variables as g
from _functools import reduce

import pyqtgraph as pg
from controllers.controller import Graph



class Label_Correction_Controller(Controller):
    def __init__(self,gui):
        super(Label_Correction_Controller, self).__init__(gui)
        
        self.was_enabled_once = False

        
        #self.windows = []
        self.current_window = -1
        
        self.setup_widgets()
        
    def setup_widgets(self):
        #----Labels----
        self.current_window_label = self.gui.get_widget(QtWidgets.QLabel,"lc_current_window_label")
        
        #----Scrollbars----
        self.scrollBar = self.gui.get_widget(QtWidgets.QScrollBar,"lc_scrollBar")
        self.scrollBar.valueChanged.connect(self.selectWindow)
        
        #----LineEdits----
        #self. = self.gui.get_widget(QtWidgets.QLineEdit,"")
        self.split_at_lineEdit      = self.gui.get_widget(QtWidgets.QLineEdit,"lc_split_at_lineEdit")
        self.move_start_lineEdit    = self.gui.get_widget(QtWidgets.QLineEdit,"lc_move_start_lineEdit")
        self.move_end_lineEdit      = self.gui.get_widget(QtWidgets.QLineEdit,"lc_move_end_lineEdit")
        
        self.start_lineEdit         = self.gui.get_widget(QtWidgets.QLineEdit,"lc_start_lineEdit")
        self.end_lineEdit           = self.gui.get_widget(QtWidgets.QLineEdit,"lc_end_lineEdit")
        
        #----Buttons----        
        self.merge_previous_button      = self.gui.get_widget(QtWidgets.QPushButton,"lc_merge_previous_button")
        self.merge_previous_button.clicked.connect(lambda _: self.merge_previous())
        self.merge_next_button          = self.gui.get_widget(QtWidgets.QPushButton,"lc_merge_next_button")
        self.merge_next_button.clicked.connect(lambda _:self.merge_next())
        self.merge_all_button           = self.gui.get_widget(QtWidgets.QPushButton,"lc_merge_all_button")
        self.merge_all_button.clicked.connect(lambda _:self.merge_all_adjacent())
        
        self.split_at_button            = self.gui.get_widget(QtWidgets.QPushButton,"lc_split_at_button")
        self.split_at_button.clicked.connect(lambda _:self.split())
        self.move_start_button          = self.gui.get_widget(QtWidgets.QPushButton,"lc_move_start_button")
        self.move_start_button.clicked.connect(lambda _:self.move_start())
        self.move_end_button            = self.gui.get_widget(QtWidgets.QPushButton,"lc_move_end_button")
        self.move_end_button.clicked.connect(lambda _:self.move_end())
        
        self.set_to_frame_split_button  = self.gui.get_widget(QtWidgets.QPushButton,"lc_set_frame_split_button")
        self.set_to_frame_split_button.clicked.connect(lambda _:self.split_at_lineEdit.setText(str(self.gui.getCurrentFrame()+1)))
        self.set_to_frame_start_button  = self.gui.get_widget(QtWidgets.QPushButton,"lc_set_frame_start_button")
        self.set_to_frame_start_button.clicked.connect(lambda _:self.move_start_lineEdit.setText(str(self.gui.getCurrentFrame()+1)))
        self.set_to_frame_end_button    = self.gui.get_widget(QtWidgets.QPushButton,"lc_set_frame_end_button")
        self.set_to_frame_end_button.clicked.connect(lambda _:self.move_end_lineEdit.setText(str(self.gui.getCurrentFrame()+1)))
        self.set_to_start_button        = self.gui.get_widget(QtWidgets.QPushButton,"lc_set_start_button")
        self.set_to_start_button.clicked.connect(lambda _:self.move_start_lineEdit.setText(str(g.data.windows[self.current_window][0]+1)))
        self.set_to_end_button          = self.gui.get_widget(QtWidgets.QPushButton,"lc_set_end_button")
        self.set_to_end_button.clicked.connect(lambda _:self.move_end_lineEdit.setText(str(g.data.windows[self.current_window][1]+1)))
        
        self.window_by_frame_button     = self.gui.get_widget(QtWidgets.QPushButton,"lc_window__by_frame_button")
        self.window_by_frame_button.clicked.connect(lambda _:self.select_window_by_frame())
        
        #----Class buttons----
        self.classButtons = [QtWidgets.QRadioButton(text) for text in Data_processor.classes]
        self.class_layout = self.gui.get_widget(QtWidgets.QGroupBox,"classesGroupBox").layout()
        
        for button in self.classButtons:
            button.setEnabled(False)
            button.toggled.connect(lambda _: self.move_buttons(self.class_layout, self.classButtons))
            button.clicked.connect(lambda _: self.changeClass())
        self.move_buttons(self.class_layout, self.classButtons)
            
        
        #----Attribute buttons----
        self.attributeButtons = [QtWidgets.QCheckBox(text) for text in Data_processor.attributes]
        layout2 = self.gui.get_widget(QtWidgets.QGroupBox,"attributesGroupBox").layout()
        
        for button in self.attributeButtons:
            button.setEnabled(False)
            button.toggled.connect(lambda _: self.move_buttons(layout2, self.attributeButtons))
            button.clicked.connect(lambda _: self.changeAttributes())
        self.move_buttons(layout2, self.attributeButtons)
        
        #----Classgraph-----------
        self.class_graph = self.gui.get_widget(pg.PlotWidget, 'lc_classGraph')
        
        #----Status windows-------
        self.statusWindow = self.gui.get_widget(QtWidgets.QTextEdit, 'lc_statusWindow')
        self.add_status_message("Here you can correct wrong Labels.")
        
        
        
    def enable_widgets(self):
        """"""
        if not self.was_enabled_once:
            self.class_graph = Graph(self.class_graph, 'class', interval_lines = False)
            self.was_enabled_once = True
            
        self.split_at_lineEdit.setValidator(QtGui.QIntValidator(0,g.data.number_samples+1))
        self.move_start_lineEdit.setValidator(QtGui.QIntValidator(0,g.data.number_samples+1))
        self.move_end_lineEdit.setValidator(QtGui.QIntValidator(0,g.data.number_samples+1))
        
        
        self.class_graph.setup()
        self.class_graph.reload_classes(g.data.windows)
        
        self.reload()
    
    def reload(self):
        """reloads all window information
        
        called when switching to label correction mode
        """
        #print("reloading LCC")
        self.class_graph.reload_classes(g.data.windows)
        
        self.update_frame_lines(self.gui.getCurrentFrame())
        
        if g.data is not None and g.data.windows.__len__() >0:
            self.set_enabled(True)
            self.select_window_by_frame()
            self.selectWindow(self.current_window)
            
        else:
            self.set_enabled(False)
            
    def set_enabled(self,enable:bool):
        """Turns the Widgets of Label Correction Mode on or off based on the enable parameter
        
        Arguments:
        ----------
        enable : bool
            If True and widgets were disabled, the widgets get enabled.
            If False and widgets were enabled, the widgets get disabled.
            Otherwise does nothing.
        ----------
        
        """
        #print("lcm.set_enabled:",\
        #      "\n\t self.enabled:",self.enabled,\
        #      "\n\t enable:",enable,\
        #      "\n\t revision:",self.revision_mode_enabled)
        if self.revision_mode_enabled:
            self.split_at_lineEdit.setEnabled(False)
            self.move_start_lineEdit.setEnabled(False)
            self.move_end_lineEdit.setEnabled(False)
            
            self.merge_previous_button.setEnabled(False)
            self.merge_next_button.setEnabled(False)
            self.merge_all_button.setEnabled(False)
        
            self.split_at_button.setEnabled(False)
            self.move_start_button.setEnabled(False)
            self.move_end_button.setEnabled(False)
        
            self.set_to_frame_split_button.setEnabled(False)
            self.set_to_frame_start_button.setEnabled(False)
            self.set_to_frame_end_button.setEnabled(False)
            self.set_to_start_button.setEnabled(False)
            self.set_to_end_button.setEnabled(False)                
        else:
            self.split_at_lineEdit.setEnabled(enable)
            self.move_start_lineEdit.setEnabled(enable)
            self.move_end_lineEdit.setEnabled(enable)
            
            self.merge_previous_button.setEnabled(enable)
            self.merge_next_button.setEnabled(enable)
            self.merge_all_button.setEnabled(enable)
                        
            self.split_at_button.setEnabled(enable)
            self.move_start_button.setEnabled(enable)
            self.move_end_button.setEnabled(enable)
            
            self.set_to_frame_split_button.setEnabled(enable)
            self.set_to_frame_start_button.setEnabled(enable)
            self.set_to_frame_end_button.setEnabled(enable)
            self.set_to_start_button.setEnabled(enable)
            self.set_to_end_button.setEnabled(enable)
            
            
            
        if not (self.enabled == enable):
            #Only reason why it might be disabled is that there were no windows
            #Therefore setting the current window to 0 as this mode is enabled 
            #as soon as there is at least one window
            self.enabled = enable
            for button in self.classButtons:
                button.setEnabled(enable)
            for button in self.attributeButtons:
                button.setEnabled(enable)
                
            self.window_by_frame_button.setEnabled(enable)
            
            self.scrollBar.setEnabled(enable)
            
            
    def selectWindow(self,window_index:int):
        """Selects the window at window_index"""        
        if window_index >= 0:
            self.current_window = window_index
        else:
            self.current_window = g.data.windows.__len__() + window_index
            
        self.scrollBar.setRange(0,g.data.windows.__len__()-1)
        self.scrollBar.setValue(self.current_window)
            
        window = g.data.windows[self.current_window]
        self.current_window_label.setText("Current Window: "+
                str(self.current_window+1)+"/"+str(g.data.windows.__len__()))
        self.start_lineEdit.setText(str(window[0]+1))
        self.end_lineEdit.setText(str(window[1]+1))
        
        self.classButtons[window[2]].setChecked(True)
        for button, checked in zip(self.attributeButtons,window[3]):
            button.setChecked(checked)
        
        if self.revision_mode_enabled:
            top_buttons = [g.data.windows_1[window_index][2],
                           g.data.windows_2[window_index][2],
                           g.data.windows_3[window_index][2]]
            for i,name in enumerate(g.data.classes):
                if i == top_buttons[0]:
                    self.classButtons[i].setText(name+" (#1)")                    
                elif i == top_buttons[1]:
                    self.classButtons[i].setText(name+" (#2)")
                elif i == top_buttons[2]:
                    self.classButtons[i].setText(name+" (#3)")
                else:
                    self.classButtons[i].setText(name)
            
            
        self.highlight_class_bar(window_index)
    
    def highlight_class_bar(self, bar_index):
        colors = Controller.highlight_class_bar(self, bar_index)
        
        self.class_graph.color_class_bars(colors)
    
    def new_frame(self, frame):
        self.update_frame_lines(frame)
        window_index = self.class_window_index(frame)
        if self.enabled and (self.current_window != window_index):
            self.current_window = window_index
            self.highlight_class_bar(window_index)
            self.selectWindow(window_index)
    
    def update_frame_lines(self,play=None):
        self.class_graph.update_frame_lines(play=play)
        
    def select_window_by_frame(self,frame=None):
        """Selects the Window around based on the current Frame shown
        
        """
        if frame is None:
            frame = self.gui.getCurrentFrame()
        window_index = self.class_window_index(frame)
        if window_index is None:
            window_index = -1
        #if the old and new index is the same do nothing.
        if self.current_window != window_index:
            self.current_window = window_index
            #self.reload()
            self.selectWindow(window_index)
            #self.scrollBar.setValue(window_index)
        else:
            self.current_window = window_index
            
            #self.scrollBar.setValue(window_index)
        
    def mergeable(self,window_index_a:int,window_index_b:int) -> bool:
        """Checks whether two windows can be merged
        
        window_index_a should be smaller than window_index_b
        """
        if (window_index_a+1 == window_index_b) and (window_index_a>=0) and (window_index_b<g.data.windows.__len__()):
            window_a = g.data.windows[window_index_a]
            window_b = g.data.windows[window_index_b]
            if window_a[2] == window_b[2]:
                a_and_b = [a == b for a,b in zip(window_a[3],window_b[3])]
                return reduce(lambda a,b: a and b, a_and_b)                
        return False
        
    def merge(self,window_index_a:int,window_index_b:int,check_mergeable = True, reload = True):
        """Tries to merge two windows"""
        if not check_mergeable or self.mergeable(window_index_a, window_index_b):
            window_b = g.data.windows[window_index_b]
            g.data.changeWindow(window_index_a, end = window_b[1], save=False)
            g.data.deleteWindow(window_index_b, save=True)
            if self.current_window == g.data.windows.__len__():
                self.current_window -= 1
            
            if reload:
                self.reload()
            
            
    def merge_all_adjacent(self):
        """Tries to merge all mergeable adjacent windows"""
        for i in range(g.data.windows.__len__()):
            while self.mergeable(i,i+1):
                self.merge(i,i+1,False,False)
        self.reload()
        
    def merge_previous(self):
        """Tries to merge the current window with the previous"""
        
        if self.current_window == 0 :
            self.add_status_message("Can't merge the first window with a previous window.")
        else:
            self.merge(self.current_window-1,self.current_window)  
            
    def merge_next(self):
        """Tries to merge the current window with the next"""
        
        if self.current_window == g.data.windows.__len__()-1 :
            self.add_status_message("Can't merge the last window with a following window.")
        else:
            self.merge(self.current_window,self.current_window+1)

    def split(self):
        """Splits the current window into two windows at a specified frame"""
        split_point = self.split_at_lineEdit.text()
        if split_point != '':
            split_point = int(self.split_at_lineEdit.text())-1
            window = g.data.windows[self.current_window]
            if window[0]+25 < split_point and split_point < window[1]-25:
                g.data.insertWindow(self.current_window, window[0], split_point, window[2], window[3], False)
                g.data.changeWindow(self.current_window+1, start=split_point, save=True)
                #self.gui.reloadClasses()
                self.reload()
            else:
                self.add_status_message("The splitting point should be inside the current window")
    
    def move_start(self):
        """Moves the start frame of the current window to a specified frame
        
        Moves the end of the previous window too.
        """
        start_new = self.move_start_lineEdit.text()
        if start_new != '':
            if self.current_window>0:
                window_previous = g.data.windows[self.current_window-1]
                window_current = g.data.windows[self.current_window]
                start_new = int(self.move_start_lineEdit.text())-1
                if window_previous[0]+50 < start_new:
                    if start_new < window_current[1]- 50 :
                        g.data.changeWindow(self.current_window-1, end=start_new, save=False)
                        g.data.changeWindow(self.current_window, start=start_new, save=True)
                        #self.gui.reloadClasses()
                        self.reload()
                    else:
                        self.add_status_message("A window can't start after it ended.")
                else:
                    self.add_status_message("A window can't start before a previous window.")
            else:
                self.add_status_message("You can't move the start point of the first window.")
        
    def move_end(self):
        """Moves the end frame of the current window to a specified frame
        
        Moves the start of the next window too.
        """
        end_new = self.move_end_lineEdit.text()
        if end_new != '':
        
            window_current = g.data.windows[self.current_window]
            end_new = int(self.move_end_lineEdit.text())
        
            if (window_current[0] + 50 < end_new):
                if self.current_window < g.data.windows.__len__()-1:
                    window_next = g.data.windows[self.current_window+1]
                    if  end_new < window_next[1]-50:
                        g.data.changeWindow(self.current_window, end = end_new, save = False)
                        g.data.changeWindow(self.current_window+1, start = end_new, save = True)
                        #self.gui.reloadClasses()
                        self.reload()
                    else:
                        self.add_status_message("A window can't end after a following window ends.")
                else:
                    if end_new <= g.data.number_samples:
                        g.data.changeWindow(self.current_window, end = end_new, save = True)
                        #self.gui.reloadClasses()
                        self.reload()
                    else:
                        self.add_status_message("A window can't end after the end of the data.")
            else:
                self.add_status_message("A window can't end before if started.")
        
    def changeClass(self):
        for i, button in enumerate(self.classButtons):
            if button.isChecked():
                g.data.changeWindow(self.current_window, class_index=i, save=True)
        #self.reload()
        self.class_graph.reload_classes(g.data.windows) 
        self.highlight_class_bar(self.current_window)
        
    def changeAttributes(self):
        """Looks which Attribute buttons are checked and saves that to the current window"""
        
        attributes = []
        for button in self.attributeButtons:
            if button.isChecked():
                attributes.append(1)
            else:
                attributes.append(0)
        g.data.changeWindow(self.current_window, attributes=attributes, save=True)
        #self.reload()
        self.class_graph.reload_classes(g.data.windows)
        self.highlight_class_bar(self.current_window)
    
    def move_buttons(self,layout:QtWidgets.QGridLayout, buttons:list):
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
        
        for i,button in enumerate(buttons):
            if button.isChecked():
                layout.addWidget(button,i+1,0)
            else:
                layout.addWidget(button,i+1,2)
            
    def get_start_frame(self) -> int:
        """returns the start of the current window"""
        if g.data.windows.__len__() >0:
            return g.data.windows[self.current_window][0]+1
        return 1
    
    def revision_mode(self, enable:bool):
        self.revision_mode_enabled = enable
        
        if not enable:
            for i,name in enumerate(g.data.classes):
                self.classButtons[i].setText(name)
        
        self.reload()

    
    
    
    
    
    
    
    
    
    
    
    
    
    