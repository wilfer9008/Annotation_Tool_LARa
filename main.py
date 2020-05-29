'''
Created on 23.11.2019

@author: Erik Altermann
@email: Erik.Altermann@tu-dortmund.de
'''
import os
import json
import webbrowser
import sys

import numpy as np

from PyQt5 import QtWidgets, uic,QtCore,QtGui
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.functions import mkPen



from data_management import Data_processor, Labeled_sliding_window_dataset
from dialogs import saveClassesDialog,saveAttributesDialog,\
    changeLabelsDialog,enterIDDialog,settingsDialog, Progress_Dialog

import ctypes
from _functools import reduce
import torch
from network import Network
from PyQt5.QtCore import QThread, pyqtSignal
from builtins import KeyError
from metrics import Metrics




pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')


settings = {'openFilePath' : '..'+os.sep+'03 unlabeled sequences',
            'saveFinishedPath':'..'+os.sep+'04 labeled sequences',
            'floorGrid' : False,
            'dynamicFloor':False,
            'annotatorID':'',
            'fastSpeed':20,
            'normalSpeed':50,
            'slowSpeed':100,
            'segmentationWindowStride':200}

data = None
version = 180

class GUI(QtWidgets.QMainWindow):
    def __init__(self):
        super(GUI, self).__init__() # Call the inherited classes __init__ method
        uic.loadUi('main.ui', self) # Load the .ui file
        self.setWindowIcon(QtGui.QIcon('icon256.png'))
        
        self.get_widget(QtWidgets.QStatusBar, 'statusbar').showMessage("Annotation Tool Version: "+str(version))
        
        self.enabled = False
        
        self.io_controller = IO_Controller(self)
        self.graphics_controller = Graphics_Controller(self)
        self.playback_controller = Playback_Controller(self)        
        self.label_controller = Label_Controller(self)
        self.label_corrector = Label_Corrector(self)
        self.automation_controller = Automation_Controller(self)
        
        self.tabWidget = self.get_widget(QtWidgets.QTabWidget, "RightWidget")
        self.mode = 0
        self.tabWidget.currentChanged.connect(self.changeMode)
        
        self.statusWindows = []
        self.statusWindows.append(self.get_widget(QtWidgets.QTextEdit, 'statusWindow'))
        self.statusWindows.append(self.get_widget(QtWidgets.QTextEdit, 'lc_statusWindow'))
        self.statusWindows.append(self.get_widget(QtWidgets.QTextEdit, 'aa_statusWindow'))
        
        self.addHelpMessage("Please read the Annotation Guidelines before beginning.")
        self.addHelpMessage("If you already did start by opening a new unlabeled file or loading your progress.")
        
        self.annotationGuideButton = self.get_widget(QtWidgets.QPushButton,'annotationGuideButton')
        self.annotationGuideButton.clicked.connect(lambda _: webbrowser.open("https://docs.google.com/document/d/1RNahPI2sCZdx1Iy0Gfp-ALjFgd_e-AKnU78_DubN7iU/edit"))
        
        self.show() # Show the GUI 
        #self.enable_widgets()
        
    def get_widget(self,qtclass:QtWidgets.QWidget,name:str) -> QtWidgets.QWidget:
        return self.findChild(qtclass,name)
    
    def updateNewFrame(self,currentFrame):
        self.graphics_controller.updateSkeletonGraph(currentFrame-1)
        if self.mode is 0:
            self.graphics_controller.highlight_classBar()
        elif self.mode is 1:
            self.label_corrector.select_window_by_frame(currentFrame)
        elif self.mode is 2:
            self.automation_controller.select_window_by_frame(currentFrame)
            
        #if settings['autoUpdateLabelEnd'] is True:
        #    self.label_controller.updateEndLineEdit(currentFrame)
    
    def updateFrameLines(self,currentFrame):
        self.graphics_controller.updateFrameLines(None, currentFrame-1)
    
    def updateFloorGrid(self):
        self.graphics_controller.updateFloorGrid()
        
    def getCurrentFrame(self):
        return self.playback_controller.getCurrentFrame()
    
    def addHelpMessage(self,message):#TODO: maybe add a parameter to choose which statuswindow to adress
        for statusWindow in self.statusWindows:
            statusWindow.append("- "+message)
            
    def enable_widgets(self):
        
        self.graphics_controller.enable_widgets()
        self.playback_controller.enable_widgets()
        self.updateFrameLines(self.getCurrentFrame())
        self.label_controller.enable_widgets()
        self.label_corrector.enable_widgets()
        self.automation_controller.enable_widgets()
            
    def saveWindow(self,start,end,class_index,attributes):
        self.graphics_controller.addClass(start, end, class_index, attributes)
    
    def reloadClasses(self):
        self.graphics_controller.reloadClassGraph()
    
    def activateSaveButton(self):
        self.io_controller.activateSaveButton()
        
    def getStartFrame(self):
        if self.mode == 0:
            return self.label_controller.getStartFrame()
        elif self.mode == 1:
            return self.label_corrector.getStartFrame()
        else:#Temporary
            return self.label_controller.getStartFrame()
        
    def changeMode(self, mode:int):
        #print("old mode: "+str(self.mode))
        #print("mode: "+str(mode))
        #if self.mode == 1 and data is not None:
            #print("savewindows()")
            #data.saveWindows()
        """    
        if self.mode == 2 and mode != 2 and data.windows_2 is not None:    
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Warning)
            msg.setText("When switching from Automatic Annotation to Label Correction, the top 3 prediction information and the attribute results get discarded.")
            msg.setInformativeText("Make sure you finished working in Automated Annotation mode first.")
            msg.setWindowTitle("Warning")
            #msg.setDetailedText("The details are as follows:")
            msg.setStandardButtons(QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel)
            
            def msgbtn(i):
                #print( "Button pressed is: ",i.text())
                if i.text() == "OK":
                    global data
                    data.windows_2 = None
                    data.windows_3 = None
                    data.attr_windows = None
                    self.mode = mode
                else:
                    self.tabWidget.setCurrentIndex(2)
                    
            msg.buttonClicked.connect(msgbtn)
            _ = msg.exec_()
        else:
            self.mode = mode
        """      
        self.mode = mode
        if mode == 0:
            self.label_controller.reload()
        elif mode == 1:
            self.label_corrector.reload()
        elif mode == 2:
            self.automation_controller.reload()
            
        self.io_controller.reload(mode)
    
    def highlight_classBar(self,window_index):
        self.graphics_controller.highlight_classBar(window_index)
        
class Playback_Controller():
    def __init__(self,gui):
        self.gui = gui
        self.speed = 2
        self.paused = True
        self.enabled = False
        self.currentFrame = 1
        self.maxFrame = 24000
        
        self.playButton        = self.gui.get_widget(QtWidgets.QPushButton,'playPauseButton')
        self.reverseFastButton = self.gui.get_widget(QtWidgets.QPushButton,'reverseFastButton')
        self.reverseButton     = self.gui.get_widget(QtWidgets.QPushButton,'reverseButton')
        self.reverseSlowButton     = self.gui.get_widget(QtWidgets.QPushButton,'reverseSlowButton')
        self.forwardSlowButton     = self.gui.get_widget(QtWidgets.QPushButton,'forwardSlowButton')
        self.forwardButton     = self.gui.get_widget(QtWidgets.QPushButton,'forwardButton')
        self.forwardFastButton = self.gui.get_widget(QtWidgets.QPushButton,'forwardFastButton')
        
        self.frameSlider          = self.gui.get_widget(QtWidgets.QScrollBar,'frameSlider')
        self.currentFrameLineEdit = self.gui.get_widget(QtWidgets.QLineEdit,'currentFrameLineEdit')
        self.maxFramesLabel       = self.gui.get_widget(QtWidgets.QLabel,'maxFramesLabel')
        self.setStartPointButton  = self.gui.get_widget(QtWidgets.QPushButton,'setStartPointButton')
        
        #Connect all gui elements to correspoding actions.
        self.playButton.clicked.connect(lambda _: self.toggle_paused()) 
        self.reverseFastButton.clicked.connect(lambda _:self.set_speed(-3))
        self.reverseButton.clicked.connect(lambda _:self.set_speed(-2))
        self.reverseSlowButton.clicked.connect(lambda _:self.set_speed(-1))
        self.forwardSlowButton.clicked.connect(lambda _:self.set_speed(1))
        self.forwardButton.clicked.connect(lambda _:self.set_speed(2))
        self.forwardFastButton.clicked.connect(lambda _:self.set_speed(3))
        
        self.frameSlider.sliderMoved.connect(lambda _: self.frame_changed('frameSlider'))
        self.currentFrameLineEdit.returnPressed.connect(lambda: self.frame_changed('currentFrameLineEdit'))
        self.setStartPointButton.clicked.connect(lambda _: self.setStartFrame())
        
        self.timer = QtCore.QTimer(gui, timeout=self.on_timeout)
        self.timer.setInterval(settings['normalSpeed'])
        
        self.currentFrameLabel = self.gui.get_widget(QtWidgets.QLabel, 'currentFrameLabel')
        
    def enable_widgets(self):
        if self.enabled is False:
            self.playButton.setEnabled(True)
            self.reverseFastButton.setEnabled(True)
            self.reverseButton.setEnabled(True)
            self.reverseSlowButton.setEnabled(True)
            self.forwardSlowButton.setEnabled(True) 
            #self.forwardButton This button remains disabled since thats the standard speed
            self.forwardFastButton.setEnabled(True)
            self.currentFrameLineEdit.setEnabled(True)
            self.frameSlider.setEnabled(True)      
            self.setStartPointButton.setEnabled(True)      
            self.enabled = True
        
        self.currentFrameLineEdit.setValidator(QtGui.QIntValidator(0,data.number_samples))
        self.maxFrame = data.number_samples
        self.frameSlider.setRange(1,self.maxFrame)
        self.frame_changed('loadedBackup')
            
            
    def set_max_frame(self,max_frame):
        self.maxFramesLabel.setText("out of " + str(max_frame))
        self.frameSlider.setRange(1, max_frame)
    
    def set_speed(self, speed):
        #print("new playback speed: "+ str(speed))
        if self.speed is -3:
            self.reverseFastButton.setEnabled(True)
        elif self.speed is -2:
            self.reverseButton.setEnabled(True)
        elif self.speed is -1:
            self.reverseSlowButton.setEnabled(True)
        elif self.speed  is 1:
            self.forwardSlowButton.setEnabled(True)
        elif self.speed  is 2:
            self.forwardButton.setEnabled(True)
        else:
            self.forwardFastButton.setEnabled(True)
        
        self.speed = speed
        
        if self.speed is -3:
            self.reverseFastButton.setEnabled(False)
            self.timer.setInterval(settings['fastSpeed'])
        elif self.speed is -2:
            self.reverseButton.setEnabled(False)
            self.timer.setInterval(settings['normalSpeed'])
        elif self.speed is -1:
            self.reverseSlowButton.setEnabled(False)
            self.timer.setInterval(settings['slowSpeed'])
        elif self.speed is 1:
            self.forwardSlowButton.setEnabled(False)
            self.timer.setInterval(settings['slowSpeed'])
        elif self.speed is 2:
            self.forwardButton.setEnabled(False)
            self.timer.setInterval(settings['normalSpeed'])
        else:
            self.forwardFastButton.setEnabled(False)
            self.timer.setInterval(settings['fastSpeed'])
            
    def toggle_paused(self):
        self.paused = not self.paused
        if self.paused:
            self.playButton.setText("Play")
            self.timer.stop()
        else:
            self.playButton.setText("Pause")
            self.timer.start()
        #print("is paused: " + self.paused)
    
    def frame_changed(self,source):
        if source is 'timer':
            pass # currentframe updated in on_timeout()
            self.frameSlider.setValue(self.currentFrame)
        elif source is 'frameSlider':
            self.currentFrame = self.frameSlider.value()
        elif source is 'loadedBackup':
            if data.windows.__len__()>0:
                
                self.currentFrame = data.windows[-1][1]#end value of the last window
            else:
                self.currentFrame = 1
            self.frameSlider.setValue(self.currentFrame)
        else:#'currentFrameLineEdit'
            self.currentFrame = int(self.currentFrameLineEdit.text())
            if self.currentFrame > self.maxFrame:
                self.currentFrame = self.maxFrame
            else:
                self.currentFrame = max((self.currentFrame,1))
            self.frameSlider.setValue(self.currentFrame)
        
        self.currentFrameLabel.setText("Current Frame: " +str(self.currentFrame) +"/"+ str(self.maxFrame))
        self.gui.updateNewFrame(self.currentFrame)
        
    def on_timeout(self):
        if self.speed < 0: 
            if self.currentFrame == 1:
                self.toggle_paused()
            else:
                self.currentFrame -=1
                self.frame_changed('timer')
        else:
            if self.currentFrame == self.maxFrame:
            
                self.toggle_paused()
            else:
                self.currentFrame += 1
                self.frame_changed('timer')
                
    def getCurrentFrame(self):
        return self.currentFrame
    
    def setStartFrame(self):
        start = str(self.gui.getStartFrame())
        self.currentFrameLineEdit.setText(start)
        self.frame_changed('currentFrameLineEdit')

class Graphics_Controller():
    def __init__(self,gui):
        
        self.gui = gui
        self.enabled = False
        
        self.graph = gui.get_widget(gl.GLViewWidget,'skeletonGraph' )
        self.current_skeleton = None
        self.old_attr_index = -1
        #self.skeleton_root = None
        """
        #Joint graphs also include the classgraph for easy use. 
        #seperate the 2 graph types. Originaly there was only 1 classgraph, 
        #      but now there are 3 so it would be better for expandability and readability to separate them from jointgraphs. 
        #If only the actual joint graphs are needed use joint_graphs[3:]
        joint_graph_names = ['classGraph','lc_classGraph','aa_classGraph','jointGraphRX','jointGraphRY','jointGraphRZ','jointGraphTX','jointGraphTY','jointGraphTZ']
        self.graph_labels = ['Classes','Classes','Classes','RX','RY','RZ','TX','TY','TZ']
        
        self.joint_graphs = []
        
        for graph in joint_graph_names:
            self.joint_graphs.append( gui.get_widget(pg.PlotWidget, graph))
            
            
        """
            
        joint_graph_names = ['jointGraphRX','jointGraphRY','jointGraphRZ','jointGraphTX','jointGraphTY','jointGraphTZ']
        class_graph_names = ['classGraph','lc_classGraph']
        top3_graph_names = ['aa_classGraph','aa_classGraph_2','aa_classGraph_3'] #Show the Top2 and Top3 predictions in Automatic Annotation Mode
        attribute_graph_names = ['aa_attributeGraph'] #Shows attribute values in Automatic Annotation Mode
        
        self.joint_graphs = []
        self.class_graphs = []
        self.top3_graphs = []
        self.attribute_graphs = []
        
        for graph in joint_graph_names:
            self.joint_graphs.append( gui.get_widget(pg.PlotWidget, graph))
        
        for graph in class_graph_names:
            self.class_graphs.append( gui.get_widget(pg.PlotWidget, graph))
        
        for graph in top3_graph_names:
            self.top3_graphs.append( gui.get_widget(pg.PlotWidget, graph))
        
        for graph in attribute_graph_names:
            self.attribute_graphs.append( gui.get_widget(pg.PlotWidget, graph))
        

    def init_graphs(self):
        """Initializes/resets all graphs"""
        
        if self.current_skeleton is None:
            self.current_skeleton = gl.GLLinePlotItem(pos= np.array([[0,0,0],[0,0,0]]) ,color=np.array([[0,0,0,0],[0,0,0,0]]),mode= 'lines')
            self.graph.addItem(self.current_skeleton)
            
            #self.skeleton_root = gl.GLScatterPlotItem(pos=np.array([[0,0,0]]),color=np.array([[1,1,0,1]]))
            #self.graph.addItem(self.skeleton_root)
        else:
            self.current_skeleton.setData(pos= np.array([[0,0,0],[0,0,0]]) ,color=np.array([[0,0,0,0],[0,0,0,0]]),mode= 'lines')
            #self.skeleton_root.setData(pos=np.array([[0,0,0]]))
        
        self.intervall_start_lines = []
        self.intervall_end_lines = []
        self.play_lines = []
        
        graphs = []
        graphs.extend(self.joint_graphs)
        graphs.extend(self.class_graphs)
        graphs.extend(self.top3_graphs)
        graphs.extend(self.attribute_graphs)
        for i, graph in enumerate(graphs):
            #graph.disableAutoRange(axis=None)
            graph.clear()
            graph.setMouseEnabled(False,False)
            graph.plot([])
        
        graphs = []
        graphs.extend(self.joint_graphs)
        graphs.extend(self.class_graphs)
        graphs.extend(self.top3_graphs)
        for i, graph in enumerate(graphs): #TODO: Separate intervall lines from playlines only manual annotation graphs need intervalls
            self.intervall_start_lines.append(pg.InfiniteLine(0,label='start',labelOpts={'anchors' : [(1, 1.5), (0, 1.5)]}))
            self.intervall_end_lines.append(pg.InfiniteLine(0,label='end',labelOpts={'anchors' : [(0, 1.5), (1, 1.5)]}))
            self.play_lines.append(pg.InfiniteLine(0,pen=mkPen(0,255,0,127)))
            graphs[i].addItem(self.intervall_start_lines[i])#TODO:
            graphs[i].addItem(self.intervall_end_lines[i])#TODO:
            graphs[i].addItem(self.play_lines[i])#TODO:
        
        
        graph_labels = ['RX','RY','RZ','TX','TY','TZ']
        for i in range(0,3):
            self.joint_graphs[i].getAxis('left').setLabel(text=graph_labels[i], units='deg')
            self.joint_graphs[i].getAxis('left').enableAutoSIPrefix(False)
        for i in range(3,6):
            self.joint_graphs[i].getAxis('left').setLabel(text=graph_labels[i], units='mm')
            
        for graph in self.class_graphs:
            graph.getAxis('left').setLabel(text='Classes',units='')
        self.top3_graphs[0].getAxis('left').setLabel(text='1#Classes',units='')
        self.top3_graphs[1].getAxis('left').setLabel(text='2#Classes',units='')
        self.top3_graphs[2].getAxis('left').setLabel(text='3#Classes',units='')
        
        self.attr_bars = []
        for graph in self.attribute_graphs:
            graph.getAxis('left').setLabel(text='Attributes',units='')
            for i,attribute in enumerate(data.attributes):
                bar = pg.BarGraphItem(x0=[i],x1=i+1,y0=0,y1=0,name=attribute)
                self.attr_bars.append(bar)
                graph.addItem(bar)
                graph.setYRange(0,1,padding=0.1)
                label = pg.TextItem(text=attribute, color='b', anchor=(0,0), border=None, fill=None, angle=-90, rotateAxis=None)
                label.setPos(i+1,1)
                
                graph.addItem(label)
                
                
        
        if self.enabled is False:
            self.zgrid = None
        
    def enable_widgets(self):
        print("GraphicsController.enable_widgets()")
        self.init_graphs()
        
        if self.enabled is False:
            self.graph.setEnabled(True)
            
            self.updateSkeletonGraph(0)
            
            self.combobox = self.gui.get_widget(QtWidgets.QComboBox,'jointSelectionBox')
            self.combobox.setEnabled(True)
            self.combobox.addItems(data.body_segments.values())
            self.combobox.currentTextChanged.connect(self.updateJointGraphs)
            
            graphs = []
            graphs.extend(self.class_graphs)
            graphs.extend(self.top3_graphs)
            for graph in graphs:
                graph.setYRange(0,data.classes.__len__()+1,padding=0)
            
            #for graph in self.attribute_graphs:
            #    graph.setYRange(0,1,padding=0)
            
            self.enabled = True
            
        #self.combobox.setCurrentIndex(0)
        self.updateJointGraphs(self.combobox.currentText())
        
        graphs = []
        graphs.extend(self.joint_graphs)
        graphs.extend(self.class_graphs)
        graphs.extend(self.top3_graphs)
        for graph in graphs:
            print("setting x range to "+str((0,data.number_samples))+" with padding "+str(0.02))
            graph.setXRange(0,data.number_samples,padding=0.02)
            
        for graph in self.attribute_graphs:
            graph.setXRange(0,data.attributes.__len__(),padding=0.02)
        
        self.reloadClassGraph()
        self.updateFloorGrid()
        
        
    def updateFloorGrid(self):
        if self.enabled:
            if (self.zgrid is None) and (settings['floorGrid']):
                self.zgrid = gl.GLGridItem()
                self.graph.addItem(self.zgrid)
                self.zgrid.translate(0, 0, -1)
            elif (self.zgrid is not None) and (not settings['floorGrid']):
                self.graph.removeItem(self.zgrid)
                self.zgrid = None
        
    def updateSkeletonGraph(self,new_frame):
        new_skeleton = data.frames[new_frame]
        self.current_skeleton.setData(pos=new_skeleton ,color=np.array(data.skeleton_colors),width=4 ,mode= 'lines')
        #self.skeleton_root.setData(pos=new_skeleton[-2])#Last two coordinates define the Root->Lowerback line
        #                                                #Therefore -2th coordinate is the root
        if settings['floorGrid'] and settings['dynamicFloor']:
            self.dynamicfloor(new_frame)
        self.updateFrameLines(None, None, new_frame)
        
    def dynamicfloor(self,new_frame):
        new_skeleton = data.frames[new_frame]
        if self.zgrid is not None:
            self.graph.removeItem(self.zgrid)
        self.zgrid = gl.GLGridItem()
        self.graph.addItem(self.zgrid)
        floor_height = 0
        for segment in [data.body_segments_reversed[i] for i in ['L toe','R toe', 'L foot', 'R foot']]:
            segment_height = new_skeleton[segment*2,2]
            floor_height = min((floor_height,segment_height))        
        self.zgrid.translate(0, 0, floor_height)
        
        
    def updateFrameLines(self,start=None,end = None,play=None):
        if start is not None:
            for line in self.intervall_start_lines:
                line.setValue(start)
        if end is not None:
            for line in self.intervall_end_lines:
                line.setValue(end)
        if play is not None:
            for line in self.play_lines:
                line.setValue(play)
        
    def updateJointGraphs(self,joint):
        joint_index = data.body_segments_reversed[joint]
        if joint_index > -1:
            for i, graph in enumerate(self.joint_graphs):
                graph.listDataItems()[0].setData(data.mocap_data[:,joint_index*6+i])
        else:
            for graph in self.joint_graphs:
                graph.listDataItems()[0].setData([],[])
                
    def addClass(self, start,end,class_index,attributes,place=None):
        if place == None:
            graphs = self.class_graphs
            classbars = self.classbars
            offset = 0
        elif place==1:
            graphs = self.top3_graphs[0:1]
            classbars = self.classbars_top3
            offset = 0
        elif place==2:
            graphs = self.top3_graphs[1:2]
            classbars = self.classbars_top3
            offset = 1
        elif place==3:
            graphs = self.top3_graphs[2:3]
            classbars = self.classbars_top3
            offset = 2
        #class_graph = self.joint_graphs[0]
        for i, class_graph in enumerate(graphs):
            bar = pg.BarGraphItem(x0=[start],x1=end,y0=0,y1=class_index+1)
            classbars[i+offset].append(bar)
            if attributes[-1] == 1:
                bar.setOpts(brush=pg.mkBrush(200,100,100))
        
            class_graph.addItem(bar)
        self.updateFrameLines(end,end,self.gui.getCurrentFrame())
        
        
    
    def reloadClassGraph(self):
        #class_graph = self.joint_graphs[0]
        self.classbars = []
        graphs = []
        graphs.extend(self.class_graphs)
        for i,graph in enumerate(graphs):
            graph.clear()
            
            j = i + 6
            
            self.intervall_start_lines[j] = pg.InfiniteLine(0,label='start',labelOpts={'anchors' : [(1, 1.5), (0, 1.5)]})
            self.intervall_end_lines[j] = pg.InfiniteLine(0,label='end',labelOpts={'anchors' : [(0, 1.5), (1, 1.5)]})
            self.play_lines[j] = pg.InfiniteLine(0,pen=mkPen(0,255,0,127))
            graph.addItem(self.intervall_start_lines[j])
            graph.addItem(self.intervall_end_lines[j])
            graph.addItem(self.play_lines[j])
            
            self.classbars.append([])
            
        for start,end,class_index,attributes in data.windows:
            self.addClass(start,end,class_index,attributes)
            
            
            
        if data.windows_2 is not None:
            self.classbars_top3 = []
            graphs = []
            graphs.extend(self.top3_graphs)
            for i,graph in enumerate(graphs):
                graph.clear()
                
                j = i + 8
                
                #self.intervall_start_lines[j] = pg.InfiniteLine(0,label='start',labelOpts={'anchors' : [(1, 1.5), (0, 1.5)]})
                #self.intervall_end_lines[j] = pg.InfiniteLine(0,label='end',labelOpts={'anchors' : [(0, 1.5), (1, 1.5)]})
                self.play_lines[j] = pg.InfiniteLine(0,pen=mkPen(0,255,0,127))
                graph.addItem(self.intervall_start_lines[j])
                graph.addItem(self.intervall_end_lines[j])
                graph.addItem(self.play_lines[j])
            
                self.classbars_top3.append([])
            
            
            
            for start,end,class_index,attributes in data.windows_1:
                self.addClass(start,end,class_index,attributes,1)
            for start,end,class_index,attributes in data.windows_2:
                self.addClass(start,end,class_index,attributes,2)
            for start,end,class_index,attributes in data.windows_3:
                self.addClass(start,end,class_index,attributes,3)
        
        
    def highlight_classBar(self,bar_index=None,color=None):
        if bar_index is None:
            frame = self.gui.getCurrentFrame()
            for i, window in enumerate(data.windows):
                if window[0] <= frame and frame < window[1]:
                    bar_index = i
        
        if color is None:
            color = 'y'
        
        for bar_list in self.classbars:
            ##if bar_index >= bar_list.__len__():
            #    continue
            
            for i,bar in enumerate(bar_list):
                #print(str(i)+"/"+str(bar_list.__len__()))
                
                if data.windows[i][3][-1] == 0:
                    bar.setOpts(brush=pg.mkBrush(0.5))
                else:
                    bar.setOpts(brush=pg.mkBrush(200,100,100))
                
            if (bar_index is not None) and (data.windows[bar_index][3][-1] == 0):
                bar_list[bar_index].setOpts(brush=pg.mkBrush(color))
            elif (bar_index is not None):
                bar_list[bar_index].setOpts(brush=pg.mkBrush(255,200,50))
    
    
                    
    def show_attributes(self,current_frame):
        if current_frame is None:
            current_frame = self.gui.getCurrentFrame()
        if data.attr_windows is not None:
            attr_index = int((current_frame-1)/settings['segmentationWindowStride'])
            
            if attr_index != self.old_attr_index :
                self.old_attr_index = attr_index
                if attr_index < data.attr_windows.__len__():
                    values = data.attr_windows[attr_index]
                    for i in range(data.attributes.__len__()):
                        self.attr_bars[i].setOpts(y1=values[i])
                else:
                    for i in range(data.attributes.__len__()):
                            self.attr_bars[i].setOpts(y1=0)
                        
    
class IO_Controller():
    def __init__(self,gui):
        self.gui = gui
        self.mode = 0
        
        self.openFileButton = self.gui.get_widget(QtWidgets.QPushButton, 'openFileButton')
        self.openFileButton.clicked.connect(lambda _: self.openFile()) 

        self.loadProgressButton = self.gui.get_widget(QtWidgets.QPushButton, 'loadProgressButton')
        self.loadProgressButton.clicked.connect(lambda _: self.loadProgress())
        
        self.currentFileLabel = self.gui.get_widget(QtWidgets.QLabel, 'currentFileLabel')
        
        self.saveWorkButton = self.gui.get_widget(QtWidgets.QPushButton,'saveWorkButton')
        self.saveWorkButton.clicked.connect(lambda _: self.saveFinishedProgress())
        
        self.settingsButton = self.gui.get_widget(QtWidgets.QPushButton, 'settingsButton')
        self.settingsButton.clicked.connect(lambda _: self.changeSettings())
        
        self.loadSettings()
        
        backup_folder_exists = os.path.exists('backups')
        if not backup_folder_exists:
            os.mkdir('backups')
        
    
    def reload(self,mode):
        self.mode = mode
        if mode==0:
            self.openFileButton.setText("Open File")
            self.loadProgressButton.setEnabled(True)
        elif mode==1:
            self.openFileButton.setText("Open Annotated")
            self.loadProgressButton.setEnabled(False)
        elif mode==2:
            self.openFileButton.setText("Open File")
            self.loadProgressButton.setEnabled(False)
        
    def openFile(self):
        global data
        #print("open")
        self.saveWorkButton.setEnabled(False)
        if self.mode == 0 or self.mode == 2:
            file,_ = QtWidgets.QFileDialog.getOpenFileName(self.gui, 'Select an unlabeled .csv file', settings['openFilePath'], 'CSV Files (*.csv)', '')
        else:
            file,_ = QtWidgets.QFileDialog.getOpenFileName(self.gui, 'Select an _norm_data.csv file', settings['saveFinishedPath'], 'CSV Files (*norm_data.csv)', '')
            
        if file is not '':
            file_name = os.path.split(file)[1]
            backup_path = 'backups'+os.sep+file_name.split('.')[0]+'_backup.txt'
            backup_exists = os.path.exists(backup_path)
            if backup_exists:
                message = "A backup was found for this file. Do you want to load the backup?\n\n"
                message += "Click \"Yes\" to load your progress.\n" 
                message += "Click \"No\" to start working from scratch and delete your previous Progress\n" 
                message += "Click \"Cancel\" to not open anything."
                load_backup = QtWidgets.QMessageBox.question(self.gui, 'Backup found', message, 
                                                           QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No | QtWidgets.QMessageBox.Cancel, 
                                                           QtWidgets.QMessageBox.Cancel)
                if load_backup == QtWidgets.QMessageBox.Yes:
                    if data is not None:
                        data.close()
                    if self.mode == 0 or self.mode == 2:
                        data = Data_processor(file,backup_path,False)
                    elif self.mode == 1:
                        data = Data_processor(file,backup_path,True)
                    self.currentFileLabel.setText("Current File: "+file_name)
                    self.gui.addHelpMessage("Successfully opened Unlabeled File: "+file_name)
                    self.gui.addHelpMessage("Successfully restored backup")
                    self.gui.enable_widgets()
                elif load_backup == QtWidgets.QMessageBox.No:
                    if data is not None:
                        data.close()
                    if self.mode == 0 or self.mode == 2:
                        data = Data_processor(file,None,False)
                    elif self.mode == 1:
                        data = Data_processor(file,None,True)
                    self.currentFileLabel.setText("Current File: "+file_name)
                    self.gui.addHelpMessage("Successfully opened Unlabeled File: "+file_name)
                    self.gui.enable_widgets()
                elif load_backup == QtWidgets.QMessageBox.Cancel:
                    pass
            else:
                if data is not None:
                    data.close()
                if self.mode == 0 or self.mode == 2:
                    data = Data_processor(file,None,False)
                elif self.mode == 1:
                    data = Data_processor(file,None,True)
                self.currentFileLabel.setText("Current File: "+file_name)
                self.gui.addHelpMessage("Successfully opened Unlabeled File: "+file_name)
                self.gui.enable_widgets()      
    
    def loadProgress(self):
        global data
        #print("load")
        self.saveWorkButton.setEnabled(False)
        
        
        backup_path,_ = QtWidgets.QFileDialog.getOpenFileName(self.gui, 'Select a backup .txt file', 'backups', 'Text (*.txt)', '')
        if backup_path is not '':
            data_directory = QtWidgets.QFileDialog.getExistingDirectory(self.gui, 'Select the directory with the original unlabeled data', settings['openFilePath'])
            if data_directory is not '':
                
                _, file_name = os.path.split(backup_path)
                file_name = file_name[:-11] #removes "_backup.txt" from the backup files name
                file_name = file_name+".csv"
                file_path = data_directory +os.sep+file_name 
                
                if data is not None:
                    data.close()
                data = Data_processor(file_path,backup_path)
                self.currentFileLabel.setText("Current File: "+file_name)
                
                
                self.gui.enable_widgets()
                
                self.gui.addHelpMessage("Successfully loaded Backup of: "+file_name)
                
    def activateSaveButton(self):
        self.saveWorkButton.setEnabled(True)
        
        
        
    def saveFinishedProgress(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory(self.gui, 'Select labeled data directory', settings['saveFinishedPath']) 
        if directory is not '':
            
            dlg = enterIDDialog(self.gui,self,settings['annotatorID'])
            result = dlg.exec_()
            if result:
                data.saveResults(directory,self.annotatorID,self.tries)
                self.gui.addHelpMessage("Saved everything!")

    def saveID(self,annotatorID,tries):
        if settings['annotatorID'] is not annotatorID:
            self.saveSetting('annotatorID', annotatorID)   
        self.annotatorID = annotatorID
        self.tries = tries
    
    def changeSettings(self):
        changed_settings = []
        dlg = settingsDialog(self.gui,settings,changed_settings)
        _ = dlg.exec_()
        
        for setting,value in changed_settings:        
            self.saveSetting(setting, value)
            
        if not changed_settings == []:
            self.gui.addHelpMessage("Saved new settings")
            self.gui.updateFloorGrid()
        
        
    def saveSetting(self,setting,value):
        settings[setting] = value
        with open('settings.json', 'w') as f:
            json.dump(settings, f)
        
    def loadSettings(self):
        global settings
        settings_exist = os.path.exists('settings.json')
        if settings_exist:
            #print("settings exist. loading them")
            with open('settings.json', 'r') as f: 
                settings_temp = json.load(f)
            
            #Remove unused settings from older versions
            to_remove = []
            for setting in settings_temp:
                if setting not in settings:
                    to_remove.append(setting)
            
            for setting in to_remove:
                del settings_temp[setting]
            
            #Add new settings from newer versions
            to_add = []
            for setting in settings:
                if setting not in settings_temp:
                    to_add.append(setting)
            
            for setting in to_add:
                self.saveSetting(setting, settings[setting])
                settings_temp[setting] = settings[setting]
            
            
            settings = settings_temp
                
                
                
        else:
            #print("settings dont exist. saving them")
            with open('settings.json', 'w') as f:
                json.dump(settings, f)

        
        

class Label_Controller():
    def __init__(self,gui):
        self.gui = gui
        self.enabled = False
        
        self.startLineEdit = self.gui.get_widget(QtWidgets.QLineEdit,'startLineEdit')
        self.endLineEdit = self.gui.get_widget(QtWidgets.QLineEdit,'endLineEdit')
        
        self.setCurrentFrameButton = self.gui.get_widget(QtWidgets.QPushButton,'setCurrentFrameButton')
        self.setCurrentFrameButton.clicked.connect(lambda _: self.updateEndLineEdit(None))
        self.endLineEdit.returnPressed.connect(lambda : self.updateEndLineEdit(self.endLineEdit.text()))
        
        self.saveLabelsButton = self.gui.get_widget(QtWidgets.QPushButton,'saveLabelsButton')
        self.changeLabelsButton = self.gui.get_widget(QtWidgets.QPushButton,'changeLabelsButton')
        self.verifyLabelsButton = self.gui.get_widget(QtWidgets.QPushButton,'verifyLabelsButton')
        
        self.saveLabelsButton.clicked.connect(lambda _: self.saveLabel())
        self.changeLabelsButton.clicked.connect(lambda _: self.changeLabels())
        self.verifyLabelsButton.clicked.connect(lambda _: self.verifyLabels())
        
        
    def enable_widgets(self):
        if self.enabled is False:
            #self.startLineEdit.setEnabled(True) #Stays disabled since start is always next unlabeled frame.
            self.endLineEdit.setEnabled(True)
            self.setCurrentFrameButton.setEnabled(True)
            self.saveLabelsButton.setEnabled(True)
            self.changeLabelsButton.setEnabled(True)
            self.verifyLabelsButton.setEnabled(True)
            self.enabled = True
        
        if data.windows.__len__()>0:
            _, end, _, _ = data.windows[-1]
            end += 1
        else:
            end = 1
        self.startLineEdit.setText(str(end)) #the start of a windows is the end of the last window.
        
        self.endLineEdit.setText(str(end))
        self.endLineEdit.setValidator(QtGui.QIntValidator(0,data.number_samples))
        
        
    def saveLabel(self):
        #print("save")
        start = int(self.startLineEdit.text())
        end = int(self.endLineEdit.text())
        if start+50<end or end == data.number_samples:
            dlg = saveClassesDialog(self.gui)
            class_index = dlg.exec_()
            if class_index >-1:
                dlg = saveAttributesDialog(self.gui)
                attribute_int = dlg.exec_()
                if attribute_int>-1:
                    format_string = '{0:0'+str(data.attributes.__len__())+'b}'
                    attributes=[(x is '1')+0 for x in list(format_string.format(attribute_int))]
                    data.saveWindow(start-1, end, class_index, attributes)  #Subtracting 1 from start to index windows from 0. 
                                                                            #End is the same because indexing a:b is equivalent to [a,b[.
                    self.startLineEdit.setText(str(end+1))#End+1 because next window will save as start-1=end. 
                    self.endLineEdit.setText(str(end+1))
                    self.gui.saveWindow(start,end,class_index,attributes)   #Updates framelines and classgraph. exact indexes aren't important, 
                                                                            # with 24k samples +-1 isn't noticable to the human eye, therefore
                                                                            #not much thought has gone into the parameter values of start and end
        else:
            self.gui.addHelpMessage("Please make sure that the Labelwindow is bigger than 50 Frames. A size of at least ~100 Frames is recommended")
        
        
    def changeLabels(self):
        if data.windows.__len__()>0:
            dlg = changeLabelsDialog(self.gui,data)
            dlg.exec_()#return value gets discarded
            
            self.gui.reloadClasses()
            
        else:
            self.gui.addHelpMessage("Save at least 1 label before trying to change them.")
        
        
    def verifyLabels(self):
        self.gui.addHelpMessage("----------------\n- Verifying labels")
        verified = True
        last_window_end = 0
        failed_test = []
        for window in data.windows:
            window_start = window[0]
            window_end = window[1]
            window_class = window[2]
            window_attributes = window[3]
            
            if not last_window_end == window_start:
                failed_test.append(1)
                verified = False
            if window_start>=window_end:
                failed_test.append(2)
                verified = False
            if not (0<= window_class and window_class < data.classes.__len__()):
                failed_test.append(3)
                verified = False
            if window_attributes.__len__() is not data.attributes.__len__():
                failed_test.append(4)
                verified = False
            
            last_window_end = window_end
        if not (last_window_end == data.number_samples):
            failed_test.append(5)
            verified = False
        
        failed_test = list(set(failed_test))
        
        
        if verified:
            self.gui.addHelpMessage("Labels are verified! You can now save your labels. Please choose the folder for labeled sequences, when saving.")
            self.gui.activateSaveButton()
        else:
            if 1 in failed_test:
                self.gui.addHelpMessage("Error: There is a gap between windows.\nPlease contact someone for help.")
            if 2 in failed_test:
                self.gui.addHelpMessage("Error: One of the windows ends before it begins.\nPlease contact someone for help.")
            if 3 in failed_test:
                self.gui.addHelpMessage("Error: There is a window with an invalid class.\nPlease contact someone for help.")
            if 4 in failed_test:
                self.gui.addHelpMessage("Error: Some windows have the wrong number of Attributes.\nPlease contact someone for help.")
            if 5 in failed_test:
                self.gui.addHelpMessage("Please Finish annotating before verifying")
    
    def updateEndLineEdit(self,end=None):
        if end is None:
            current_frame = self.gui.getCurrentFrame()
            self.endLineEdit.setText(str(current_frame))
            self.gui.updateFrameLines(current_frame)
        else:
            self.endLineEdit.setText(str(end))
            self.gui.updateFrameLines(int(end))
    
    def getStartFrame(self):
        return int(self.startLineEdit.text())
    
    def reload(self):
        """reloads start line edit.
        
        called when switching to manual annotation mode
        """
        
        if self.enabled and data.windows.__len__() >0:
            start = data.windows[-1][1] + 1
            self.startLineEdit.setText(str(start))
        

class Label_Corrector():
    def __init__(self,gui):
        self.gui = gui
        self.enabled = False
        
        self.windows = []
        self.current_window = -1
        
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
        self.set_to_frame_split_button.clicked.connect(lambda _:self.split_at_lineEdit.setText(str(self.gui.getCurrentFrame())))
        self.set_to_frame_start_button  = self.gui.get_widget(QtWidgets.QPushButton,"lc_set_frame_start_button")
        self.set_to_frame_start_button.clicked.connect(lambda _:self.move_start_lineEdit.setText(str(self.gui.getCurrentFrame())))
        self.set_to_frame_end_button    = self.gui.get_widget(QtWidgets.QPushButton,"lc_set_frame_end_button")
        self.set_to_frame_end_button.clicked.connect(lambda _:self.move_end_lineEdit.setText(str(self.gui.getCurrentFrame())))
        self.set_to_start_button        = self.gui.get_widget(QtWidgets.QPushButton,"lc_set_start_button")
        self.set_to_start_button.clicked.connect(lambda _:self.move_start_lineEdit.setText(str(data.windows[self.current_window][0]+1)))
        self.set_to_end_button          = self.gui.get_widget(QtWidgets.QPushButton,"lc_set_end_button")
        self.set_to_end_button.clicked.connect(lambda _:self.move_end_lineEdit.setText(str(data.windows[self.current_window][1]+1)))
        
        self.window_by_frame_button     = self.gui.get_widget(QtWidgets.QPushButton,"lc_window__by_frame_button")
        self.window_by_frame_button.clicked.connect(lambda _:self.select_window_by_frame())
        
        #----Class buttons----
        self.classButtons = [QtWidgets.QRadioButton(text) for text in Data_processor.classes]
        layout1 = self.gui.get_widget(QtWidgets.QGroupBox,"classesGroupBox").layout()
        
        for button in self.classButtons:
            button.setEnabled(False)
            button.toggled.connect(lambda _: self.move_buttons(layout1, self.classButtons))
            button.clicked.connect(lambda _: self.changeClass())
        self.move_buttons(layout1, self.classButtons)
            
        
        #----Attribute buttons----
        self.attributeButtons = [QtWidgets.QCheckBox(text) for text in Data_processor.attributes]
        layout2 = self.gui.get_widget(QtWidgets.QGroupBox,"attributesGroupBox").layout()
        
        for button in self.attributeButtons:
            button.setEnabled(False)
            button.toggled.connect(lambda _: self.move_buttons(layout2, self.attributeButtons))
            button.clicked.connect(lambda _: self.changeAttributes())
        self.move_buttons(layout2, self.attributeButtons)
        
        
        
    def enable_widgets(self):
        """"""
        self.split_at_lineEdit.setValidator(QtGui.QIntValidator(0,data.number_samples+1))
        self.move_start_lineEdit.setValidator(QtGui.QIntValidator(0,data.number_samples+1))
        self.move_end_lineEdit.setValidator(QtGui.QIntValidator(0,data.number_samples+1))
        
        self.reload()
    
    def reload(self):
        """reloads all window information
        
        called when switching to label correction mode
        """
        
        
        if data is not None and data.windows.__len__() >0:
            self.gui.reloadClasses()
            self.setEnabled(True)
            self.scrollBar.setRange(0,data.windows.__len__()-1)
            self.scrollBar.setValue(self.current_window)
            self.selectWindow(self.current_window)
            
        else:
            self.setEnabled(False)
            
    def setEnabled(self,enable:bool):
        """Turns the Widgets of Label Correction Mode on or off based on the enable parameter
        
        Arguments:
        ----------
        enable : bool
            If True and widgets were disabled, the widgets get enabled.
            If False and widgets were enabled, the widgets get disabled.
            Otherwise does nothing.
        ----------
        
        """
        if not (self.enabled == enable):
            #Only reason why it might be disabled is that there were no windows
            #Therefore setting the current window to 0 as this mode is enabled 
            #as soon as there is at least one window
            self.current_window = 0
            self.enabled = enable
            for button in self.classButtons:
                button.setEnabled(enable)
            for button in self.attributeButtons:
                button.setEnabled(enable)
            
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
            
            self.window_by_frame_button.setEnabled(enable)
            
            self.scrollBar.setEnabled(enable)
            
    def selectWindow(self,window_index:int):
        """Selects the window at window_index"""
        if window_index >= 0:
            self.current_window = window_index
        else:
            self.current_window = data.windows.__len__() + window_index
        window = data.windows[self.current_window]
        self.current_window_label.setText("Current Window: "+str(self.current_window+1)+"/"+str(data.windows.__len__()))
        self.start_lineEdit.setText(str(window[0]+1))
        self.end_lineEdit.setText(str(window[1]+1))
        
        self.classButtons[window[2]].setChecked(True)
        for button, checked in zip(self.attributeButtons,window[3]):
            button.setChecked(checked)
        
        self.gui.highlight_classBar(window_index)
        
    def select_window_by_frame(self,frame=None):
        """Selects the Window around based on the current Frame shown
        
        """
        if frame is None:
            frame = self.gui.getCurrentFrame()
        window_index = -1
        for i, window in enumerate(data.windows):
            if window[0] <= frame and frame < window[1]:
                window_index = i
                break
        #if the old and new index is the same do nothing.
        if not self.current_window == window_index:
            self.current_window = window_index
            self.reload()
        else:
            self.current_window = window_index
        
    def mergeable(self,window_index_a:int,window_index_b:int) -> bool:
        """Checks whether two windows can be merged
        
        window_index_a should be smaller than window_index_b
        """
        if (window_index_a+1 == window_index_b) and (window_index_a>=0) and (window_index_b<data.windows.__len__()):
            window_a = data.windows[window_index_a]
            window_b = data.windows[window_index_b]
            if window_a[2] == window_b[2]:
                a_and_b = [a == b for a,b in zip(window_a[3],window_b[3])]
                return reduce(lambda a,b: a and b, a_and_b)                
        return False
        
    def merge(self,window_index_a:int,window_index_b:int,check_mergeable = True, reload = True):
        """Tries to merge two windows"""
        if not check_mergeable or self.mergeable(window_index_a, window_index_b):
            window_b = data.windows[window_index_b]
            data.changeWindow(window_index_a, end = window_b[1], save=False)
            data.deleteWindow(window_index_b, save=True)
            if self.current_window == data.windows.__len__():
                self.current_window -= 1
            
            if reload:
                self.reload()
            
            
    def merge_all_adjacent(self):
        """Tries to merge all mergeable adjacent windows"""
        for i in range(data.windows.__len__()):
            while self.mergeable(i,i+1):
                self.merge(i,i+1,False,False)
        self.reload()
        
    def merge_previous(self):
        """Tries to merge the current window with the previous"""
        
        if self.current_window == 0 :
            self.gui.addHelpMessage("Can't merge the first window with a previous window.")
        else:
            self.merge(self.current_window-1,self.current_window)  
            
    def merge_next(self):
        """Tries to merge the current window with the next"""
        
        if self.current_window == data.windows.__len__()-1 :
            self.gui.addHelpMessage("Can't merge the last window with a following window.")
        else:
            self.merge(self.current_window,self.current_window+1)

    def split(self):
        """Splits the current window into two windows at a specified frame"""
        split_point = self.split_at_lineEdit.text()
        if split_point != '':
            split_point = int(self.split_at_lineEdit.text())-1
            window = data.windows[self.current_window]
            if window[0]+25 < split_point and split_point < window[1]-25:
                data.insertWindow(self.current_window, window[0], split_point, window[2], window[3], False)
                data.changeWindow(self.current_window+1, start=split_point, save=True)
                #self.gui.reloadClasses()
                self.reload()
            else:
                self.gui.addHelpMessage("The splitting point should be inside the current window")
    
    def move_start(self):
        """Moves the start frame of the current window to a specified frame
        
        Moves the end of the previous window too.
        """
        start_new = self.move_start_lineEdit.text()
        if start_new != '':
            if self.current_window>0:
                window_previous = data.windows[self.current_window-1]
                window_current = data.windows[self.current_window]
                start_new = int(self.move_start_lineEdit.text())-1
                if window_previous[0]+50 < start_new:
                    if start_new < window_current[1]- 50 :
                        data.changeWindow(self.current_window-1, end=start_new, save=False)
                        data.changeWindow(self.current_window, start=start_new, save=True)
                        #self.gui.reloadClasses()
                        self.reload()
                    else:
                        self.gui.addHelpMessage("A window can't start after it ended.")
                else:
                    self.gui.addHelpMessage("A window can't start before a previous window.")
            else:
                self.gui.addHelpMessage("You can't move the start point of the first window.")
        
    def move_end(self):
        """Moves the end frame of the current window to a specified frame
        
        Moves the start of the next window too.
        """
        end_new = self.move_end_lineEdit.text()
        if end_new != '':
        
            window_current = data.windows[self.current_window]
            end_new = int(self.move_end_lineEdit.text())
        
            if (window_current[0] + 50 < end_new):
                if self.current_window < data.windows.__len__()-1:
                    window_next = data.windows[self.current_window+1]
                    if  end_new < window_next[1]-50:
                        data.changeWindow(self.current_window, end = end_new, save = False)
                        data.changeWindow(self.current_window+1, start = end_new, save = True)
                        #self.gui.reloadClasses()
                        self.reload()
                    else:
                        self.gui.addHelpMessage("A window can't end after a following window ends.")
                else:
                    if end_new <= data.number_samples:
                        data.changeWindow(self.current_window, end = end_new, save = True)
                        #self.gui.reloadClasses()
                        self.reload()
                    else:
                        self.gui.addHelpMessage("A window can't end after the end of the data.")
            else:
                self.gui.addHelpMessage("A window can't end before if started.")
        
    def changeClass(self):
        for i, button in enumerate(self.classButtons):
            if button.isChecked():
                data.changeWindow(self.current_window, class_index=i, save=True)
        #self.gui.reloadClasses()
        self.reload()
        
    def changeAttributes(self):
        """Looks which Attribute buttons are checked and saves that to the current window"""
        
        attributes = []
        for button in self.attributeButtons:
            if button.isChecked():
                attributes.append(1)
            else:
                attributes.append(0)
        data.changeWindow(self.current_window, attributes=attributes, save=True)
        #self.gui.reloadClasses()
        self.reload()
    
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
            
    def getStartFrame(self) -> int:
        """returns the start of the current window"""
        
        return data.windows[self.current_window][0]+1
    
    
class Automation_Controller():
    def __init__(self,gui):
        self.gui = gui
        self.enabled = False
        
        self.window_step = settings['segmentationWindowStride']
        
        self.selected_network = 0 #TODO: Save last selected in settings
        self.current_window = -1
        
        
        #ComboBoxes
        self.network_comboBox = self.gui.get_widget(QtWidgets.QComboBox,"aa_network_comboBox")
        #self.network_comboBox.currentIndexChanged.connect(self.load_network)
        self.network_comboBox.currentIndexChanged.connect(self.select_network)
        self.network_comboBox.addItem('Class Network')
        self.network_comboBox.addItem('Attribute Network')
        
        
        self.post_processing_comboBox = self.gui.get_widget(QtWidgets.QComboBox,"aa_post_processing_comboBox")
        

        #Buttons
        self.annotate_button = self.gui.get_widget(QtWidgets.QPushButton,"aa_annotate_button")
        self.annotate_button.clicked.connect(lambda _: self.annotate())
        
        #self.annotate_folder_button = self.gui.get_widget(QtWidgets.QPushButton,"aa_annotate_folder_button")
        #self.annotate_folder_button.clicked.connect(lambda _: self.annotate())
        """
        self.choose_2nd_button = self.gui.get_widget(QtWidgets.QPushButton,"aa_choose_2nd_button")
        self.choose_2nd_button.clicked.connect(lambda _: self.choose_2nd())
        
        self.choose_3rd_button = self.gui.get_widget(QtWidgets.QPushButton,"aa_choose_3rd_button")
        self.choose_3rd_button.clicked.connect(lambda _: self.choose_3rd())
        
        self.window_by_frame_button     = self.gui.get_widget(QtWidgets.QPushButton,"aa_window__by_frame_button")
        self.window_by_frame_button.clicked.connect(lambda _:self.select_window_by_frame())
        
        
        #Scrollbars
        self.scrollBar = self.gui.get_widget(QtWidgets.QScrollBar,"aa_scrollBar")
        self.scrollBar.valueChanged.connect(self.selectWindow)
        """
        
    def enable_widgets(self):
        if not self.enabled:
            self.enabled = True
        else:
            self.choose_2nd_button.setEnabled(False)
            self.choose_3rd_button.setEnabled(False)
        
        self.reload()
        self.enable_annotate_button()
        

    def enable_annotate_button(self):
        if self.selected_network>0:
            #self.annotate_folder_button.setEnabled(True)
            if self.enabled:
                self.annotate_button.setEnabled(True)
            else:
                self.annotate_button.setEnabled(False)
        else:
            self.annotate_button.setEnabled(False)
            #self.annotate_folder_button.setEnabled(False)
            
    def reload(self):
        if data is not None and data.windows_2 is not None and data.windows_2.__len__() >0:
            self.gui.reloadClasses()
            #print("enabling")
            """
            self.window_by_frame_button.setEnabled(True)
            self.scrollBar.setEnabled(True)         
               
            self.scrollBar.setRange(0,data.windows.__len__()-1)
            self.scrollBar.setValue(self.current_window)
            """
            self.selectWindow(self.current_window)
            """
            if data.windows_2 is not None:
                self.choose_2nd_button.setEnabled(True)
                self.choose_3rd_button.setEnabled(True)
            """
        else:
            #print("disabling")
            """
            self.scrollBar.setEnabled(False)
            self.window_by_frame_button.setEnabled(False)
            
            self.choose_2nd_button.setEnabled(False)
            self.choose_3rd_button.setEnabled(False)
            """
        
                
    def select_network(self,index):
        """Saves the selected network and tries to activate annotation if one was selected"""
        self.selected_network = index
        self.enable_annotate_button()
    
    def annotate(self):
        self.progress = Progress_Dialog(self.gui, "annotating", 7)
        
        self.annotator = Annotator(self.gui,self.selected_network)
        self.annotator.progress.connect(self.progress.setStep)
        self.annotator.nextstep.connect(self.progress.newStep)
        self.annotator.cancel.connect(lambda _: self.cancel_annotation())
        self.annotator.done.connect(lambda _: self.finish_annotation())
        
        data.attr_windows = None
        for i in range(data.attributes.__len__()):
            self.gui.graphics_controller.attr_bars[i].setOpts(y1=0)
        
        self.progress.show()
        self.annotator.start()
        
        
        
    def finish_annotation(self):
        #del self.annotator
        #self.progress.close()
        self.gui.reloadClasses()
        self.reload()
        #print("done")
    
    def cancel_annotation(self):
        self.progress.close()
    
    
    
    def select_window_by_frame(self,frame=None):
        """Selects the Window around based on the current Frame shown
        
        """
        if frame is None:
            frame = self.gui.getCurrentFrame()
        window_index = -1
        for i, window in enumerate(data.windows):
            if window[0] <= frame and frame < window[1]:
                window_index = i
                break
        
        #if the old and new index is the same do nothing.
        if not self.current_window == window_index:
            self.current_window = window_index
            self.reload()
        else:
            self.current_window = window_index
        self.gui.graphics_controller.show_attributes(frame)
        
    def selectWindow(self,window_index:int):
        """Selects the window at window_index"""
        if window_index >= 0:
            self.current_window = window_index
        else:
            self.current_window = data.windows.__len__() + window_index
        
        #self.gui.highlight_classBar(window_index)
        self.gui.graphics_controller.show_attributes(None)
    """
    def choose_2nd(self):
        start,end,class_label,attributes = data.windows_2[self.current_window]
        data.changeWindow(self.current_window, start, end, class_label, attributes, True)
        self.gui.reloadClasses()
        self.gui.highlight_classBar(self.current_window)
        
    def choose_3rd(self):
        start,end,class_label,attributes = data.windows_3[self.current_window]
        data.changeWindow(self.current_window, start, end, class_label, attributes, True)
        self.gui.reloadClasses()
        self.gui.highlight_classBar(self.current_window)
    """
        
class Annotator(QThread):
    progress = pyqtSignal(int)
    nextstep = pyqtSignal(str,int)
    done = pyqtSignal(int)
    cancel = pyqtSignal(int)
    
    def __init__(self,gui,selected_network):
        super(Annotator, self).__init__()
        self.gui = gui
        self.selected_network = selected_network
        self.window_step = settings['segmentationWindowStride']
        self.device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")
        
    def load_network(self,index):
        """Loads the selected network"""
        if index == 1:
            print('loading class_network')
            checkpoint = torch.load('networks' + os.sep + 'class_network.pt', map_location=self.device)
            self.network=None
        elif index == 2:
            print('loading attrib_network')
            checkpoint = torch.load('networks' + os.sep + 'attrib_network.pt', map_location=self.device)
            self.network=None
        else:
            self.network=None
            self.cancel.emit(0)
            raise Exception
            
        
        state_dict = checkpoint['state_dict']
        config = checkpoint['network_config']
        if 'att_rep' in checkpoint.keys():
            att_rep = checkpoint['att_rep']
        else:
            att_rep = None
        network = Network(config)
        network.load_state_dict(state_dict)
        network.eval()
        return network,config,att_rep
    
    
    def run(self):
        global data
        self.nextstep.emit("loading network", 1)
        
        #Load network
        #print("loading network")
        network,config,att_rep = self.load_network(self.selected_network)
        #print("network loaded")
        
        self.nextstep.emit("segmenting", 1)
        
        #Segment Data
        #print("segmenting data")
        window_length = config['sliding_window_length']
        dataset = Labeled_sliding_window_dataset(data.mocap_data, window_length, self.window_step)
        dataset_len = dataset.__len__()
        #print("data segmented")
        
        
        #Forward through network
        self.nextstep.emit("annotating", dataset_len)
        
        label_kind = config['labeltype']
        if att_rep is not None:
            metrics = Metrics(config, self.device, att_rep)
        
        for i in range(dataset_len):
            label = network(dataset.__getitem__(i))
            #label = torch.argmax(label).item()
            #dataset.setlabel(i, label, label_kind)
            
            if label_kind == 'class':
                label = torch.argsort(label, descending=True)[0:3]
                
                dataset.set_top3_labels(i, label, label_kind)
                
            elif label_kind == 'attributes':
                
                dataset.set_top3_labels(i, label, label_kind,metrics)
                
            else:
                raise Exception
            
            
            
            #print(str(i+1)+"/"+str(dataset_len)+"\tRange:"+str(self.dataset.__range__(i)))
            self.progress.emit(i)
        
        #Evaluate results
        self.nextstep.emit("evaluating", 1)
        
        #windows = dataset.evaluate()
        windows_1,windows_2,windows_3 = dataset.evaluate_top3()
              
        
        #Save windows
        self.nextstep.emit("saving", 1)
        #print("Saving Windows")
        data.windows_1 = windows_1
        data.windows_2 = windows_2
        data.windows_3 = windows_3
        data.attr_windows = dataset.attr_labels
        #self.gui.reloadClasses()
        #print("Saved Windows")
        
        #Merge windows
        self.nextstep.emit("cleaning up", 1)
        #print("merging windows")
        #self.gui.label_corrector.merge_all_adjacent()
        #print("merged windows")
        
        if label_kind == 'class':
            self.merge_all_top3()
            
            
        elif label_kind == 'attributes':
            self.merge_all_top3()
        else:
            raise Exception
        
        
        
        
        self.nextstep.emit("done", 0)
        self.done.emit(0)
    
    def merge_all_top3(self):
        for i in range(data.windows.__len__()):
            while self.mergeable_top3(i,i+1):
                self.merge_top3(i,i+1)
        data.saveWindows()
    
    def mergeable_top3(self,window_index_a,window_index_b):
        if (window_index_a+1 == window_index_b) and (window_index_a>=0) and (window_index_b<data.windows_1.__len__()):
            #2 Windows should only be mergable if all 3 Predictions can merge these 2 windows
            window_a_1 = data.windows_1[window_index_a]
            window_b_1 = data.windows_1[window_index_b]
            mergeable1 = window_a_1[2] == window_b_1[2]
            
            window_a_2 = data.windows_2[window_index_a]
            window_b_2 = data.windows_2[window_index_b]
            mergeable2 = window_a_2[2] == window_b_2[2]
            
            window_a_3 = data.windows_3[window_index_a]
            window_b_3 = data.windows_3[window_index_b]
            mergeable3 = window_a_3[2] == window_b_3[2]
            
            return mergeable1 and mergeable2 and mergeable3
        return False
    
    def merge_top3(self,window_index_a:int,window_index_b:int):
        global data
        
        
        window_a_1 = data.windows_1[window_index_a]
        window_b_1 = data.windows_1[window_index_b]
            
        window_a_2 = data.windows_2[window_index_a]
        window_b_2 = data.windows_2[window_index_b]
            
        window_a_3 = data.windows_3[window_index_a]
        window_b_3 = data.windows_3[window_index_b]
        
        """
        data.changeWindow(window_index_a, end = window_b[1], save=False)
        
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
        """
        
        window_a_1 = list(window_a_1)
        window_a_1[1] = window_b_1[1]
        data.windows_1[window_index_a] = tuple(window_a_1)
        
        window_a_2 = list(window_a_2)
        window_a_2[1] = window_b_2[1]
        data.windows_2[window_index_a] = tuple(window_a_2)
        
        window_a_3 = list(window_a_3)
        window_a_3[1] = window_b_3[1]
        data.windows_3[window_index_a] = tuple(window_a_3)
        
        
        """
        data.deleteWindow(window_index_b, save=True)
        
        self.windows.pop(window_index)
        """
        
        data.windows_1.pop(window_index_b)
        data.windows_2.pop(window_index_b)
        data.windows_3.pop(window_index_b)
        
        
    
def except_hook(cls, exception, traceback):
    sys.__excepthook__(cls, exception, traceback)

if __name__ == "__main__":
    sys.excepthook = except_hook        
    
    #Needed on windows for icon in taskbar. Source:
    #https://stackoverflow.com/questions/1551605/how-to-set-applications-taskbar-icon-in-windows-7/1552105#1552105
    myappid = u'Annotation_Tool_V'+str(version) # arbitrary string
    if os.name == 'nt':
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    
    app = QtWidgets.QApplication(sys.argv) 
    app.setWindowIcon(QtGui.QIcon('icon256.png'))
    window = GUI() 
    app.exec_() 

    
