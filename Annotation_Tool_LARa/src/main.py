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



from data_management import Data_processor


from dialogs import enterIDDialog,settingsDialog, open_file_dialog

import ctypes
from controllers.manual_annotation_controller import Manual_Annotation_Controller
from controllers.label_correction_controller import Label_Correction_Controller
from controllers.automatic_annotation_controller import Automatic_Annotation_Controller


import global_variables as g
from controllers.prediction_revision_controller import Prediction_Revision_Controller
from PyQt5.QtCore import pyqtSignal
from time import sleep


pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')


class GUI(QtWidgets.QMainWindow):
    def __init__(self):
        super(GUI, self).__init__() # Call the inherited classes __init__ method
        uic.loadUi('..'+os.sep+'main.ui', self) # Load the .ui file
        self.setWindowIcon(QtGui.QIcon('..'+os.sep+'icon256.png'))
        
        self.get_widget(QtWidgets.QStatusBar, 'statusbar').showMessage("Annotation Tool Version: "+str(g.version))
        
        self.mode = 0
        self.enabled = False
        
        self.io_controller = IO_Controller(self)
        self.graphics_controller = Skeleton_Graph_Controller(self)
        self.playback_controller = Playback_Controller(self)        
        self.label_controller = Manual_Annotation_Controller(self)
        self.label_corrector = Label_Correction_Controller(self)
        self.automation_controller = Automatic_Annotation_Controller(self)
        self.revision_controller = Prediction_Revision_Controller(self)
        
        self.tabWidget = self.get_widget(QtWidgets.QTabWidget, "RightWidget")
        
        self.tabWidget.currentChanged.connect(self.changeMode)
        #print(self.tabWidget.tabBar())
        
        self.annotationGuideButton = self.get_widget(QtWidgets.QPushButton,'annotationGuideButton')
        self.annotationGuideButton.clicked.connect(lambda _: self.pause())
        self.annotationGuideButton.clicked.connect(lambda _: \
                                                   webbrowser.open(g.annotation_guide_link))
        
        self.show() # Show the GUI
        
        self.io_controller.check_directories() 
        #self.enable_widgets()
        
    def get_widget(self,qtclass:QtWidgets.QWidget,name:str) -> QtWidgets.QWidget:
        return self.findChild(qtclass,name)
    
    def update_new_frame(self,currentFrame):
        self.graphics_controller.updateSkeletonGraph(currentFrame)
        if self.mode == 0:
            self.label_controller.new_frame(currentFrame)
        elif self.mode == 1:
            self.label_corrector.new_frame(currentFrame)
        elif self.mode == 2:
            self.automation_controller.new_frame(currentFrame)
        elif self.mode == 3:
            self.revision_controller.new_frame(currentFrame)

    def updateFloorGrid(self):
        self.graphics_controller.updateFloorGrid()
        
    def getCurrentFrame(self):
        return self.playback_controller.getCurrentFrame()
    
    def add_status_message(self,msg,mode=None):
        if mode is None:
            mode = self.mode
        if mode == 0:
            self.label_controller.add_status_message(msg)
        elif mode == 1:
            self.label_corrector.add_status_message(msg)
        elif mode == 2:
            self.automation_controller.add_status_message(msg)
        elif mode == 3:
            self.revision_controller.add_status_message(msg)
            
    def enable_widgets(self):
        self.graphics_controller.enable_widgets()
        self.playback_controller.enable_widgets()
        self.label_controller.enable_widgets()
        self.label_corrector.enable_widgets()
        self.automation_controller.enable_widgets()
        self.revision_controller.enable_widgets()
        self.io_controller.enable_widgets()
        self.playback_controller.frame_changed('loadedBackup')
        
        #self.revision_mode(False)
        
        self.enabled = True
    
    def activateSaveButton(self):
        self.io_controller.activateSaveButton()
        
    def getStartFrame(self):
        if self.mode == 0:
            return self.label_controller.get_start_frame()
        elif self.mode == 1:
            return self.label_corrector.get_start_frame()
        elif self.mode == 2:
            return self.automation_controller.get_start_frame()
        elif self.mode == 3: 
            return self.revision_controller.get_start_frame()
        else:#Temporary
            return None
        
    def changeMode(self, mode:int):        
        self.mode = mode
        if self.enabled:
            if mode == 0:
                self.label_controller.reload()
            elif mode == 1:
                self.label_corrector.reload()
            elif mode == 2:
                self.automation_controller.reload()
            elif mode == 3:
                self.revision_controller.reload()
        self.io_controller.reload(mode)
        
        
    def revision_mode(self,enable:bool):
        self.label_controller.revision_mode(enable)
        self.label_corrector.revision_mode(enable)
        self.automation_controller.revision_mode(enable)
        self.revision_controller.revision_mode(enable)
        
    def pause(self):
        self.playback_controller.pause()
        
class Playback_Controller():
    def __init__(self,gui):
        self.gui = gui
        self.speed = 2
        self.paused = True
        self.enabled = False
        self.currentFrame = 1
        
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
        
        #self.timer = QtCore.QTimer(gui, timeout=self.on_timeout)
        #self.timer.setInterval(g.settings['normalSpeed'])
        self.timer = Timer_Thread()
        self.timer.on_timeout.connect(self.on_timeout)
        self.timer.set_interval(g.settings['normalSpeed'])
        #self.timer.start()
        
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
        
        self.currentFrameLineEdit.setValidator(QtGui.QIntValidator(0,g.data.number_samples))
        self.frameSlider.setRange(1,g.data.number_samples)
        #self.frame_changed('loadedBackup')
            
            
    def set_max_frame(self,max_frame):
        self.maxFramesLabel.setText("out of " + str(max_frame))
        self.frameSlider.setRange(1, max_frame)
    
    def set_speed(self, speed):
        #print("new playback speed: "+ str(speed))
        if self.speed == -3:
            self.reverseFastButton.setEnabled(True)
        elif self.speed == -2:
            self.reverseButton.setEnabled(True)
        elif self.speed == -1:
            self.reverseSlowButton.setEnabled(True)
        elif self.speed  == 1:
            self.forwardSlowButton.setEnabled(True)
        elif self.speed  == 2:
            self.forwardButton.setEnabled(True)
        else:
            self.forwardFastButton.setEnabled(True)
        
        self.speed = speed
        
        if self.speed == -3:
            self.reverseFastButton.setEnabled(False)
            #self.timer.setInterval(g.settings['fastSpeed'])
            self.timer.set_interval(g.settings['fastSpeed'])
            
        elif self.speed == -2:
            self.reverseButton.setEnabled(False)
            #self.timer.setInterval(g.settings['normalSpeed'])
            self.timer.set_interval(g.settings['normalSpeed'])
        elif self.speed == -1:
            self.reverseSlowButton.setEnabled(False)
            #self.timer.setInterval(g.settings['slowSpeed'])
            self.timer.set_interval(g.settings['slowSpeed'])
        elif self.speed == 1:
            self.forwardSlowButton.setEnabled(False)
            #self.timer.setInterval(g.settings['slowSpeed'])
            self.timer.set_interval(g.settings['slowSpeed'])
        elif self.speed == 2:
            self.forwardButton.setEnabled(False)
            #self.timer.setInterval(g.settings['normalSpeed'])
            self.timer.set_interval(g.settings['normalSpeed'])
        else:
            self.forwardFastButton.setEnabled(False)
            #self.timer.setInterval(g.settings['fastSpeed'])
            self.timer.set_interval(g.settings['fastSpeed'])
            
    def toggle_paused(self):
        self.paused = not self.paused
        if self.paused:
            self.pause()
            
        else:
            self.play()
            
    
    def pause(self):
        self.paused = True
        self.playButton.setText("Play")
        #self.timer.stop()
        self.timer.stop_timer()
        
    def play(self):
        self.paused = False
        self.playButton.setText("Pause")
        #self.timer.start()
        self.timer.start_timer()
        
    def frame_changed(self,source):
        if source == 'timer':
            pass # currentframe updated in on_timeout()
            self.frameSlider.setValue(self.currentFrame)
        elif source == 'frameSlider':
            self.currentFrame = self.frameSlider.value()
        elif source == 'loadedBackup':
            if g.data.windows.__len__()>0:
                
                self.currentFrame = g.data.windows[-1][1]#end value of the last window
            else:
                self.currentFrame = 1
            self.frameSlider.setValue(self.currentFrame)
        else:#'currentFrameLineEdit'
            self.currentFrame = int(self.currentFrameLineEdit.text())
            if self.currentFrame > g.data.number_samples:
                self.currentFrame = g.data.number_samples
            else:
                self.currentFrame = max((self.currentFrame,1))
            self.frameSlider.setValue(self.currentFrame)
        
        self.currentFrameLabel.setText(
            f"Current Frame: {self.currentFrame}/{g.data.number_samples}")
        self.gui.update_new_frame(self.currentFrame-1)
        
    def on_timeout(self):
        if self.speed < 0: 
            if self.currentFrame == 1:
                self.pause()
            else:
                self.currentFrame -=1
                self.frame_changed('timer')
        else:
            if self.currentFrame == g.data.number_samples:
                self.pause()
            else:
                self.currentFrame += 1
                self.frame_changed('timer')
                
    def getCurrentFrame(self):
        return self.currentFrame-1
    
    def setStartFrame(self):
        start = str(self.gui.getStartFrame())
        self.currentFrameLineEdit.setText(start)
        self.frame_changed('currentFrameLineEdit')

class Timer_Thread(QtCore.QThread):
    on_timeout = pyqtSignal()
    
    def __init__(self):
        super(QtCore.QThread, self).__init__()
        self.timer = 0

    def start_timer(self):
        self.timer = self.startTimer(self.interval)
        #print("timer started, id:",self.timer)
        
    def stop_timer(self):
        self.killTimer(self.timer)
        self.timer = 0
        #print("timer stopped")
        
    def set_interval(self,interval):
        self.interval = interval
        if self.timer != 0: #If already running: stop and start with new value.
            self.stop_timer()
            self.start_timer()
        
    def timerEvent(self,_): #Timer gets discarded. There is only 1 active timer anyway.
        self.on_timeout.emit()

    
class Skeleton_Graph_Controller():
    def __init__(self,gui):
        
        self.gui = gui
        self.enabled = False
        
        self.graph = gui.get_widget(gl.GLViewWidget,'skeletonGraph' )
        self.current_skeleton = None
        self.old_attr_index = -1

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
        
        if self.enabled is False:
            self.zgrid = None
            
        
    def enable_widgets(self):
        self.init_graphs()
        
        if self.enabled is False:
            self.graph.setEnabled(True)
            self.updateSkeletonGraph(0)
            self.enabled = True
            
        self.updateFloorGrid()
        
        
    def updateFloorGrid(self):
        if self.enabled:
            if (self.zgrid is None) and (g.settings['floorGrid']):
                self.zgrid = gl.GLGridItem()
                self.graph.addItem(self.zgrid)
                self.zgrid.translate(0, 0, -1)
            elif (self.zgrid is not None) and (not g.settings['floorGrid']):
                self.graph.removeItem(self.zgrid)
                self.zgrid = None
        
    def updateSkeletonGraph(self,new_frame):
        new_skeleton = g.data.frames[new_frame]
        self.current_skeleton.setData(pos=new_skeleton ,color=np.array(g.data.skeleton_colors),width=4 ,mode= 'lines')
        #self.skeleton_root.setData(pos=new_skeleton[-2])#Last two coordinates define the Root->Lowerback line
        #                                                #Therefore -2th coordinate is the root
        if g.settings['floorGrid'] and g.settings['dynamicFloor']:
            self.dynamicfloor(new_frame)
        #self.updateFrameLines(None, None, new_frame)
        
    def dynamicfloor(self,new_frame):
        new_skeleton = g.data.frames[new_frame]
        if self.zgrid is not None:
            self.graph.removeItem(self.zgrid)
        self.zgrid = gl.GLGridItem()
        self.graph.addItem(self.zgrid)
        floor_height = 0
        for segment in [g.data.body_segments_reversed[i] for i in ['L toe','R toe', 'L foot', 'R foot']]:
            segment_height = new_skeleton[segment*2,2]
            floor_height = min((floor_height,segment_height))        
        self.zgrid.translate(0, 0, floor_height)
    
        
class IO_Controller():
    def __init__(self,gui):
        self.gui = gui
        self.mode = 0
        self.enabled = False
        
        self.openFileButton = self.gui.get_widget(QtWidgets.QPushButton, 'openFileButton')
        self.openFileButton.clicked.connect(lambda _: self.openFile()) 

        self.currentFileLabel = self.gui.get_widget(QtWidgets.QLabel, 'currentFileLabel')
        
        self.saveWorkButton = self.gui.get_widget(QtWidgets.QPushButton,'saveWorkButton')
        self.saveWorkButton.clicked.connect(lambda _: self.saveFinishedProgress())
        
        self.settingsButton = self.gui.get_widget(QtWidgets.QPushButton, 'settingsButton')
        self.settingsButton.clicked.connect(lambda _: self.changeSettings())
        
        self.loadSettings()
        
        backup_folder_exists = os.path.exists('..'+os.sep+'backups')
        if not backup_folder_exists:
            os.mkdir('..'+os.sep+'backups')
        
        #self.check_directories()
    
    def enable_widgets(self):
        if not self.enabled:
            self.openFileButton.clicked.connect(lambda _: self.gui.pause())
            self.saveWorkButton.clicked.connect(lambda _: self.gui.pause())
            self.settingsButton.clicked.connect(lambda _: self.gui.pause())
            self.enabled = True
        
        
    def check_directories(self):
        """Checks if directories for new and labeled data exist 
        
        opens settings dialogue if those directories don't exist.
        
        """
        
        new_folder_exists = os.path.exists(g.settings['openFilePath'])
        labeled_folder_exists = os.path.exists(g.settings['saveFinishedPath'])
        
        if not (new_folder_exists and labeled_folder_exists):
            message = "If you are seeing this Dialog, it means you are either starting this Tool for the first Time\n"
            message += "Or a setting related to data locations is wrong/missing.\n\n"
            message += "Please setup the File Settings and Annotator ID in the following Window."
            
            QtWidgets.QMessageBox.warning(
            self.gui, 'Welcome!', message, 
            QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Ok)
            
            self.changeSettings()
            
            #Checking again for Directories because the User might have kept the standard options.
            if not os.path.exists(g.settings['openFilePath']):
                os.mkdir(g.settings['openFilePath'])
            if not os.path.exists(g.settings['saveFinishedPath']):
                os.mkdir(g.settings['saveFinishedPath'])
                
                
    def reload(self,mode):
        pass
        
    def openFile(self):
        dlg = open_file_dialog(self.gui)
        if dlg.exec_():
            file_path, annotated, load_backup = dlg.result
            file_name = os.path.split(file_path)[1]
            
            if g.data is not None:
                self.gui.revision_mode(False)
                g.data.close()
            
            #TODO: add a try catch block here
            g.data = Data_processor(file_path, load_backup, annotated)
            self.currentFileLabel.setText(f"Current File: {file_name}")
            self.gui.enable_widgets()
            
    def activateSaveButton(self):
        self.saveWorkButton.setEnabled(True)
        
    def saveFinishedProgress(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory(self.gui, 'Select labeled data directory', g.settings['saveFinishedPath']) 
        if directory != '':
            
            dlg = enterIDDialog(self.gui,self,g.settings['annotatorID'])
            result = dlg.exec_()
            if result:
                g.data.saveResults(directory,self.annotatorID,self.tries)
                self.gui.add_status_message("Saved everything!")

    def saveID(self,annotatorID,tries):
        if g.settings['annotatorID'] is not annotatorID:
            self.saveSetting('annotatorID', annotatorID)   
        self.annotatorID = annotatorID
        self.tries = tries
    
    def changeSettings(self):
        dlg = settingsDialog(self.gui)
        result = dlg.exec_()
            
        if result:
            self.gui.add_status_message("Saved new settings")
            self.gui.updateFloorGrid()
        
    def saveSetting(self,setting,value):
        g.settings[setting] = value
        with open('settings.json', 'w') as f:
            json.dump(g.settings, f)
        
    def loadSettings(self):
        settings_exist = os.path.exists('..'+os.sep+'settings.json')
        if settings_exist:
            #print("settings exist. loading them")
            with open('..'+os.sep+'settings.json', 'r') as f: 
                settings_temp = json.load(f)
            
            #Remove unused settings from older versions
            to_remove = []
            for setting in settings_temp:
                if setting not in g.settings:
                    to_remove.append(setting)
            
            for setting in to_remove:
                del settings_temp[setting]
            
            #Add new settings from newer versions
            to_add = []
            for setting in g.settings:
                if setting not in settings_temp:
                    to_add.append(setting)
            
            for setting in to_add:
                self.saveSetting(setting, g.settings[setting])
                settings_temp[setting] = g.settings[setting]
            
            
            g.settings = settings_temp
                
        else:
            #print("settings dont exist. saving them")
            with open('..'+os.sep+'settings.json', 'w') as f:
                json.dump(g.settings, f)


    
def except_hook(cls, exception, traceback):
    sys.__excepthook__(cls, exception, traceback)

if __name__ == "__main__":
    sys.excepthook = except_hook        
    
    #Needed on windows for icon in taskbar. Source:
    #https://stackoverflow.com/questions/1551605/how-to-set-applications-taskbar-icon-in-windows-7/1552105#1552105
    myappid = u'Annotation_Tool_V'+str(g.version) # arbitrary string
    if os.name == 'nt':
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    
    app = QtWidgets.QApplication(sys.argv) 
    app.setWindowIcon(QtGui.QIcon('..'+os.sep+'icon256.png'))
    window = GUI() 
    app.exec_() 

    
