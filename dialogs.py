'''
Created on 30.11.2019

@author: Erik Altermann
@email: Erik.Altermann@tu-dortmund.de
'''

from PyQt5 import QtWidgets, uic
from PyQt5.QtGui import QIntValidator
import pyqtgraph as pg

from data_management import Data_processor
import data_management
from PyQt5.QtCore import QThread
#from main import IO_Controller

class saveClassesDialog(QtWidgets.QDialog):
    """Dialog displaying displaying an exlusive choise between all classes
    
    example code:
        dlg = saveClassesDialog(QWidget)
        result = dlg.exec_()
    the result will be an integer, see in the method pressedButtons
    
    """
    
    def __init__(self,parent:QtWidgets.QWidget,selected_class:int=None): 
        """Initializes the dialog and sets up the gui
        
        Arguments:
        ----------
        parent : QWidget
            parent widget of this dialog
        selected_class : int (optional)
            Index of an already selected class, or None if 
            no class was selected before starting this dialog.
        ----------
            
        """
        super(saveClassesDialog, self).__init__(parent)
        self.setWindowTitle("Save Class")
        
        self.classButtons = [QtWidgets.QRadioButton(text) for text in Data_processor.classes]
        
        self.layout = QtWidgets.QVBoxLayout()
        for button in self.classButtons:
            self.layout.addWidget(button)
            button.clicked.connect(lambda _: self.enableOkButton())
            
        QBtn = QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        
        self.buttonBox = QtWidgets.QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(lambda :self.done(self.pressedButton()))
        self.buttonBox.rejected.connect(lambda :self.done(-1))
        
        self.buttonBox.button(QtWidgets.QDialogButtonBox.Ok).setEnabled(False)
        
        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)
        
        if selected_class is not None:
            self.classButtons[selected_class].setChecked(True)
            self.enableOkButton()
        
    def pressedButton(self) -> int:
        """Checks which radioButton is currently checked
        
        Returns:
        --------
        class_index : int
            index of the selected class.
            -1 if no class was selected
        --------
        """
        class_index = -1
        for i,button in enumerate(self.classButtons):
            if button.isChecked():
                class_index = i
        return class_index

    def enableOkButton(self):
        """Enables the OkButton"""
        
        self.buttonBox.button(QtWidgets.QDialogButtonBox.Ok).setEnabled(True)
        
        
class saveAttributesDialog(QtWidgets.QDialog):
    """Dialog displaying displaying a non-exlusive choice between all attributes
    
    example code:
        dlg = saveAttributesDialog(QWidget)
        result = dlg.exec_()
    the result will be an integer, see in the method pressedButtons
    
    """
    def __init__(self,parent:QtWidgets.QWidget,selected_attr:list=None): 
        """Initializes the dialog and sets up the gui

        Arguments:
        ----------
        parent : QWidget
            parent widget of this dialog
        selected_attr : list (optional)
            list of booleans with one value per attribute. 
            1 if an attribute is present. 0 if its not.
            argument is None if no attributes were selected before starting this dialog.
        ----------
            
        """
        super(saveAttributesDialog, self).__init__(parent)
        self.setWindowTitle("Save Attributes")

        self.attributeButtons = [QtWidgets.QCheckBox(text) for text in Data_processor.attributes]
        
        self.layout = QtWidgets.QVBoxLayout()
        for button in self.attributeButtons:
            self.layout.addWidget(button)
        QBtn = QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        
        self.buttonBox = QtWidgets.QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(lambda :self.done(self.pressedButtons()))
        self.buttonBox.rejected.connect(lambda :self.done(-1))
        
        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)
        
        if selected_attr is not None:
            for i, attr_val in enumerate(selected_attr):
                self.attributeButtons[i].setChecked(attr_val)
        
        
    def pressedButtons(self) -> int:
        """Checks which buttons are currently checked
        
        Returns:
        --------
        attr_int : int
            the list of buttons gets transformed into a boolean list 
            by checking their isChecked() state. This boolean list then gets interpreted 
            as an integer because QDialog.done(r) asks for integers
            it can be converted back to a boolean list later when needed
        --------
        """
        pressedButtons = []
        for button in self.attributeButtons:
            pressedButtons.append(button.isChecked())
        #print(pressedButtons)
        attr_int = sum(2**i for i, v in enumerate(reversed(pressedButtons)) if v)
        return attr_int

class settingsDialog(QtWidgets.QDialog):
    """Dialog displaying displaying all settings that are changeable by the user
    
    example code:
        changed_settings = []
        dlg = settingsDialog(QWidget,settings,changed_settings)
        _ = dlg.exec_()
    results are stored in the changed_settings parameter of the __init__ method
    """
    
    def __init__(self,parent:QtWidgets.QWidget,settings:dict,changed_settings:list): 
        """Initializes the dialog and sets up the gui

        Arguments:
        ----------
        parent : QWidget
            parent widget of this dialog
        settings : dict
            all settings
        changed_settings : list
            list of tuples (key,value) for the settings dict.
            should be empty, it gets filled with changes in the accept method.
            
        ----------
        """
        
        super(settingsDialog, self).__init__(parent)
        uic.loadUi('settings.ui', self)
        self.settings = settings
        self.changed_settings = changed_settings
        #self.findChild(qtclass,name)
        
        #----- Video Settings Widgets -----
        
        self.floorEnableButton = self.findChild(QtWidgets.QRadioButton, 'floorEnableRadioButton')
        self.floorDisableButton = self.findChild(QtWidgets.QRadioButton, 'floorDisableRadioButton')
        
        self.dynamicEnableButton = self.findChild(QtWidgets.QRadioButton, 'dynamicEnableRadioButton')
        self.dynamicDisableButton = self.findChild(QtWidgets.QRadioButton, 'dynamicDisableRadioButton')
        
        self.fastLineEdit = self.findChild(QtWidgets.QLineEdit, 'fastLineEdit')
        self.fastLineEdit.setValidator(QIntValidator(1,1000))
        self.normalLineEdit = self.findChild(QtWidgets.QLineEdit, 'normalLineEdit')
        self.normalLineEdit.setValidator(QIntValidator(1,1000))
        self.slowLineEdit = self.findChild(QtWidgets.QLineEdit, 'slowLineEdit')
        self.slowLineEdit.setValidator(QIntValidator(1,1000))
        
        #----- File Settings widgets -----
        
        self.annotatorLineEdit = self.findChild(QtWidgets.QLineEdit, 'annotatorLineEdit')
        self.annotatorLineEdit.setValidator(QIntValidator(0,1000))
        
        self.unlabeledLineEdit = self.findChild(QtWidgets.QLineEdit, 'unlabeledLineEdit')
        self.labeledLineEdit = self.findChild(QtWidgets.QLineEdit, 'labeledLineEdit')
        #TODO: impement the browse buttons to change the paths
        #TODO: backup settings currently have no effect. Implement them in data_management
        self.unlabeledButton = self.findChild(QtWidgets.QPushButton, 'unlabeledButton')
        self.labeledButton = self.findChild(QtWidgets.QPushButton, 'labeledButton')
        
        #-----Get all settings-----
        
        if self.settings['floorGrid']:
            self.floorEnableButton.setChecked(True)
        else:
            self.floorDisableButton.setChecked(True)
        
        if self.settings['dynamicFloor']:
            self.dynamicEnableButton.setChecked(True)
        else:
            self.dynamicDisableButton.setChecked(True)
        
        self.fastLineEdit.setText(str(self.settings['fastSpeed']))
        self.normalLineEdit.setText(str(self.settings['normalSpeed']))
        self.slowLineEdit.setText(str(self.settings['slowSpeed']))
        
        self.annotatorLineEdit.setText(str(self.settings['annotatorID']))
        
        self.unlabeledLineEdit.setText(str(self.settings['openFilePath']))
        self.labeledLineEdit.setText(str(self.settings['saveFinishedPath']))
        
        
        #-----Connecting signals and slots-----
        self.unlabeledButton.clicked.connect(lambda _: self.browseDir('unlabeled'))
        self.labeledButton.clicked.connect(lambda _: self.browseDir('labeled'))
        
        
        
    def browseDir(self,directoryType:str):
        """Open directory dialog and saves the chosen directory in the correct lineEdit
        
        Arguments:
        ----------
        directoryType : str
            Tells which lineEdit to fill with the new directory
        """
        
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select a new Directory','')
        if directory is not '':
            if directoryType == 'unlabeled':
                self.unlabeledLineEdit.setText(directory)
            elif directoryType == 'labeled':
                self.labeledLineEdit.setText(directory)
            
    def accept(self, *args, **kwargs):
        """Computes the result and saves it to changed_settings
        
        Overwritten method from QtWidgets.QDialog
        gets called when the dialog gets closed via the ok button and closes the dialog
        
        """
        
        if self.settings['floorGrid'] == self.floorDisableButton.isChecked():
            self.changed_settings.append(('floorGrid',self.floorEnableButton.isChecked()))
            
        if self.settings['dynamicFloor'] == self.dynamicDisableButton.isChecked():
            self.changed_settings.append(('dynamicFloor',self.dynamicEnableButton.isChecked()))
        
        
        lineEdits = [self.fastLineEdit,self.normalLineEdit,self.slowLineEdit,self.annotatorLineEdit]
        lineSetting = ['fastSpeed','normalSpeed','slowSpeed','annotatorID']
        
        for lineSetting,lineEdit in zip(lineSetting,lineEdits):
            if not str(self.settings[lineSetting]) == lineEdit.text():
                self.changed_settings.append((lineSetting,int(lineEdit.text())))
        
        
        lineEdits = [self.unlabeledLineEdit,self.labeledLineEdit]
        lineSetting = ['openFilePath','saveFinishedPath']
        
        for lineSetting,lineEdit in zip(lineSetting,lineEdits):
            if not str(self.settings[lineSetting]) == lineEdit.text():
                self.changed_settings.append((lineSetting,lineEdit.text()))
        
        return QtWidgets.QDialog.accept(self, *args, **kwargs)
    
        
class changeLabelsDialog(QtWidgets.QDialog):
    """Dialog displaying all the label windows made by the Annotator, with an option to change the labels
    
    example code:
        dlg = changeLabelsDialog(QWidget,Data_processor)
        _ = dlg.exec_()
    
    anytime the user changes a label it gets saved automatically 
    """
    
    def __init__(self,parent:QtWidgets.QWidget, data:data_management.Data_processor): 
        """Initializes the dialog and sets up the gui

        Arguments:
        ----------
        parent : QWidget
            parent widget of this dialog
        data : data_management.Data_processor
        ----------
        """
        super(changeLabelsDialog, self).__init__(parent)
        uic.loadUi('selectWindow.ui', self)
        self.data = data
        #self.findChild(qtclass,name)
        
        self.graph = self.findChild(pg.PlotWidget,"classGraph")
        self.graph.setXRange(0,self.data.number_samples)
        self.graph.setYRange(0,self.data.classes.__len__()+1,padding=0)
        self.graph.getAxis('left').setLabel(text='Classes',units='')
        
        self.windows = self.data.windows
        self.classes = self.data.classes
        self.attributes = self.data.attributes
        
        self.window_bars = []
        for start,end,class_index,attributes in self.windows:
            bar = pg.BarGraphItem(x0=[start],x1=end,y0=0,y1=class_index+1)
            if attributes[-1] == 1:
                bar.setOpts(brush=pg.mkBrush(200,100,100))
            self.window_bars.append(bar)
            self.graph.addItem(bar)
        self.current_window = self.windows.__len__()-1
        self.selectWindow(self.current_window)       
        
        self.scrollbar = self.findChild(QtWidgets.QScrollBar,'windowScrollBar')
        self.scrollbar.setRange(0,self.windows.__len__()-1)
        self.scrollbar.setValue(self.windows.__len__()-1)
        self.scrollbar.valueChanged.connect(self.selectWindow)
        
        self.startLineEdit = self.findChild(QtWidgets.QLineEdit,"startLineEdit")
        self.endLineEdit = self.findChild(QtWidgets.QLineEdit,"endLineEdit")
        self.classLineEdit = self.findChild(QtWidgets.QLineEdit,"classLineEdit")
        self.attributeTextEdit = self.findChild(QtWidgets.QTextEdit,"attributeTextEdit")
        
        self.buttonBox = self.findChild(QtWidgets.QDialogButtonBox,'buttonBox')
        self.buttonBox.button(QtWidgets.QDialogButtonBox.Ok).clicked.connect(lambda _: self.accept())
        #self.buttonBox.button(QtWidgets.QDialogButtonBox.Cancel).clicked.connect(lambda _: self.reject())
        
        self.changeClassButton = self.findChild(QtWidgets.QPushButton, 'changeClassButton')
        self.changeClassButton.clicked.connect(lambda _: self.changeClass())
        self.changeAttributesButton = self.findChild(QtWidgets.QPushButton, 'changeAttributesButton')
        self.changeAttributesButton.clicked.connect(lambda _: self.changeAttributes())

    def selectWindow(self,window_index:int):
        """Marks the window at window_index on the graph and loads its labels
        
        Arguments:
        ----------
        window_index : int
            index of the window that should be selected.
        ----------
        """
        
        #Looking at the Error attribute of the old window.
        #If it's 1 set the old windows colour to grayish red
        #otherwise to gray
        _,_,_,attributes = self.windows[self.current_window]
        if attributes[-1] == 0:
            self.window_bars[self.current_window].setOpts(brush=pg.mkBrush(0.5))
        else:
            self.window_bars[self.current_window].setOpts(brush=pg.mkBrush(200,100,100))
        
        #updating the new window
        self.current_window = window_index
        
        #loading the data of the current window
        start,end,class_index,attributes = self.windows[self.current_window]
        
        #checking the error attribute. if present colors the bar orange, otherwise yellow
        if attributes[-1] == 0:
            self.window_bars[self.current_window].setOpts(brush=pg.mkBrush('y'))
        else:
            self.window_bars[self.current_window].setOpts(brush=pg.mkBrush(255,200,50))
        
        #updates all textboxes
        self.startLineEdit.setText(str(start))
        self.endLineEdit.setText(str(end))
        self.classLineEdit.setText(self.classes[class_index])
        
        self.attributeTextEdit.clear()
        for i,attribute in enumerate(attributes):
            if attribute:
                self.attributeTextEdit.append(self.attributes[i])
            
    def changeClass(self):
        """Runs the saveClassesDialog to select a new class
        
        If a new class gets selected it gets saved.
        """
        current_class = self.windows[self.current_window][2]
        dlg = saveClassesDialog(self,current_class)
        class_index = dlg.exec_()
        if class_index >-1:
            self.window_bars[self.current_window].setOpts(y1=class_index+1)
            self.data.changeWindow(self.current_window,class_index=class_index) #TODO: minimize IO operations by setting save=False and save elsewhere
            self.selectWindow(self.current_window)
    
    def changeAttributes(self):
        """Runs the saveAttributesDialog to select new attributes
        
        If new attributes get selected they get saved.
        """
        current_attrbutes = self.windows[self.current_window][3]
        dlg = saveAttributesDialog(self,current_attrbutes)
        attribute_int = dlg.exec_()
        if attribute_int>-1:
            format_string = '{0:0'+str(self.data.attributes.__len__())+'b}'
            attributes=[(x is '1')+0 for x in list(format_string.format(attribute_int))]
            self.data.changeWindow(self.current_window,attributes=attributes) #TODO: minimize IO operations by setting save=False and save elsewhere
            self.selectWindow(self.current_window) 
        


class enterIDDialog(QtWidgets.QDialog):
    """Dialog for entering the annotatorID and the number of annotations
    
    example code:
        dlg = enterIDDialog(QWidget,io_controller,0)
        result = dlg.exec_()
        
    result is a boolean that shows wether the annotator aborted or confirmed
    if confirmed the annotatorID and number of annotations are saved to the io_controller
    
    """
    def __init__(self,parent,io_controller,annotatorID):
        """Initializes the dialog and sets up the gui

        Arguments:
        ----------
        parent : QWidget
            parent widget of this dialog
        io_controller : main.IO_Controller
            io_controller of the application
        annotatorID : int
            sets the initial value of the annotatorIDLineEdit for faster confirmation
        ----------
        """ 
        super(enterIDDialog, self).__init__(parent)
        self.setWindowTitle("Enter your ID and number of this attempt")
        self.io_controller = io_controller
        
        self.layout = QtWidgets.QGridLayout()
        
        self.layout.addWidget(QtWidgets.QLabel("Please fill out this information"),0,0,1,2)
        
        self.layout.addWidget(QtWidgets.QLabel("Annotator ID:"),1,0)
        self.layout.addWidget(QtWidgets.QLabel("Number of this annotation run:"),3,0)
        
        self.annotatorIDLineEdit = QtWidgets.QLineEdit(str(annotatorID))        
        self.annotatorIDLineEdit.setValidator(QIntValidator(0,1000))
        self.layout.addWidget(self.annotatorIDLineEdit,1,1)
        
        self.annotatorIDErrorLabel = QtWidgets.QLabel("")
        self.annotatorIDErrorLabel.setStyleSheet('color: red')
        self.layout.addWidget(self.annotatorIDErrorLabel,2,1)
        
        self.triesSpinBox = QtWidgets.QSpinBox()
        self.triesSpinBox.setMinimum(1)
        self.layout.addWidget(self.triesSpinBox,3,1)
        
        self.triesErrorLabel = QtWidgets.QLabel("")
        self.triesErrorLabel.setStyleSheet('color: red')
        self.layout.addWidget(self.triesErrorLabel,4,1)
        
        QBtn = QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        
        self.buttonBox = QtWidgets.QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(lambda: self.saveInputs())
        self.buttonBox.rejected.connect(lambda: self.reject())
        
        self.layout.addWidget(self.buttonBox,5,0,1,2)
        self.setLayout(self.layout)
        
    def saveInputs(self):
        """Saves the annotatorID and closes the dialog returning true
        
        """
        annotatorID = self.annotatorIDLineEdit.text()
        tries = self.triesSpinBox.value()
        
        if (annotatorID is not '') and (tries is not ''):
            self.io_controller.saveID(annotatorID,tries)
            self.accept()
        else:
            if annotatorID is '':
                self.annotatorIDErrorLabel.setText("This field is required")
            if tries is '':
                self.triesErrorLabel.setText("This field is required")
        
        
class Progress_Dialog(QtWidgets.QDialog):
    """Displays 2 Progressbars. The first for the progress of a single step. The second for progress of steps.
    
    example code:
        dlg = changeLabelsDialog(QWidget)
        _ = dlg.exec_()
    
    anytime the user changes a label it gets saved automatically 
    """
    
    def __init__(self,parent:QtWidgets.QWidget,process_name:str, num_steps:int): 
        """Initializes the dialog and sets up the gui

        Arguments:
        ----------
        parent : QWidget
            parent widget of this dialog
        process_name : str
            
        num_steps : int
        
        ----------
        """
        super(Progress_Dialog, self).__init__(parent)
        uic.loadUi('progressbar.ui', self)

        #self.findChild(qtclass,name)
        
        self.process_label = self.findChild(QtWidgets.QLabel, 'process_label')
        self.process_label.setText(process_name)
        
        self.step_label = self.findChild(QtWidgets.QLabel, 'step_label')
        
        self.step_progressBar = self.findChild(QtWidgets.QProgressBar, 'progressBar')
        
        self.steps_progressBar = self.findChild(QtWidgets.QProgressBar, 'progressBar_2')
        self.steps_progressBar.setMaximum(num_steps)
        
    def advanceStep(self,value):
        current_value = self.step_progressBar.value()+value
        self.step_progressBar.setValue(current_value)
    def setStep(self,value):
            self.step_progressBar.setValue(value)
    
    def advanceSteps(self,value):
        current_value = self.steps_progressBar.value()+value
        self.steps_progressBar.setValue(current_value)
        if current_value >= self.steps_progressBar.maximum():
            self.done(0)
        
    def newStep(self, step_name:str, max_value:int):
        self.step_progressBar.setValue(0)
        self.step_progressBar.setMaximum(max_value)
        self.advanceSteps(1)
        
        self.step_label.setText(step_name)
    

