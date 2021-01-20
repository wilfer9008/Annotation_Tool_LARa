'''
Created on 30.11.2019

@author: Erik Altermann
@email: Erik.Altermann@tu-dortmund.de
'''

from PyQt5 import QtWidgets, uic
from PyQt5.QtGui import QIntValidator
import pyqtgraph as pg

from data_management import Data_processor

from PyQt5.QtCore import QThread
import os
#from main import IO_Controller

import global_variables as g
import json

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
    
    def __init__(self,parent:QtWidgets.QWidget): 
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
        uic.loadUi('..'+os.sep+'settings.ui', self)
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
        
        if g.settings['floorGrid']:
            self.floorEnableButton.setChecked(True)
        else:
            self.floorDisableButton.setChecked(True)
        
        if g.settings['dynamicFloor']:
            self.dynamicEnableButton.setChecked(True)
        else:
            self.dynamicDisableButton.setChecked(True)
        
        self.fastLineEdit.setText(str(g.settings['fastSpeed']))
        self.normalLineEdit.setText(str(g.settings['normalSpeed']))
        self.slowLineEdit.setText(str(g.settings['slowSpeed']))
        
        self.annotatorLineEdit.setText(str(g.settings['annotatorID']))
        
        self.unlabeledLineEdit.setText(str(g.settings['openFilePath']))
        self.labeledLineEdit.setText(str(g.settings['saveFinishedPath']))
        
        
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
        if directory != '':
            if directoryType == 'unlabeled':
                self.unlabeledLineEdit.setText(directory)
            elif directoryType == 'labeled':
                self.labeledLineEdit.setText(directory)
            
    def accept(self, *args, **kwargs):
        """Computes the result and saves it to changed_settings
        
        Overwritten method from QtWidgets.QDialog
        gets called when the dialog gets closed via the ok button and closes the dialog
        
        """
        
        g.settings['floorGrid'] = self.floorEnableButton.isChecked()
        g.settings['dynamicFloor']= self.dynamicEnableButton.isChecked()        
        
        int_lineEdits = [self.fastLineEdit,self.normalLineEdit,self.slowLineEdit,self.annotatorLineEdit]
        int_setting = ['fastSpeed','normalSpeed','slowSpeed','annotatorID']
        
        for lineSetting,lineEdit in zip(int_setting,int_lineEdits):
            if not lineEdit.text() == '':
                g.settings[lineSetting] = int(lineEdit.text())
            
        str_lineEdits = [self.unlabeledLineEdit,self.labeledLineEdit]
        str_setting = ['openFilePath','saveFinishedPath']
        
        for lineSetting,lineEdit in zip(str_setting,str_lineEdits):
            g.settings[lineSetting] = lineEdit.text()
        
        
        with open(g.settings_path, 'w') as f:
            json.dump(g.settings, f)
        
        
        return QtWidgets.QDialog.accept(self, *args, **kwargs)


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
        
        if (annotatorID != '') and (tries != ''):
            self.io_controller.save_id(annotatorID,tries)
            self.accept()
        else:
            if annotatorID == '':
                self.annotatorIDErrorLabel.setText("This field is required")
            if tries == '':
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
        uic.loadUi('..'+os.sep+'progressbar.ui', self)

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
    

class open_file_dialog(QtWidgets.QDialog):
    """Dialog to select a new File to open.
    
    example code:
        dlg = open_file_dialog(QWidget)
        result = dlg.exec_()
    
    anytime the user changes a label it gets saved automatically 
    """
    
    def __init__(self,parent:QtWidgets.QWidget): 
        """Initializes the dialog and sets up the gui

        Arguments:
        ----------
        parent : QWidget
            parent widget of this dialog
        data : data_management.Data_processor
        ----------
        """
        super(open_file_dialog, self).__init__(parent)
        uic.loadUi('..'+os.sep+'openfile.ui', self)
        
        self.load_backup = False
        self.result = None
        
        browse_button = self.findChild(QtWidgets.QPushButton,"browse_button")
        browse_button.clicked.connect(lambda _: self.browse())
        
        self.path_lineEdit = self.findChild(QtWidgets.QLineEdit, "path_lineEdit")
        self.path_lineEdit.textChanged.connect(self.check_path)
        
        self.backup_button = self.findChild(QtWidgets.QRadioButton, 'backup_radioButton')
        self.backup_button.clicked.connect(lambda _: self.set_load_backup(True))
        self.scratch_button = self.findChild(QtWidgets.QRadioButton, 'scratch_radioButton')
        self.scratch_button.clicked.connect(lambda _: self.set_load_backup(False))
        
        self.new_button = self.findChild(QtWidgets.QRadioButton, "new_radioButton")
        self.new_button.clicked.connect(lambda _: self.scratch_button.setText("Start from Scratch"))
        self.new_button.clicked.connect(lambda _: self.backup_button.setText("Load Backup"))
        
        self.annotated_button = self.findChild(QtWidgets.QRadioButton, "annotated_radioButton")
        self.annotated_button.clicked.connect(lambda _: self.scratch_button.setText("Load original labels"))
        self.annotated_button.clicked.connect(lambda _: self.backup_button.setText("Load modified labels"))
        
        self.buttonBox = self.findChild(QtWidgets.QDialogButtonBox, 'buttonBox')
        self.ok_button = self.buttonBox.button(QtWidgets.QDialogButtonBox.Ok)
        self.ok_button.setEnabled(False)
        
    def set_load_backup(self, load):
        self.load_backup = load
        
    def browse(self):
        
        if self.new_button.isChecked():
            directory = g.settings['openFilePath']
            message = 'Select an unlabeled .csv file'
            filter_ = 'CSV Files (*.csv)'
        else:
            directory = g.settings['saveFinishedPath']
            message = 'Select an _norm_data.csv file'
            filter_ = 'CSV Files (*norm_data.csv)'
        
        file,_ = QtWidgets.QFileDialog.getOpenFileName(self, \
                message, directory, filter_, '')
        
        if file != '':
            self.path_lineEdit.setText(file)
            
    def check_path(self,new_path):
        
        
        
        file_name = os.path.split(new_path)[1]
        
        name_fragments = file_name.split('.')[0].split('_')
        raw_data_name = name_fragments[0]
        for fragment in name_fragments[1:3]: 
        #for fragment in name_fragments[1:]:
            raw_data_name += "_"+fragment
        file_name = raw_data_name
        
        #TODO: make this into a setting
        backup_path = '..'+os.sep+'backups'+os.sep+file_name+'_backup.txt'
        
        self.ok_button.setEnabled(os.path.exists(new_path))
        
        if os.path.exists(backup_path):
            self.backup_button.setEnabled(True)
            self.scratch_button.setEnabled(True)
            self.load_backup = self.backup_button.isChecked()
        else:
            self.backup_button.setEnabled(False)
            self.scratch_button.setEnabled(False)
            self.load_backup = False
            
    def accept(self, *args, **kwargs):
        
        annotated = self.annotated_button.isChecked()
        
        path = self.path_lineEdit.text()      
        
        self.result = (path, annotated, self.load_backup)
        
        return QtWidgets.QDialog.accept(self, *args, **kwargs)
    
        
class Plot_Dialog(QtWidgets.QDialog):
    """
    example code:
        dlg = Plot_Dialog(QWidget)
        _ = dlg.exec_()
    
    """
    
    def __init__(self,parent:QtWidgets.QWidget = None): 
        """Initializes the dialog and sets up the gui

        Arguments:
        ----------
        parent : QWidget
            parent widget of this dialog
        ----------
        """
        
        super(Plot_Dialog, self).__init__(parent)
        uic.loadUi('..'+os.sep+'plotwindow.ui', self)
        
    def graph_widget(self):
        return self.findChild(pg.PlotWidget, 'plot')


