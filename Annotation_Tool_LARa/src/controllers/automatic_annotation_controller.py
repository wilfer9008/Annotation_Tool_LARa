'''
Created on 09.07.2020

@author: Erik Altermann
@email: Erik.Altermann@tu-dortmund.de
'''
from PyQt5 import QtWidgets


from .controller import Controller
from dialogs import Progress_Dialog
from PyQt5.QtCore import QThread, pyqtSignal
import torch
import os
from network import Network
from data_management import Labeled_sliding_window_dataset
import time


import global_variables as g
import pyqtgraph as pg
from controllers.controller import Graph


class Automatic_Annotation_Controller(Controller):
    def __init__(self,gui):
        super(Automatic_Annotation_Controller, self).__init__(gui)
        
        self.window_step = g.settings['segmentationWindowStride']
        
        self.selected_network = 0 #TODO: Save last selected in settings
        self.current_window = -1
        
        self.setup_widgets()
        
    def setup_widgets(self):
        #ComboBoxes
        self.network_comboBox = self.gui.get_widget(QtWidgets.QComboBox,"aa_network_comboBox")
        self.network_comboBox.currentIndexChanged.connect(self.select_network)
        for k in sorted(g.networks.keys()):
            self.network_comboBox.addItem(g.networks[k]['name'])
            
        
        self.post_processing_comboBox = self.gui.get_widget(QtWidgets.QComboBox,"aa_post_processing_comboBox")
        

        #Buttons
        self.annotate_button = self.gui.get_widget(QtWidgets.QPushButton,"aa_annotate_button")
        self.annotate_button.clicked.connect(lambda _: self.gui.pause())
        self.annotate_button.clicked.connect(lambda _: self.annotate())
        
        #self.annotate_folder_button = self.gui.get_widget(QtWidgets.QPushButton,"aa_annotate_folder_button")
        #self.annotate_folder_button.clicked.connect(lambda _: self.annotate())
        
        self.average_attributes_checkBox = self.gui.get_widget(QtWidgets.QCheckBox,"aa_average_attributes_checkBox")
        
        self.load_predictions_button = self.gui.get_widget(QtWidgets.QPushButton,"aa_load_prediction_button")
        self.load_predictions_button.clicked.connect(lambda _: self.load_predictions())
        
        #Graphs
        self.class_graph_1 = self.gui.get_widget(pg.PlotWidget,'aa_classGraph')
        self.class_graph_2 = self.gui.get_widget(pg.PlotWidget,'aa_classGraph_2') 
        self.class_graph_3 = self.gui.get_widget(pg.PlotWidget,'aa_classGraph_3')
        self.attribute_graph = self.gui.get_widget(pg.PlotWidget,'aa_attributeGraph')
        
        #Status window
        self.statusWindow = self.gui.get_widget(QtWidgets.QTextEdit, 'aa_statusWindow')
        self.add_status_message("This mode is for using a Neural Network to annotate Data.")
        
    def enable_widgets(self):
        if not self.enabled:
            self.class_graph_1 = Graph(self.class_graph_1, 'class', 
                                     interval_lines=False, label='Classes #1')
            self.class_graph_2 = Graph(self.class_graph_2, 'class', 
                                     interval_lines=False, label='Classes #2')
            self.class_graph_3 = Graph(self.class_graph_3, 'class', 
                                     interval_lines=False, label='Classes #3')
            self.attribute_graph = Graph(self.attribute_graph, 'attribute',interval_lines = False)
            
            
            
            self.enabled = True
        
        self.class_graph_1.setup()
        self.class_graph_2.setup()
        self.class_graph_3.setup()
        self.attribute_graph.setup()
        
        self.reload()
        self.enable_annotate_button()
        self.enable_load_button()

    def enable_annotate_button(self):
        if self.selected_network>0\
                and self.enabled\
                and not self.revision_mode_enabled:
            self.annotate_button.setEnabled(True)
        else:
            self.annotate_button.setEnabled(False)
            #self.annotate_folder_button.setEnabled(False)
    
    def enable_load_button(self):
        if self.selected_network>0 \
                and self.enabled\
                and not self.revision_mode_enabled:
            directory = g.settings['saveFinishedPath']
            annotator_id =  g.networks[self.selected_network]['annotator_id']
            
            files_present = True
            for pred_id in range(3):
                file_name = f"{g.data.file_name.split('.')[0]}_A{annotator_id:0>2}_N{pred_id:0>2}.txt"
                path = directory+os.sep+file_name
                if not os.path.exists(path):
                    files_present = False
                    
            if files_present:
                self.load_predictions_button.setEnabled(True)
            else:
                self.load_predictions_button.setEnabled(False)
            
        else:
            self.load_predictions_button.setEnabled(False)
    
    def reload(self):
        if g.data is not None \
                and g.data.windows_1 is not None \
                and g.data.windows_1.__len__() >0:
            
            graphs = [self.class_graph_1, self.class_graph_2, self.class_graph_3]
            windows = [g.data.windows_1, g.data.windows_2, g.data.windows_3]
            for graph,window in zip(graphs,windows):
                graph.reload_classes(window)
            
            
            self.select_window_by_frame()
            self.selectWindow(self.current_window)
            self.highlight_class_bar(self.current_window)
                
    def select_network(self,index):
        """Saves the selected network and tries to activate annotation if one was selected"""
        self.selected_network = index
        if index == 2:
            self.average_attributes_checkBox.setEnabled(True)
        else:
            self.average_attributes_checkBox.setEnabled(False)
        self.enable_annotate_button()
        self.enable_load_button()
    
    def annotate(self):
        self.annotate_start_time = time.time()
        
        self.progress = Progress_Dialog(self.gui, "annotating", 7)
        
        self.annotator = Annotator(self.gui,self.selected_network)
        self.annotator.progress.connect(self.progress.setStep)
        self.annotator.nextstep.connect(self.progress.newStep)
        self.annotator.cancel.connect(lambda _: self.cancel_annotation())
        self.annotator.done.connect(lambda _: self.finish_annotation())
        
        self.attribute_graph.update_attributes(None) 
        #for i in range(g.data.attributes.__len__()):
        #    self.gui.graphics_controller.attr_bars[i].setOpts(y1=0)
         
        
        self.progress.show()
        self.annotator.start()

        
    def finish_annotation(self):
        self.reload()
        #print("windows_1: ", g.data.windows_1)
        #print("windows_2: ", g.data.windows_2)
        #print("windows_3: ", g.data.windows_3)
        self.time_annotation()
        
    def cancel_annotation(self):
        self.progress.close()
        self.time_annotation()
        
    def time_annotation(self):
        annotate_end_time = time.time()
        time_elapsed = int(annotate_end_time - self.annotate_start_time)
        seconds = time_elapsed%60
        minutes = (time_elapsed//60)%60
        hours = time_elapsed//3600
        #print(time_elapsed)
        self.add_status_message("The annotation took {}:{}:{}".format(hours,minutes,seconds))
        
        
    def load_predictions(self):
        g.data.loadPredictions(g.settings['saveFinishedPath'],
                               g.networks[self.selected_network]['annotator_id'])
        self.reload()
        #print("windows_1: ", g.data.windows_1)
        #print("windows_2: ", g.data.windows_2)
        #print("windows_3: ", g.data.windows_3)
        
        
    def new_frame(self, frame):
        
        classgraphs = [self.class_graph_1, self.class_graph_2, self.class_graph_3]
        
        for graph in classgraphs:
            graph.update_frame_lines(play=frame)
        
        
            
        if g.data is not None \
                and g.data.windows_1 is not None \
                and g.data.windows_1.__len__() >0:
            
            window_index = self.class_window_index(frame)
            if (self.current_window != window_index):
                self.current_window = window_index
                self.selectWindow(self.current_window)
                self.highlight_class_bar(self.current_window)
            
    def class_window_index(self,frame):
        if frame is None:
            frame = self.gui.getCurrentFrame()
        for i, window in enumerate(g.data.windows_1):
            if window[0] <= frame and frame < window[1]:
                return i
        return None
    
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
            self.selectWindow(window_index)
        else:
            self.current_window = window_index
    
    def selectWindow(self,window_index:int):
        """Selects the window at window_index"""
        
        if window_index >= 0:
            self.current_window = window_index
                
            #needs to update shown attributes and start-, end-lines for top3 graphs
            #start end and attributes are the same in each prediction
        
            #classgraphs = [self.class_graph_1, self.class_graph_2, self.class_graph_3]
            #for graph in classgraphs:
                #graph.update_frame_lines(start, end)

            _, _, _, attributes = g.data.windows_1[self.current_window]
            self.attribute_graph.update_attributes(attributes)
    
    def highlight_class_bar(self, bar_index):
        
        normal_color = 0.5 #gray
        error_color = 200,100,100 #gray-ish red
        selected_color = 'y' #yellow
        selected_error_color = 255,200,50 #orange
        
        num_windows = g.data.windows_1.__len__()
        
        colors = []
        for i in range(num_windows):
            if g.data.windows_1[i][3][-1] == 0:
                colors.append(normal_color)
            else:
                colors.append(error_color)
        
        if (bar_index is not None) :
            if (g.data.windows_1[bar_index][3][-1] == 0):
                colors[bar_index] = selected_color
            else:
                colors[bar_index] = selected_error_color
        
        self.class_graph_1.color_class_bars(colors)
        self.class_graph_2.color_class_bars(colors)
        self.class_graph_3.color_class_bars(colors)
    
    def revision_mode(self, enable:bool):
        #Controller.revision_mode(self, enable)
        self.revision_mode_enabled = enable
        
        self.enable_annotate_button()
        self.enable_load_button()
    
    def get_start_frame(self) -> int:
        """returns the start of the current window"""
        if g.data.windows_1 is not None and g.data.windows_1.__len__() >0:
            return g.data.windows_1[self.current_window][0]+1
        return self.gui.getCurrentFrame()
    
class Annotator(QThread):
    progress = pyqtSignal(int)
    nextstep = pyqtSignal(str,int)
    done = pyqtSignal(int)
    cancel = pyqtSignal(int)
    
    def __init__(self,gui,selected_network):
        super(Annotator, self).__init__()
        self.gui = gui
        self.selected_network = selected_network
        self.window_step = g.settings['segmentationWindowStride']
        self.device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")
        
    def load_network(self,index):
        """Loads the selected network"""
        try:
            checkpoint = torch.load(g.networks_path + g.networks[index]['file_name'], 
                                    map_location=self.device)
            self.network=None
            
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
    
        except KeyError as e:
            self.network=None
            self.cancel.emit(0)
            raise e
            
        
    
    def run(self):
        self.nextstep.emit("loading network", 1)
        
        #Load network
        #print("loading network")
        network,config,att_rep = self.load_network(self.selected_network)
        #print("network loaded")
        
        self.nextstep.emit("segmenting", 1)
        
        #Segment Data
        #print("segmenting data")
        window_length = config['sliding_window_length']
        dataset = Labeled_sliding_window_dataset(g.data.mocap_data, window_length, self.window_step)
        dataset_len = dataset.__len__()
        #print("data segmented")
        
        
        #Forward through network
        self.nextstep.emit("annotating", dataset_len)
        
        label_kind = config['labeltype']
        
        for i in range(dataset_len):
            label = network(dataset.__getitem__(i))
            #print(label)
            
            #label = torch.argmax(label).item()
            #dataset.setlabel(i, label, label_kind)
            if label_kind == 'class':
                label = torch.argsort(label, descending=True)[0]
                #print(label)
                dataset.save_labels(i, label, label_kind)
            elif label_kind == 'attributes':
                label = label.detach()
                dataset.save_labels(i, label[0], label_kind)
            else:
                raise Exception
            
            #print(str(i+1)+"/"+str(dataset_len)+"\tRange:"+str(self.dataset.__range__(i)))
            self.progress.emit(i)
        
        #snippet = dataset.attributes[:,100,:]
        
        
        #Evaluate results
        self.nextstep.emit("evaluating", 1)
        
        #windows = dataset.evaluate()
        
        average_checkbox = self.gui.get_widget(QtWidgets.QCheckBox,"aa_average_attributes_checkBox")
        if att_rep is not None:
            #metrics = Metrics(config, self.device, att_rep)
            metrics = att_rep
            
        else:
            metrics = None
        windows_1,windows_2,windows_3 = dataset.make_windows(
            label_kind, average_checkbox.isChecked(), metrics)

        
        #Save windows
        self.nextstep.emit("saving", 1)
        g.data.windows_1 = windows_1
        g.data.windows_2 = windows_2
        g.data.windows_3 = windows_3
        g.data.savePredictions(g.settings['saveFinishedPath'],
                               g.networks[self.selected_network]['annotator_id'])
        
        
        
        #Merge windows
        self.nextstep.emit("cleaning up", 1)#TODO: eliminate this step
        
        self.nextstep.emit("done", 0)
        self.done.emit(0)


    


