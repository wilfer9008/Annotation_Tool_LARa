'''
Created on 10.11.2020

@author: Erik Altermann
@email: Erik.Altermann@tu-dortmund.de


'''

import os
from PyQt5 import QtWidgets, QtGui

import global_variables as g
import sys
from PyQt5.QtWidgets import QApplication
import torch
from network import Network
from data_management import Labeled_sliding_window_dataset, Data_processor,\
    Deep_representation_dataset
import dill
import numpy as np
from dialogs import Plot_Dialog
import pyqtgraph as pg

pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

windows_path = f"..{os.sep}windows"

def browse_files(subject_id,caption):
    paths = QtWidgets.QFileDialog.getOpenFileNames(
                parent = None, 
                caption = caption,
                directory = g.settings['saveFinishedPath'], 
                filter = f'CSV Files (*{subject_id}*norm_data.csv)', 
                initialFilter = '')[0]
    return paths

def load_network(index):
    """Loads the selected network"""
    try:
        device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(g.networks_path + g.networks[index]['file_name'], 
                                map_location=device)
        network=None
            
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
        network=None
        raise e

def save_windows(classes, distances,i):
    file_name = g.data.file_name.split('.')[0]+f"_A{i}_N00_classes.txt"
    np.savetxt(windows_path+os.sep+file_name, classes)
    
    if distances is not None:
        file_name_distances = g.data.file_name.split('.')[0]+f"_A{i}_N00_distances.txt"
        np.savetxt(windows_path+os.sep+file_name_distances, distances)
    

def load_windows(files): 
    # A90 real labels
    # A91 method B labels
    # A92 method D labels
    # A93 method C labels
    real_classes_files = [f for f in files if "A90" in f]
    
    b_classes_files = [f for f in files if "A91" in f and "classes" in f]
    b_distance_files = [f for f in files if "A91" in f and "distances" in f]
    
    c_classes_files = [f for f in files if "A93" in f and "classes" in f]
    c_distance_files = [f for f in files if "A93" in f and "distances" in f]
    
    d_classes_files = [f for f in files if "A92" in f and "classes" in f]
    d_distance_files = [f for f in files if "A92" in f and "distances" in f]
    
    #making sure the sorting order is the same
    #for i in range(len(real_classes_files)):
    #    print(real_classes_files[i][:8] 
    #          == b_classes_files[i][:8] == b_distance_files[i][:8] 
    #          == c_classes_files[i][:8] == c_distance_files[i][:8]
    #          == d_classes_files[i][:8] == d_distance_files[i][:8])
    real_classes = None
    b_classes = None
    b_distances = None
    c_classes = None
    c_distances = None
    d_classes = None
    d_distances = None
    
    for file in real_classes_files :
        if real_classes is None:
            real_classes = np.loadtxt(file, dtype=np.int)
        else:
            real_classes = np.hstack((real_classes, np.loadtxt(file, dtype=np.int)))
    for file in b_classes_files :
        if b_classes is None:
            b_classes = np.loadtxt(file, dtype=np.int)
        else:
            b_classes = np.hstack((b_classes, np.loadtxt(file, dtype=np.int)))
    for file in b_distance_files :
        if b_distances is None:
            b_distances = np.loadtxt(file)
        else:
            b_distances = np.hstack((b_distances, np.loadtxt(file)))
    
    for file in c_classes_files :
        if c_classes is None:
            c_classes = np.loadtxt(file, dtype=np.int)
        else:
            c_classes = np.hstack((c_classes, np.loadtxt(file, dtype=np.int)))
    for file in c_distance_files :
        if c_distances is None:
            c_distances = np.loadtxt(file)
        else:
            c_distances = np.hstack((c_distances, np.loadtxt(file)))
    
    for file in d_classes_files :
        if d_classes is None:
            d_classes = np.loadtxt(file, dtype=np.int)
        else:
            d_classes = np.hstack((d_classes, np.loadtxt(file, dtype=np.int)))
    for file in d_distance_files :
        if d_distances is None:
            d_distances = np.loadtxt(file)
        else:
            d_distances = np.hstack((d_distances, np.loadtxt(file)))
    

    return real_classes, b_classes, b_distances, c_classes, c_distances, d_classes, d_distances

def get_deep_representations(paths, config, network,window_step):
        current_file_name = g.data.file_name
        name_parts = current_file_name.split('_')
        subject_id = [s for s in name_parts if 'S' in s][0]
        pickled_deep_rep_path = f"{g.settings['saveFinishedPath']}{os.sep}{subject_id}.p"
        if os.path.exists(pickled_deep_rep_path):
            deep_rep = dill.load(open(pickled_deep_rep_path,"rb"))
            existing_files = deep_rep.file_names
            new_files = [os.path.split(path)[1] for path in paths]
            
            if [file for file in existing_files if file not in new_files] != []:
                #The deep_rep has files that are not needed. Better make new deep_rep
                #print("making new deep_rep. unneeded files")
                deep_rep = None
            elif [file for file in new_files if file not in existing_files] != []:
                #There are new files that need to be added to deep_rep. 
                #It will be updated in for loop
                #print("updating deep_rep. too few files")
                pass
            else:
                #existing and new files are identical.
                #print("returning old deep_rep. identical file list")
                return deep_rep
        else:
            deep_rep = None
        
        for path in paths:
            
            #getting the data
            data = np.loadtxt(path, delimiter=',', skiprows=1)
            data = data[:,2:]
            
            #Getting windows file path
            directory, data_name = os.path.split(path)
            window_name_parts = data_name.split('_')[:5]
            window_name_parts.append("windows.txt")
            window_name = window_name_parts[0]
            for part in window_name_parts[1:]:
                window_name += "_"+part
            window_path = directory+os.sep+window_name
            
            #reading the windows_file
            windows = []
            with open(window_path, 'r+t') as windows_file:
                lines = windows_file.readlines()
                for line in lines:
                    window = eval(line[:-1])
                    windows.append(window)
            
            if deep_rep is None:
                window_length = config['sliding_window_length']
                deep_rep = Deep_representation_dataset(data, window_length, 
                                                       window_step, data_name, 
                                                       windows, network)
            else:
                deep_rep.add_deep_rep_data(data_name, data, windows,network)        
        dill.dump(deep_rep,open(pickled_deep_rep_path,"wb"))
        return deep_rep


def method_d(labels_15, labels_16, boost_15, boost_16):
    #use normal network
    #compute both attr rep and deep rep
    #return both
    
    network, config, att_rep = load_network(3)
    network.deep_rep = True
    
    window_length = config['sliding_window_length']
    window_step = g.settings['segmentationWindowStride']
    label_kind = config['labeltype']
    
    attr_datasets = []
    deep_datasets = []
    for i, files, boost in [(15, labels_15, boost_15), (16, labels_16, labels_16)]:        
        for file in files:
            print(f"method D: processing file {file}")
            g.data = Data_processor(file, False, True)
            deep_rep = get_deep_representations(boost, config, network, window_step)
            
            dataset = Labeled_sliding_window_dataset(g.data.mocap_data, window_length, window_step)
            dataset_len = dataset.__len__()
            #dataset2 = Labeled_sliding_window_dataset(g.data.mocap_data, window_length, window_step)

    
            for i in range(dataset_len):
                label,fc2 = network(dataset.__getitem__(i))
                deep_rep.save_fc2(i, fc2)
                if label_kind == 'class':
                    label = torch.argsort(label, descending=True)[0]
                    dataset.save_labels(i, label, label_kind)
                    #dataset2.save_labels(i, label, label_kind)
                elif label_kind == 'attributes':
                    label = label.detach()
                    dataset.save_labels(i, label[0], label_kind)
                    #dataset2.save_labels(i, label, label_kind)
                else:
                    raise Exception
            deep_rep.predict_labels_from_fc2()
            metrics = att_rep
            
            #saving method b A91
            _, _, _ = dataset.make_windows(label_kind, False, metrics, None)
            save_windows(dataset.classes[:,0], dataset.top3_distances[:,0], 91)
        
            
            #Saving method d A92
            #dataset2.make_windows(label_kind, True, metrics, deep_rep)
            #save_windows(dataset2.classes[:,0], deep_rep.top3_distances[:,0], 92)
            
            
            save_windows(deep_rep.classes[:,0], deep_rep.top3_distances[:,0], 92)
            
            attr_datasets.append(dataset)
            deep_datasets.append(deep_rep)
            
    return attr_datasets, deep_datasets
            
def method_c(files):
    #use retrained network
    #return dataset labels from it.
    
    network, config, att_rep = load_network(4)
    
    window_length = config['sliding_window_length']
    window_step = g.settings['segmentationWindowStride']
    label_kind = config['labeltype']
    
    datasets = []
    for file in files:
        print(f"method C: processing file {file}")
        g.data = Data_processor(file, False, True)
        
        dataset = Labeled_sliding_window_dataset(g.data.mocap_data, window_length, window_step)
        dataset_len = dataset.__len__()
    
        for i in range(dataset_len):
            label = network(dataset.__getitem__(i))

            if label_kind == 'class':
                label = torch.argsort(label, descending=True)[0]
                dataset.save_labels(i, label, label_kind)
            elif label_kind == 'attributes':
                label = label.detach()
                dataset.save_labels(i, label[0], label_kind)
            else:
                raise Exception
        metrics = att_rep    
        _, _, _ = dataset.make_windows(label_kind, True, metrics, None)
        datasets.append(dataset)
        
        #real labels A90
        #g.data.createBackup(windows_path, f"A{g.networks[1]['annotator_id']}_N00")
        windows = [list(np.repeat(class_, end-start)) for (start, end, class_, _) in g.data.windows]
        real_classes = []
        for window in windows:
            real_classes.extend(window)
        real_classes = np.array(real_classes,dtype = int)
        save_windows(real_classes, None, 90)
        
        #method C A93
        #g.data.windows_1 = windows_1
        #g.data.windows_2 = windows_2
        #g.data.windows_3 = windows_3
        #g.data.savePredictions(windows_path, g.networks[4]['annotator_id'])
        save_windows(dataset.classes[:,0], dataset.top3_distances[:,0], 93)
        
        
    return datasets

def make_histogramms_2(real_classes, baseline_classes, test_classes, test_distances,title="Title"):
        improved_color = "b"
        same_color = (127,127,127,255)#gray
        worse_color = "r"
        
        divisions = 40
        improved_distances = test_distances[np.array(
            [a!=b and a==c for a,b,c in zip(real_classes,baseline_classes,test_classes)])]
        same_distances = test_distances[np.array(
            [(a==b and a==c) or (a!=b and a!=c) for a,b,c in zip(real_classes,baseline_classes,test_classes)])]
        worse_distances = test_distances[np.array(
            [a==b and a!=c for a,b,c in zip(real_classes,baseline_classes,test_classes)])]
        
        improved_y,improved_x = np.histogram(improved_distances, bins=divisions, range=(0,0.40))
        same_y,same_x = np.histogram(same_distances, bins=divisions, range=(0,0.4))
        worse_y,worse_x = np.histogram(worse_distances, bins=divisions, range=(0,0.4))
        
        dlg = Plot_Dialog(window)
        dlg.setWindowTitle("Graph")
        plot = dlg.graph_widget()
        plot.setTitle(f'<font size="6"><b>{title}</b></font>')
        legend = plot.addLegend(offset=(-10,15), labelTextSize='20pt')
        
        #plot.getAxis('left').setLabel("")
        plot.getAxis('bottom').setLabel("1-Cosine Similarity", **{'font-size': '14pt'})
        
        plot.plot(improved_x, improved_y/8, pen=improved_color, 
                  name=f"Improved predictions. Total: {int(np.sum(improved_y/8))}",
                  stepMode=True, fillLevel=0, fillOutline=True, brush=(0,0,255,100),)
        plot.plot(same_x, same_y/8, pen=same_color, 
                  name=f"Equal predictions. Total: {int(np.sum(same_y/8))}",
                  stepMode=True, fillLevel=0, fillOutline=True, brush=(200,200,200,100),)
        plot.plot(worse_x, worse_y/8, pen=worse_color, 
                  name=f"Worsened predictions. Total: {int(np.sum(worse_y/8))}",
                  stepMode=True, fillLevel=0, fillOutline=True, brush=(255,0,0,100),)
        
        # CHANGE THE FONT SIZE AND COLOR OF ALL LEGENDS LABEL
        legendLabelStyle = {'size': '14pt', 'bold': True, 'italic': False}
        for item in plot.addLegend().items:
            for single_item in item:
                if isinstance(single_item, pg.LabelItem):
                    single_item.setText(single_item.text, **legendLabelStyle)
        
        font = QtGui.QFont()
        font.setPixelSize(16)
        plot.getAxis("bottom").setTickFont(font)
        plot.getAxis("left").setTickFont(font)
        
        #print(plot.getContentsMargins())
        #plot.setContentsMargins(500,0,500,0)
        #print(plot.getContentsMargins())
        plot.getAxis('left').textWidth = 200
        plot.getAxis('left').setWidth(50)

        
        _ = dlg.show()
        return dlg
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = QtWidgets.QWidget()
    window.show()
    """
    S15_label_files = browse_files(15, "label")
    S16_label_files = browse_files(16, "label")
    S15_boost_files = browse_files(15, "boost")
    S16_boost_files = browse_files(16, "boost")
    
    files = []
    files.extend(S15_label_files)
    files.extend(S16_label_files)
    baseline, boosted = method_d(S15_label_files, S16_label_files, S15_boost_files, S16_boost_files)
    retrained = method_c(files)
    """
    # A90 real labels
    # A91 method B labels
    # A92 method D labels
    # A93 method C labels
    
    paths = QtWidgets.QFileDialog.getOpenFileNames(
                parent = None, 
                caption = "classes and distances",
                directory = windows_path, 
                filter = '', 
                initialFilter = '')[0]
    real_classes, b_classes, b_distances, c_classes, c_distances, \
        d_classes, d_distances = load_windows(paths)
    
    #dlg = make_histogramms_2(real_classes, b_classes, c_classes, c_distances, "B vs C")
    dlg = make_histogramms_2(real_classes, b_classes, d_classes, d_distances, "Method B vs NN<sub>d</sub>")
    dlg = make_histogramms_2(real_classes, c_classes, d_classes, d_distances, "Method C vs NN<sub>d</sub>")
    #dlg = make_histogramms_2(real_classes, d_classes, c_classes, c_distances, "D vs C")
    
    
    app.exec_()
    
    