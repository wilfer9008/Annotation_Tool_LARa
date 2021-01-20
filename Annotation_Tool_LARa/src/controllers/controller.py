'''
Created on 09.07.2020

@author: Erik Altermann
@email: Erik.Altermann@tu-dortmund.de
'''

import pyqtgraph as pg
#import pyqtgraph.opengl as gl
from pyqtgraph.functions import mkPen

import global_variables as g

class Controller():
    def __init__(self,gui):
        self.gui = gui
        self.enabled = False
        self.revision_mode_enabled = False
        self.statusWindow = None
        #self.setup_widgets()
        
    def setup_widgets(self):
        print("overwrite setup_widgets(self) in", type(self))
    
    def enable_widgets(self):
        print("overwrite enable_widgets(self) in", type(self))
    
    def reload(self):
        print("overwrite reload(self) in", type(self))
    
    def setEnabled(self,enable:bool):
        print("overwrite setEnabled(self,enable:bool) in", type(self))    
    
    def new_frame(self,frame):
        print("overwrite new_frame(self,frame) in", type(self))
    
    def revision_mode(self,enable:bool):
        print("overwrite revision_mode(self,enable) in", type(self))
    
    def get_start_frame(self):
        print("overwrite revision_mode(self,enable) in", type(self))
    
    def add_status_message(self,msg):
        self.statusWindow.append(msg+'\n')
        
        
    def class_window_index(self,frame):
        if frame is None:
            frame = self.gui.get_current_frame()
        for i, window in enumerate(g.data.windows):
            if window[0] <= frame and frame < window[1]:
                return i
        return None
        
    def highlight_class_bar(self,bar_index):
        """Generates a list of colors for use in classgraphs
        
        When using this method in subclasses of controller: 
        use super to get the colors and call color_class_bars(colors) on needed graphs
        """
        normal_color = 0.5 #gray
        error_color = 200,100,100 #gray-ish red
        selected_color = 'y' #yellow
        selected_error_color = 255,200,50 #orange
        
        num_windows = g.data.windows.__len__()
        
        colors = []
        for i in range(num_windows):
            if g.data.windows[i][3][-1] == 0:
                colors.append(normal_color)
            else:
                colors.append(error_color)
        
        if (bar_index is not None) :
            if (g.data.windows[bar_index][3][-1] == 0):
                colors[bar_index] = selected_color
            else:
                colors[bar_index] = selected_error_color
        return colors
        
        
    
class Graph():
    def __init__(self, plot_widget, graph_type, **kwargs):
        """
        
        graph_type: 'joint','class','attribute'
        
        kwargs = {'label','unit','AutoSIPrefix',interval_lines,
                  'number_classes'}
        
        """
        
        if graph_type not in ['joint','class','attribute']:
            raise ValueError
        
        self.graph = plot_widget
        self.graph_type = graph_type
        self.kwargs = kwargs
        
        self.graph.setMouseEnabled(False,False)
        
    def setup(self):
        
        self.graph.clear()
        self.graph.plot([])
        
        self.classbars = []
        
        if self.graph_type != 'attribute':
            if self.kwargs['interval_lines']:
                self.startline = pg.InfiniteLine(0,label='start',labelOpts={'anchors' : [(1, 1.5), (0, 1.5)]})
                self.graph.addItem(self.startline)
            
                self.endline = pg.InfiniteLine(0,label='end',labelOpts={'anchors' : [(0, 1.5), (1, 1.5)]})
                self.graph.addItem(self.endline)
            
            self.playline = pg.InfiniteLine(0,pen=mkPen(0,255,0,127))
            self.graph.addItem(self.playline)
            
            self.graph.setXRange(0,g.data.number_samples,padding=0.02)
            
        else:
            if 'label' in self.kwargs.keys():
                self.graph.getAxis('left').setLabel(text=self.kwargs['label'],units='')
            else:
                self.graph.getAxis('left').setLabel(text='Attributes',units='')
            for i,attribute in enumerate(g.data.attributes):
                bar = pg.BarGraphItem(x0=[i],x1=i+1,y0=0,y1=0,name=attribute)
                self.classbars.append(bar)
                self.graph.addItem(bar)
                self.graph.setYRange(0,1,padding=0.1)
                label = pg.TextItem(text=attribute, color='b', anchor=(0,0), border=None, fill=None, angle=-90, rotateAxis=None)
                label.setPos(i+1,1)
                
                self.graph.addItem(label)
            self.graph.setXRange(0,g.data.attributes.__len__(),padding=0.02)
            
        if self.graph_type == 'joint':
            self.graph.getAxis('left').setLabel(text=self.kwargs['label'], 
                                                units=self.kwargs['unit'])
            self.graph.getAxis('left').enableAutoSIPrefix(self.kwargs['AutoSIPrefix'])
        
        if self.graph_type == 'class':
            if 'label' in self.kwargs.keys():
                self.graph.getAxis('left').setLabel(text=self.kwargs['label'],units='')
            else:
                self.graph.getAxis('left').setLabel(text='Classes',units='')
            self.graph.setYRange(0,g.data.classes.__len__()+1,padding=0)
            
        
    def update_frame_lines(self,start=None,end = None,play=None):
        """Updates the framelines of the graph
        
        Cannot be used with attribute graphs
        """
        if self.kwargs['interval_lines']:
            if start is not None:
                self.startline.setValue(start)
            if end is not None:
                self.endline.setValue(end)
        if play is not None:
            self.playline.setValue(play)
        
    def update_plot(self,plot_data):
        """Updates Data for plotting on the graph.
        
        Should only be used with joint graphs
        """
        
        if plot_data is not None:
            self.graph.listDataItems()[0].setData(plot_data)
        else:
            self.graph.listDataItems()[0].setData([],[])
    
    def add_class(self, start,end,class_index,attributes):
        """Adds a classwindow to the graph. 
        
        Should only be used with class graphs
        """
        
        bar = pg.BarGraphItem(x0=[start],x1=end,y0=0,y1=class_index+1)
        self.classbars.append(bar)
        if attributes[-1] == 1:
            bar.setOpts(brush=pg.mkBrush(200,100,100))
        self.graph.addItem(bar)
        
    def reload_classes(self,windows):
        """Reloads a classwindow to the graph. 
        
        Should only be used with class graphs
        """
        self.setup()
        
        for start,end,class_index,attributes in windows:
            self.add_class(start,end,class_index,attributes)
            
    def update_attributes(self,attributes=None):
        """Updates attribute bars in graph
        
        Should only be used with attribute graphs
        """
        if attributes is not None:
            for i in range(self.classbars.__len__()):
                self.classbars[i].setOpts(y1=attributes[i])
        else:
            for i in range(self.classbars.__len__()):
                self.classbars[i].setOpts(y1=0)
    
    
    def color_class_bars(self, colors=None):
        """Colors each classbar with corresponding color in colors
        
        Can be used with attribute graph, but should only be used with class graphs
        """
        for bar,color in zip(self.classbars,colors):
            bar.setOpts(brush=pg.mkBrush(color))       
    
    
    def update_kwargs(self,**kwargs):
        """
        for kwargs explanation see __init__()
        """
        for k,v in kwargs.items():
            self.kwargs[k] = v
        self.setup()
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        
        
        
        
        
        
        
        