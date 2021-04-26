import sys
import os

from PyQt5.QtWidgets import QApplication
import pyqtgraph as pg

os.chdir("..")

from functional import *
import global_variables as g
from data_management import DataProcessor, WindowProcessor

pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

def except_hook(cls, exception, traceback):
    sys.__excepthook__(cls, exception, traceback)


if __name__ == "__main__":
    sys.excepthook = except_hook
    app = QApplication(sys.argv)

    paths = browse_files()
    for i, path in enumerate(paths):
        file_name = os.path.split(path)[1]

        if g.windows is not None:
            g.windows.close()

        g.data = DataProcessor(path, True)
        g.windows = WindowProcessor(path, True, False)
        print(f"Current File: {file_name}. {i+1}/{len(paths)}")
        annotator = Annotator(3)
        dataset, graphs = annotator.run()
        #print(dataset.retrieve_list(0))

        for i,cls in enumerate(g.classes):
            print(f"Average Precision of {cls}: {dataset.average_precision(i)}")

        print(f"MAP: {dataset.mean_average_precision()}")


    app.exec_()
