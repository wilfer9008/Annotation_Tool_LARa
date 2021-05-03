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

        for j, attr in enumerate(g.attribute_rep):
            avep = dataset.average_precision(attr_index=j)
            if not math.isnan(avep):
                print(f"Average Precision of {attr}: {avep}")
        for j, cls in enumerate(g.classes):
            print(f"Average Precision of {cls}: {dataset.average_precision(j)}")

        print(f"MAP Attributes: {dataset.mean_average_precision(classes=False)}")
        print(f"MAP Classes: {dataset.mean_average_precision(classes=True)}")

        """query_index = 5

        avep = dataset.average_precision(attr_index=5)
        print(f"Average Precision of 5: {avep}")

        print(g.attribute_rep[5, 1:])
        print(dataset.retrieve_attr_list(attr_index=query_index), 10)
        """

    app.exec_()
