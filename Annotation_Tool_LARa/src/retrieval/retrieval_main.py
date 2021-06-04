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
    datasets = None
    annotator = Annotator(3)

    map_attr_sum = 0
    map_cls_sum = 0

    for i, path in enumerate(paths):
        file_name = os.path.split(path)[1]

        if g.windows is not None:
            g.windows.close()

        g.data = DataProcessor(path, True)
        g.windows = WindowProcessor(path, True, False)
        print(f"Current File: {file_name}. {i+1}/{len(paths)}")

        dataset, graphs = annotator.run()
        #print(dataset.retrieve_list(0))

        if datasets is None:
            datasets = dataset
        else:
            datasets.add_dataset(dataset)

        map_cls_sum += dataset.mean_average_precision(classes=True)
        map_attr_sum += dataset.mean_average_precision(classes=False)


    #for j, attr in enumerate(g.attribute_rep):
    #    avep = datasets.average_precision(attr_index=j)
    #    if not math.isnan(avep):
    #        print(f"Average Precision of {attr}: {avep}")
    #for j, cls in enumerate(g.classes):
    #    print(f"Average Precision of {cls}: {datasets.average_precision(j)}")

    #print(f"MAP Attributes: {datasets.mean_average_precision(classes=False)}")
    #print(f"MAP Classes: {datasets.mean_average_precision(classes=True)}")

    print(f"MMAP Attributes: {map_attr_sum/len(paths)}")
    print(f"MMAP Classes: {map_cls_sum/len(paths)}")





    window = QtWidgets.QWidget()
    window.show()
    app.exec_()
