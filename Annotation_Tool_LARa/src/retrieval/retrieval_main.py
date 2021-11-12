import math
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

    # average_att_rep()

    paths = browse_files()
    # datasets = None

    for network_id, name in [(5, "w50 s12"), (6, "w100 s25"), (7, "w150 s25"), (3, "w200 s?")]:
        annotator = Annotator(network_id)

        map_attr_rep_sum_cos = 0
        map_attr_sum_cos = 0
        map_cls_sum_cos = 0

        map_attr_rep_sum_bce = 0
        map_attr_sum_bce = 0
        map_cls_sum_bce = 0

        ap_attr = [0 for i in range(len(g.attributes))]

        for i, path in enumerate(paths):
            file_name = os.path.split(path)[1]

            if g.windows is not None:
                g.windows.close()

            g.data = DataProcessor(path, True)
            g.windows = WindowProcessor(path, True, False)
            print(f"Current File: {file_name}. {i + 1}/{len(paths)}")

            dataset, graphs = annotator.run()
            # print(dataset.retrieve_list(0))

            """if datasets is None:
                datasets = dataset
            else:
                datasets.add_dataset(dataset)"""

            metric = "cosine"
            dataset.predict_classes_from_attributes(metric)
            dataset.predict_attribute_reps(metric)

            map_cls_sum_cos += dataset.mean_average_precision("classes")
            map_attr_rep_sum_cos += dataset.mean_average_precision("attr_rep")
            map_attr_sum_cos += dataset.mean_average_precision("attr")

            metric = "bce"
            dataset.predict_classes_from_attributes(metric)
            dataset.predict_attribute_reps(metric)

            map_cls_sum_bce += dataset.mean_average_precision("classes")
            map_attr_rep_sum_bce += dataset.mean_average_precision("attr_rep")
            map_attr_sum_bce += dataset.mean_average_precision("attr")

            for j in range(len(g.attributes)):
                ap = dataset.average_precision(attr_index=j)
                if not math.isnan(ap):
                    ap_attr[j] += ap

        # for j, attr in enumerate(g.attributes):
        #    print(f"Average Precision of {attr:<25}: {datasets.average_precision(attr_index=j)}")
        # for j, attr in enumerate(g.attribute_rep):
        #    avep = datasets.average_precision(attr_rep_index=j)
        #    if not math.isnan(avep):
        #        print(f"Average Precision of {attr}: {avep}")
        # for j, cls in enumerate(g.classes):
        #    print(f"Average Precision of {cls}: {datasets.average_precision(j)}")

        # print(f"MAP Attributes: {datasets.mean_average_precision(classes=False)}")
        # print(f"MAP Classes: {datasets.mean_average_precision(classes=True)}")
        print(f"Network: {name}")

        print(f"MMAP Attributes COS: {map_attr_sum_cos / len(paths)}")
        print(f"MMAP Attributes BCE: {map_attr_sum_bce / len(paths)}")
        print(f"MMAP Attribute Representations COS: {map_attr_rep_sum_cos / len(paths)}")
        print(f"MMAP Attribute Representations BCE: {map_attr_rep_sum_bce / len(paths)}")
        print(f"MMAP Classes COS: {map_cls_sum_cos / len(paths)}")
        print(f"MMAP Classes BCE: {map_cls_sum_bce / len(paths)}")

        for j, attr in enumerate(g.attributes):
            print(f"Averaged Average Precision of {attr:<25}: {ap_attr[j] / len(paths)}")

    window = QtWidgets.QWidget()
    window.show()
    app.exec_()
