# Logistic Activity Recognition Challenge (LARa) 
Implementation code for the Annotation Tool that is using for LARa dataset, which is published [Logistic Activity Recognition Challenge (LARa) – A Motion Capture and Inertial Measurement Dataset]

Updating in progress.

## Abstract

Optimizations in logistics require recognition and analysis of human activities. The potential of sensor-based human activity recognition (HAR) in logistics are not yet well explored. One reason is the lack of specific data. Although there has been a significant increase in HAR datasets in the past twenty years, no available dataset depicts activities in logistics. This contribution presents the first freely accessible logistics-dataset. In the 'Innovationlab Hybrid Services in Logistics' at TU Dortmund University, two picking and one packing scenarios with 14 subjects were recorded using OMoCap, IMUs, and an RGB camera. 758 minutes of recordings were labeled by 12 annotators in 474 person-hours. The subsequent revision was carried out by 4 revisers in 143 person-hours. All the given data have been labeled and categorized into 8 activity classes and $19$ binary coarse-semantic descriptions, also called attributes. The dataset is deployed for solving HAR using deep networks.

# Annotation_Tool_LARa
Annotation_Tool_LARa

![Annotation Tool](AnnotationTool.png)

## Prerequisites
The implementation is done in Python:
- torch
- numpy

## Dataset

LARa dataset can be downloaded in https://zenodo.org/record/3862782#.XtVJOy9h3UI

## Example

Running the `main.py` script in Annotation_Tool_LARa. 
- For using the tCNNs for predicting activities classes, download the 'class_network.pt' and 'attr_network.pt' from LARa dataset. 
- Store the networks 'class_network.pt' and 'attr_network.pt' in Annotation_Tool_LARa/networks/
  Annotation_Tool_LARa/networks/class_network.pt
  Annotation_Tool_LARa/networks/attr_network.pt
  

  - Erik Altermann        erik.altermann@tu-dortmund.de
  - Fernando Moya Rueda   fernando.moya@tu-dortmund.de
  
  
The work on this publication was supported by Deutsche Forschungsgemeinschaft (DFG) in the context of the project Fi799/10-2, HO2403/14-2 ''Transfer Learning for Human Activity Recognition in Logistics''.
