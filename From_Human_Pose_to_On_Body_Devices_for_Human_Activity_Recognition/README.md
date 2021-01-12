# ICPR20:From Human Pose to On Body Devices for Human Activity Recognition
Implementation code for "From Human Pose to On Body Devices for Human Activity Recognition.

Updating in progress.

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
  

## Contact

  - Fernando Moya Rueda   fernando.moya@tu-dortmund.de
  
Technische University of Dortmund
Department of Computer Science
Dortmund, Germany
  
  
The work on this publication was supported by Deutsche Forschungsgemeinschaft (DFG) in the context of the project Fi799/10-2, HO2403/14-2 ''Transfer Learning for Human Activity Recognition in Logistics''.
