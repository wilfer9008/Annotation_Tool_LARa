# ICPR20:From Human Pose to On Body Devices for Human Activity Recognition
Implementation code for "From Human Pose to On Body Devices for Human Activity Recognition.

Updating in progress..

## Dataset
Pamap2 preprocessing found in CNN_IMU_rep/src/preprocessing_pamap2.py
 Dataset [1][2] can be downloaded from:
 http://archive.ics.uci.edu/ml/datasets/pamap2+physical+activity+monitoring

## Usage

### Preprocessing of:
 - Pamap2
 - Opportunity-Locomotion
 - Opportunity-Gestures

Following the create_dataset() method in `*preprocessing.py` files

### Dataloader using a simple CSV file:
Please check, `HARWindows.py`


### Experiments
Training/Testing, please address to the `main.py` script.
Experiments can set set according to a given dataset, network and output

## Contact

  - Fernando Moya Rueda   fernando.moya@tu-dortmund.de
  
Technische University of Dortmund
Department of Computer Science
Dortmund, Germany
  
  
The work on this publication was supported by Deutsche Forschungsgemeinschaft (DFG) in the context of the project Fi799/10-2, HO2403/14-2 ''Transfer Learning for Human Activity Recognition in Logistics''.


[1] A. Reiss and D. Stricker. Introducing a New Benchmarked Dataset for Activity Monitoring. The 16th IEEE International Symposium on Wearable Computers (ISWC), 2012.

[2] A. Reiss and D. Stricker. Creating and Benchmarking a New Dataset for Physical Activity Monitoring. The 5th Workshop on Affect and Behaviour Related Assistance (ABRA), 2012.