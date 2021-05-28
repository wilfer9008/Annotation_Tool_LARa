# ICPR20:From Human Pose to On Body Devices for Human Activity Recognition
Implementation code for "From Human Pose to On Body Devices for Human Activity Recognition.

Updating in progress..

## Dataset

LARa dataset can be downloaded in https://zenodo.org/record/3862782#.XtVJOy9h3UI

## Usage

### Preprocessing of:
 - LARa Mocap
 - LARa IMUs (called mbientlab)
 - LARa virtual IMUs

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
