# Predicting Knee Adduction Moment Using 3D Motion Capture Marker Trajectories
*Authors: Melissa Boswell and Scott Uhlrich*
## Requirements
We recommend using python5.  This code uses the keras package.
```
pip install -r requirements.txt
```
## Task
Given only 3D motion capture marker positions, predict the knee adduction moment.
## The Dataset
We have an example data set available on google drive, download it [here](https://www.ourdatalinkhere)

The data was collected from 3D motion capture data during gait from 98 people, giving a total of 125415 steps.  

The input is the 3D position time series (30 steps) from 13 anatomical landmarks (Toe (R/L), Heel (R/L), Ankle(R/L), Knee (R/L), Front Pelvis (R/L), Back Pelvis (R/L), Neck) per step along with scalar anthropometric data. There are three additional columns: the leg step as a binary (1 for right or 0 for left), a height constant, and a weight constant, all propagated along the time series, giving the data a shape of (42, 8, 128416). After the data is loaded into the workspace, the height and weight columns are removed from the matrix and used to normalize the marker positions by height and knee adduction moment by weight.  Velocities and accelerations are calculated from the positions along the time series and added to the matrix, which gives a final input shape of (118, 8, 128416). The knee adduction moment data file is one peak value per step for a shape of (1,128416). The subject index data file is the subject number per step for a shape of (1,128416) and is used to split up the data by person.

## Guidelines for Use

## Resources
More information on data collection and preprocessing:
- [Subject-specific toe-in or toe-out gait modifications reduce the larger knee adduction moment peak more than a non-personalized approach](https://www.ncbi.nlm.nih.gov/pubmed/29174534)
