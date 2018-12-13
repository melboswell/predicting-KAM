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

The input is the 3D position time series (8 steps) from 13 anatomical landmarks (Toe (R/L), Heel (R/L), Ankle(R/L), Knee (R/L), Front Pelvis (R/L), Back Pelvis (R/L), Neck) per step along with scalar anthropometric data. We added an input column for whether the leg stepping was right or left as a binary (1 for right or 0 for left) propagated along the time series, giving the data a shape of (42, 8, 128416). After we load the data into our workspace, we remove the height and weight columns from the matrix and normalize the marker positions by height and knee adduction moment by weight.  We calculate velocities and accelerations from the positions along the time series, which gives a final input shape of (118, 8, 128416). The knee adduction moment data file is one peak value per step for a shape of (1,128416). The subject index data file is the subject number per step for a shape of (1,128416) and we use this to split up the data by person.
Once the download is complete, move the dataset into `model`
*Do we want to make a build_dataset.py to reshape the data?*

## Guidelines for Use

## Resources
More information on data collection and preprocessing:
- [Subject-specific toe-in or toe-out gait modifications reduce the larger knee adduction moment peak more than a non-personalized approach](https://www.ncbi.nlm.nih.gov/pubmed/29174534)
