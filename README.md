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
The input for this model has a final shape determined by the number of marker positions, velocities, acclerations, and leg step by number of time steps, by number of steps.  The output has a final shape of 1 (peak KAM value) by the number of steps.

The data used to generate this model was collected from 3D motion capture data during gait from 98 people, giving a total of 128416 steps.  

The input is the 3D position time series (30 steps) from 13 anatomical landmarks (Toe (R/L), Heel (R/L), Ankle(R/L), Knee (R/L), Front Pelvis (R/L), Back Pelvis (R/L), Neck) per step along with scalar anthropometric data. There are three additional columns: the leg step as a binary (1 for right or 0 for left), a height constant, and a weight constant, all propagated along the time series, giving the data a shape of (42, 8, 128416). After the data is loaded into the workspace, the height and weight columns are removed from the matrix and used to normalize the marker positions by height and knee adduction moment by weight.  Velocities and accelerations are calculated from the positions along the time series and added to the matrix, which gives a final input shape of (118, 8, 128416). The knee adduction moment data file is one peak value per step for a shape of (1,128416). The subject index data file is the subject number per step for a shape of (1,128416) and is used to split up the data by person.

## Guidelines for Use
There are three models available for this task which can all be found in `Models`:
1. Fully Connected Network
2. LSTM
3. CNN

We found the fully connected network to perform the best of the three models.  Each script performs the following:
1. Imports the data
2. Augments the input to include velocities and accelerations and to 8 time steps
3. Resizes the input and output matrices to the correct format
4. Divides into Test, Dev, Train sets
5. Flatten Input and Output Data (Fully connected only)
6. Bulids the model
7. Runs the model
8. Evaluates the performance with r^2 and RMSE values

Depending on the data being used, you may want to adjust and tune the model hyperparameters.

## Resources
More information on data collection and preprocessing:
- [Subject-specific toe-in or toe-out gait modifications reduce the larger knee adduction moment peak more than a non-personalized approach](https://www.ncbi.nlm.nih.gov/pubmed/29174534)
