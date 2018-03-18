### Sound Classification in Python using libROSA and Tensorflow

#### Files Included:
1. __extractFeatures__**.py**: extracts features from audio signal
2. __generateDatasets__**.py**: loops through specified dir's to generate training and test datasets then saves datasets to files for later use
3. __neuralNetwork__**.py**: sets up a NN, loads the saved datasets to use for training, runs a training session then runs a testing session.  Outputs accuracy score, a plot of training epoch vs. cost, a confusion matrix, and final score.

#### Step by Step Instructions:
1. use ffmpeg to cut 4sec. segments from audio files.
  
    - example in MS-DOS:
  
        **ffmpeg -i C:\users\joe\file -ss 00:00:45.0 -t 4 -acodec copy chimp1.flac**

2. set paths in __generateDatasets.__**py** lines 8-11 to directories containing segmented audio files.
3. set paths in __generateDatasets__**.py** lines 66-69 to desired save destination for generated datasets.
4. set paths in __neuralNetwork__**.py** lines 13-16 to files containing saved datasets which were generated in Step 3.
5. Line 40 of __neuralNetwork__**.py** contains code to produce visual representation of dataflow graph using Tensorboard.  Set the event file desitination relative to your machine.  Then in CLI call two commands:            

    **activate Tensorflow**

    **Tensorboard --logdir=directory_containing_event_file**
