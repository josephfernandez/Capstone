# Generate datasets from noise,chimp and bird directories

import os
import numpy as np
from capstone import extractFeatures

#these file paths are specifc to computer being used
tr_chimp_files_path = r'c:\users\joe\desktop\capstone_audio\sound_bytes\chimp\training'
tr_noise_files_path = r'c:\users\joe\desktop\capstone_audio\sound_bytes\noise\training'
ts_chimp_files_path = r'c:\users\joe\desktop\capstone_audio\sound_bytes\chimp\test'
ts_noise_files_path = r'c:\users\joe\desktop\capstone_audio\sound_bytes\noise\test'

#initialize empty feature and label vectors
tr_features, tr_labels = np.empty((0,193)), np.empty(0)
ts_features, ts_labels = np.empty((0,193)), np.empty(0)


i = -1
#this loop extracts training data from chimp files
for entry in os.scandir(tr_chimp_files_path):
    i+=1
    index = str.find(entry.path,'chimp')
    file_name_with_extension = entry.path[index+6:]
    file_name = file_name_with_extension[:-5]
    extractFeatures.extractFeatures(entry.path,file_name,'chimp')
    tr_features = np.vstack([tr_features,extractFeatures.extractFeatures(entry.path,file_name,'chimp')])
    tr_labels = np.append(tr_labels,1)

#this loop extracts training data from other files
for entry in os.scandir(tr_noise_files_path):
    i+=1
    index = str.find(entry.path,'noise')
    file_name_with_extension = entry.path[index+6:]
    file_name = file_name_with_extension[:-5]

    tr_features = np.vstack([tr_features,extractFeatures.extractFeatures(entry.path,file_name,'noise')])
    tr_labels = np.append(tr_labels,0)
    
i = -1
#this loop extracts test data from chimp files
for entry in os.scandir(ts_chimp_files_path):
    i+=1
    index = str.find(entry.path,'chimp')
    file_name_with_extension = entry.path[index+6:]
    file_name = file_name_with_extension[:-5]
    extractFeatures.extractFeatures(entry.path,file_name,'chimp')
    ts_features = np.vstack([ts_features,extractFeatures.extractFeatures(entry.path,file_name,'chimp')])
    ts_labels = np.append(ts_labels,1)
    
#this loop extracts test data from other files
for entry in os.scandir(ts_noise_files_path):
    i+=1
    index = str.find(entry.path,'noise')
    file_name_with_extension = entry.path[index+6:]
    file_name = file_name_with_extension[:-5]

    ts_features = np.vstack([ts_features,extractFeatures.extractFeatures(entry.path,file_name,'noise')])
    ts_labels = np.append(ts_labels,0)

tr_features = np.array(tr_features)
tr_labels = np.array(tr_labels, dtype=np.int)
ts_features = np.array(ts_features)
ts_labels = np.array(ts_labels, dtype=np.int)

#files are saved for later use also to avoid re-extraction
np.save(r'c:\users\joe\desktop\capstone_audio\tr_features', tr_features)    
np.save(r'c:\users\joe\desktop\capstone_audio\tr_labels', tr_labels)    
np.save(r'c:\users\joe\desktop\capstone_audio\ts_features', ts_features)    
np.save(r'c:\users\joe\desktop\capstone_audio\ts_labels', ts_labels)