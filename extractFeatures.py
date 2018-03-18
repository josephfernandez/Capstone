import sys
import numpy as np
import librosa

#append path name for importing librosa module
sys.path.append(r'c:\users\joe\anaconda2\lib\site-packages')

def extractFeatures(audio_path,file_name,classname):

  # load file
  y, sr = librosa.load(audio_path)

  # Extracts Mel Spectogram
  S = np.mean(librosa.feature.melspectrogram(y, sr=sr, n_mels=128),axis=1)

  # Extracts Mel-Frequency Cepstrum Coefficients
  mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40),axis=1)

  # Extracts STFT-Chromagram
  chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr),axis=1)

  # Extracts octave-based spectral contrast
  contrast = np.mean(librosa.feature.spectral_contrast(S=np.abs(librosa.stft(y)), sr=sr),axis=1)

  # Extracts tonal-centroid
  tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y),sr=sr),axis=1)

  ext_features = np.hstack([mfccs,chroma,S,contrast,tonnetz])

  return ext_features