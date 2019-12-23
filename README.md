# Emotion-Classification-of-Music

The proposed deep learning architecture for extracting valence and arousal from music is shown above. Given the musically meaningful features, we simply needed to provide dense and convolutional layers, pooling, and dropout for the network to learn these values. That is, we approach this as a regression problem. 


![Deep Learning Architecture: Visual Representation](https://github.com/vivekr123/Emotion-Classification-of-Music/blob/master/neural-net-architecture-color.png)

We have two branches of the same network: one predicts valence and the other predicts arousal. The convolutional layers help identify local patterns in the chroma frequencies and Mel-Frequency Cepstral Coefficients. Meanwhile, the dense layers combine and summarize the extracted information from each of the features to predict the final values for valence and arousal.



# 1000 Songs Dataset & Modifications

The preprocessing was performed on the 1000 songs dataset (stored in "Training Data"), which provides the original music files and annotations of arousal and valence (both dynamic and static).

The dataset contains the following:

  ./Annotations<br>
  Dynamic and static annotations in csv format of valence and arousal for 744 songs (song numbers provided)
  
  ./Extracted-Features<br>
  <ul>
  <li>Tempo (scalar): the speed at which a passage of music is or should be played (librosa.beat.tempo)<br>
  <li>Spectral Centroid (vector): Indicates where the ”centre of mass” for a sound is located and is calculated as the weighted mean of the frequencies present in the sound. (librosa.feature.spectral_centroid)<br>
  <li>Spectral Rolloff (vector): measure of the shape of the signal. It represents the frequency below which a specified percentage of the total spectral energy. (librosa.feature.spectral_rolloff)<br>
  <li>Mel-Frequency Cepstral Coefficients (image): Concisely describe the overall shape of a spectral envelope. It models the characteristics of the human voice. (librosa.feature.mfcc)<br>
  <li>Chroma Frequencies (image):  the entire spectrum is projected onto 12 bins representing the 12 distinct semitones (or chroma) of the musical octave. (librosa.feature.chroma_stft)<br>
  <li>Timbre (vector): (first 8 computed with AudioCommons’ timbral_models)<br>
  &emsp Hardness (related to the attack times and the associated bandwidth - difference between low and high frequencies)<br>
  &emsp Depth (a weighted sum of fundamental frequency, approximate duration/decay-time, weighted mean of lower frequency ratios)<br>
  &emsp Brightness (prevalence of upper mid and high frequency content)<br>
  &emsp Roughness (buzzing,  harsh,  raspy  sound  quality  of  narrow  harmonic  intervals)<br>
  &emsp Warmth (the tilt towards the bass frequencies)<br>
  &emsp Sharpness (related  to  the  spectral  balance of frequencies)<br>
  &emsp Boominess (prevalence of lower frequencies)<br>
  &emsp Reverberation (persistence of sound after the sound is produced)<br>
  <li>Zero Crossing Rate (scalar): Simply the number of times the signal crosses the x-axis (i.e. changing from negative to positive or vice versa). Computed by (np.where(np.diff(np.sign(data)))[0].size)/duration <br>
  <li>Loudness (scalar): The Python pyloudnorm library provides a couple of concise methods to compute this.<br>
  </ul>

Schemas

The variables listed above are conveniently placed in appropriately shaped numpy arrays that are convenient for deep learning models. Simply specify these numpy arrays as inputs.

Tempo
Stored as tempo.npy
Shape: (744,)

Spectral Centroid
Stored as spectral-centroid.npy
Shape: (744, 200)

Spectral Rolloff
Stored as spectral-rolloff.npy
Shape: (744, 200)

Mel-Frequency Cepstral Coefficients
Stored as mel-cepstral-coeffs.npy
Shape: (744, 20, 200, 1)

Chroma Frequencies
Stored as chroma.npy
Shape: (744, 12, 200, 1)

Timbre
Stored as timbre.npy
Shape: (744, 10)

Static Standardized Valence
Stored as static-standardized-valence.npy
Shape: (744,)
Intended to be used as ground truth for training

NOTE: These are the annotations for ALL 1000 songs. Look at song ID’s to get appropriate 744 songs
