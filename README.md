# Using Deep Learning to Perform Emotion Classification of Music

The proposed deep learning architecture for extracting valence and arousal from music is shown below. Given the musically meaningful features, we use dense and convolutional layers, pooling, and dropout to learn the valence and arousal of songs. That is, we approach this as a regression problem. 


![Deep Learning Architecture: Visual Representation](https://github.com/vivekr123/Emotion-Classification-of-Music/blob/master/neural-net-architecture-color.png)

We have two branches of the same network: one predicts valence and the other predicts arousal. The convolutional layers help identify local patterns in the chroma frequencies and Mel-Frequency Cepstral Coefficients. Meanwhile, the dense layers combine and summarize the extracted information from each of the features to predict the final values for valence and arousal.



# 1000 Songs Dataset & Modifications

The preprocessing was performed on the 1000 songs dataset, which provides the original music files and annotations of arousal and valence (results stored in "Training Data").

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
  &nbsp &nbsp &nbsp &nbsp Hardness (related to attack times and associated bandwidth- difference between low and high frequencies)<br>
  &nbsp &nbsp &nbsp &nbsp Depth (weighted sum of fundamental frequency, approximate duration/decay-time, weighted mean of lower frequency ratios)<br>
  &nbsp &nbsp &nbsp &nbsp Brightness (prevalence of upper mid and high frequency content)<br>
  &nbsp &nbsp &nbsp &nbsp Roughness (buzzing,  harsh,  raspy  sound  quality  of  narrow  harmonic  intervals)<br>
  &nbsp &nbsp &nbsp &nbsp Warmth (the tilt towards the bass frequencies)<br>
  &nbsp &nbsp &nbsp &nbsp Sharpness (related  to  the  spectral  balance of frequencies)<br>
  &nbsp &nbsp &nbsp &nbsp Boominess (prevalence of lower frequencies)<br>
  &nbsp &nbsp &nbsp &nbsp Reverberation (persistence of sound after the sound is produced)<br>
  <li>Zero Crossing Rate (scalar): Simply the number of times the signal crosses the x-axis (i.e. changing from negative to positive or vice versa). Computed by (np.where(np.diff(np.sign(data)))[0].size)/duration <br>
  <li>Loudness (scalar): The Python pyloudnorm library provides a couple of concise methods to compute this.<br>
  </ul>

<strong>Schemas</strong>

The variables listed above are conveniently placed in appropriately shaped numpy arrays that are convenient for deep learning models. Simply specify these numpy arrays as inputs.

Tempo<br>
Stored as tempo.npy<br>
Shape: (744,)

Spectral Centroid<br>
<ul>
<li>Stored as spectral-centroid.npy<br>
<li>Shape: (744, 200)
</ul>

Spectral Rolloff<br>
<ul>
<li>Stored as spectral-rolloff.npy<br>
<li>Shape: (744, 200)
  </ul>

Mel-Frequency Cepstral Coefficients<br>
<ul>
<li>Stored as mel-cepstral-coeffs.npy<br>
<li>Shape: (744, 20, 200, 1)
  </ul>

Chroma Frequencies<br>
<ul>
<li>Stored as chroma.npy<br>
<li>Shape: (744, 12, 200, 1)
</ul>

Timbre<br>
<ul>
<li>Stored as timbre.npy<br>
<li>Shape: (744, 10)
</ul>

Static Standardized Valence<br>
<ul>
<li>Stored as static-standardized-valence.npy<br>
<li>Shape: (744,)<br>
<li>Intended to be used as ground truth for training<br>
</ul>

NOTE: These are the annotations for ALL 1000 songs. Look at song ID’s to get appropriate 744 songs
