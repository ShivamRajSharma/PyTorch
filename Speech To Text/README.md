# Speech To Text Translator 

Text to speech (TTS) is the use of software to create an audio output in the form of a spoken voice. The program that is used by programs to change text on the page to an audio output of the spoken voice is normally a text to speech engine. TTS engines are needed for an audio output of machine translation results. </br>
The audio waves are converted to MFCC's which is a representation of the short-term power spectrum of a sound, based on a linear cosine transform of a log power spectrum on a nonlinear mel scale of frequency. 
The MFCC is passed to an CNN-RNN based encoder decoder architecture with CTC loss to extract text from the audio.


## Dataset Information

The model was trained on LibriSpeech-100 dataset. </br>
Dataset : http://www.openslr.org/12/

## Usage

1) Install the required libraries mentioned in requirement.txt.
2) Download the dataset from url provided above and place it inside ``` input/ ``` folder.
3) Run ```python3 train.py``` and let the model train for 10 iterations.
4) To infer on the trained model run ```python3 predict.py```.

## Model Architecture 

<p align="center">
  <img src="https://media.springernature.com/original/springer-static/image/chp%3A10.1007%2F978-3-030-31756-0_5/MediaObjects/480626_1_En_5_Fig1_HTML.png" height="200"/>
</p>

Model architeture is an Enoder-Decoder architecture with 
1) Encoder - A CNN based encoder
2) Decoder - A LSTM based decoder


## Extra Info
<pre>
1) Training Stratergy      : Training the whole network from scratch.
2) Optimizer               : Adam optimizer was used with weight decay.
3) Learning Rate Scheduler : 
4) Regularization          : Dropout, MFCC's masking,  
5) Loss                    : CTC Loss.
6) Performance Metric      : CTC Loss.
7) Epochs Trained          : .
8) Training Time           : 
</pre>

## Further Improvement
1) Larger Dataset
2) More Regularization
3) Beam Search Decoding
4) Better attention based encoder-decoder architecture
