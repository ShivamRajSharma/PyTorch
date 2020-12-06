# Captcha Reader

A CAPTCHA is a type of challenge–response test used in computing to determine whether or not the user is human.This form of CAPTCHA requires someone to correctly evaluate and enter a sequence of letters or numbers perceptible in a distorted image displayed on their screen. Although CAPTCHA's are really effective against bad bots, with the recent development of the neural networks in the field of computer vision, these CAPTCHAs can be easily decoded and bypassed requiring a more stronger method to classify a bot and a human.

In this repository we build a CNN-RNN based encoder decoder based network for reading CAPTCHA's.


## Dataset Information

The dataset contains 1040 captcha files as png images. The label for each sample is a string, the name of the file (minus the file extension).

```
!curl -LO https://github.com/AakashKumarNain/CaptchaCracker/raw/master/captcha_images_v2.zip
!unzip -qq captcha_images_v2.zip
```

## Usage

1) Install the required libraries mentioned in requirement.txt.
2) Download the dataset from url provided above and place it inside ``` input/ ``` folder.
3) Run ```python3 train.py``` and let the model train for 10 iterations.
4) To infer on the trained model run ```python3 predict.py```.

## Result:
<img src="https://github.com/ShivamRajSharma/PyTorch/blob/master/Captcha%20Reader/Output/output.png"/>


## Model Architecture 

<p align="center">
  <img src="https://media.springernature.com/original/springer-static/image/chp%3A10.1007%2F978-3-030-31756-0_5/MediaObjects/480626_1_En_5_Fig1_HTML.png" height="200"/>
</p>

Model architeture is an Enoder-Decoder architecture with 
1) Encoder - A simple CNN based encoder
2) Decoder - A simple GRU based decoder


## Extra Info
<pre>
1) Training Stratergy      : Training the whole network from scratch.
2) Optimizer               : Adam optimizer was used with weight decay.
3) Learning Rate Scheduler : Exponential decay at Plateau.
4) Regularization          : Dropout, 
5) Loss                    : CTC Loss.
6) Performance Metric      : CTC Loss.
7) Epochs Trained          : 10-20.
8) Training Time           : 2 minutes.
</pre>

## Further Improvement
1) Larger Dataset
2) More Regularization
3) Beam Search Decoding
4) Better Encoder Architecture 
5) Attention Based Decoder Architecture
