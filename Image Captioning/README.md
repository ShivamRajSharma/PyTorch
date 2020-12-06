# Image Captioning 

Image Captioning refers to the process of generating textual description from an image – based on the objects and actions in the image. A pre-trained CNN based encoder to encode the image and a RNN decoder was used to decode the encoded image into captions

## Dataset 

The model was trained on Flicker8k dataset.</br>
Dataset : https://www.kaggle.com/ming666/flicker8k-dataset


## Usage

1) Install the required libraries mentioned in requirement.txt.
2) Download the dataset from url provided above and place it inside ``` input/ ``` folder.
3) Run ```python3 train.py``` and let the model train for 10 iterations.
4) To infer on the trained model run ```python3 predict.py```.

## Model Architecture 

<p align="center">
  <img src="https://kharshit.github.io/img/image_captioner_structure.png" height="200"/>
</p>

The model is an Encoder-Decoder based CNN-RNN achtecture with:
1) Pretrained EfficientNet B0 based encoder
2) LSTM based decoder 

## Extra Info
<pre>
1) Training Stratergy      : Only Decoder part was trained.
2) Optimizer               : Adam optimizer was used with weight decay.
3) Learning Rate Scheduler : Exponential decay at Plateau.
4) Regularization          : Dropout, Image Augmentation
5) Loss                    : Categorical Cross-Entropy 
6) Epochs Trained          : 20.
7) Training Time           : 
8) Decoding                : Greedy
</pre>

## Further Improvement
1) Larger Dataset
2) Attention based Decoder 
3) Beam Search decoding
