# Image Segmentation

Image segmentation is the process of partitioning an image into multiple segments (sets of pixels, also known as image objects). The goal of segmentation is to simplify and/or change the representation of an image into something that is more meaningful and easier to analyze. </br>
Here we use a CNN based neural network is used with a combination of Dice Loss and Binary Cross-Entropy loss to partition an image. Instead of just using Binary Cross-Entropy loss we use Dice Loss with Binary Cross-Entropy which results in more fine edges.


## Dataset Information

The model was trained on 566 X-ray images of lungs from kaggle.</br>
Dataset : https://www.kaggle.com/ianmoone0617/chest-xray-with-masks-for-image-segmentation

## Usage

1) Install the required libraries mentioned in requirement.txt.
2) Download the dataset from url provided above and place it inside ``` input/ ``` folder.
3) Run ```python3 train.py``` and let the model train for 10 iterations.
4) To infer on the trained model run ```python3 predict.py```.

## Model Architecture 

<p align="center">
  <img src="https://miro.medium.com/max/1620/1*eKrh8FqJL3jodebYlielNg.png" height="400"/>
</p>

Model architecture was U-Net with ResNet for the construction of the mask from the image for segmentation.</br>
Paper : https://arxiv.org/abs/1505.04597

## Extra Info
<pre>
1) Training Stratergy      : Training the whole network from scratch.
2) Optimizer               : Adam optimizer was used with weight decay.
3) Learning Rate Scheduler : Exponential decay at Plateau.
4) Regularization          : Dropout 
5) Loss                    : Dice Loss + Binary Cross-Entropy Loss
6) Performance Metric      : Dice Loss + Binary Cross-Entropy Loss.
7) Epochs Trained          : .
8) Training Time           : 
</pre>

## Further Improvement
1) Larger Dataset
