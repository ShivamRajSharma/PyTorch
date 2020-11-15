# Document Summarization 

</br>
<p align="center">
  <img src="https://images.squarespace-cdn.com/content/55ff6aece4b0ad2d251b3fee/1522758193315-MXEB72GR74GA8T47LOHF/tumblr_inline_och5k95kSe1ta78fg_540.png?content-type=image%2Fpng" height="200"/>
</p> 

Automatic Document Summarization is the task of rewriting a document into its shorter form while still retaining its important content. The most popular two paradigms are extractive approaches and abstractive approaches. Extractive approaches generate summaries by extracting parts of the original document (usually sentences), while abstractive methods may generate new words or phrases which are not in the original document.


## Dataset Information

The dataset was downloaded from kaggle. It consists of 98401 examples and contains  Short text, Complete Article. </br>
Dataset : https://www.kaggle.com/sunnysai12345/news-summary?select=news_summary_more.csv

## Usage

1) Install the required libraries mentioned in requirement.txt.
2) Download the dataset from url provided above and place it inside ``` input/ ``` folder.
3) Run ```python3 train.py``` and let the model train for 10 iterations.
4) To infer on the trained model run ```python3 predict.py```.


## Model Architecture 

<p align="center">
  <img src="https://jalammar.github.io/images/t/transformer_resideual_layer_norm_3.png" height="300"/>
</p>

A Multi-headed self attention based transformer achetecture was used. A transformer is as encoder based archtecture used for machine translation, document summarization etc.</br>
Paper : https://arxiv.org/abs/1706.03762


## Extra Info
<pre>
1) Training Stratergy      : Training the whole network from scratch.
2) Optimizer               : Adam optimizer was used with weight decay.
3) Learning Rate Scheduler : Linear decay with warmup.
4) Regularization          : Dropout, Weight decay, 
5) Loss                    : Categorical Cross-Entropy.
6) Performance Metric      : .
7) Epochs Trained          : 25.
8) Training Time           : 5 Hours.
9) Decoding                : Greedy
</pre>
