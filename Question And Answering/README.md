# Question And Answering System

An End to End implementation of a Question Answering System using RoBERTa model on Tweet-Sentiment Extraction Dataset as context. As input a comprehession and a question are passed to the network and the network tries to find the answer to the question within the comprehession and extract it.

## Dataset Information
The model was trained on Tweet-Sentiment Extraction from kaggle.</br>
Dataset : https://www.kaggle.com/c/tweet-sentiment-extraction

## Usage

1) Install the required libraries mentioned in requirement.txt.
2) Download the dataset from url provided above and place it inside ``` input/ ``` folder.
3) Run ```python3 train.py``` and let the model train for 10 iterations.
4) To infer on the trained model run ```python3 predict.py```.

## Model Architecture 

<p align="center">
  <img src="https://www.vproexpert.com/wp-content/uploads/2019/12/google-bert-745x342-1.png" height="280" />
</p>

Robustly Optimized Bidirectional Encoder Representation Approach from Transformers (RoBERTa) is a Transformer-based machine learning technique for natural language processing (NLP) pre-training developed by Google. RoBERTa is a “deeply bidirectional” model meaning that BERT learns information from both the left and the right side of a token’s context during the training phase. The bidirectionality of a model is important for truly understanding the meaning of a language. </br>

Paper : https://arxiv.org/abs/1907.11692


## Extra Info
<pre>
1) Trainin Stratergy       : The whole network was fine-tuned on the dataset.
2) Optimizer               : Adam optimizer was used with weight decay.
3) Learning Rate Scheduler : Linear decay with warmup scheduler.
4) Loss                    : Binary Cross-Entropy Loss.
5) Regularization          : Weight Decay, Dropout, 
6) Performance Metric      : Jaccard Distance.
7) Epochs Trained          : 3
8) Performance             : 71.3 Jaccard Score.
9) Training Time           : .
</pre>
