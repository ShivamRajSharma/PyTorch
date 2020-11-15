# IMDB MOVIE REVIEWS SENTIMENT ANALYSIS 

<p align="center">
  <img src="https://mk0ecommercefas531pc.kinstacdn.com/wp-content/uploads/2019/12/sentiment-analysis.png" height="280" />
</p>

## Introduction

Sentiment Analysis is the process of determining whether a piece of writing is positive, negative or neutral. A sentiment analysis system for text analysis combines natural language processing (NLP) and machine learning techniques to assign weighted sentiment scores to the entities, topics, themes and categories within a sentence or phrase.

Sentiment analysis helps data analysts within large enterprises gauge public opinion, conduct nuanced market research, monitor brand and product reputation, and understand customer experiences. In addition, data analytics companies often integrate third-party sentiment analysis APIs into their own customer experience management, social media monitoring, or workforce analytics platform, in order to deliver useful insights to their own customers.

## Output

## Dataset Information 

The model was trained on IMDB movie reviews from kaggle. The dataset contains 50K movie reviews with 25k positive and 25k negetive movie reviews. </br>
Dataset : https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

## Usage 

1) Install the required libraries mentioned in requirement.txt.
2) Download the dataset from url provided above and place it inside ``` input/ ``` folder.
3) Run ```python3 train.py``` and let the model train for 2-3 iterations.
4) To infer on the trained model run ```python3 predict.py```.


## Model Architecure 
<p align="center">
  <img src="https://www.vproexpert.com/wp-content/uploads/2019/12/google-bert-745x342-1.png" height="280" />
</p>

Bidirectional Encoder Representations from Transformers (BERT) is a Transformer-based machine learning technique for natural language processing (NLP) pre-training developed by Google. BERT is a “deeply bidirectional” model meaning that BERT learns information from both the left and the right side of a token’s context during the training phase. The bidirectionality of a model is important for truly understanding the meaning of a language. Let’s see an example to illustrate this. There are two sentences in this example and both of them involve the word “bank”:
BERT is pre-trained on a large corpus of unlabelled text including the entire Wikipedia(that’s 2,500 million words!) and Book Corpus (800 million words). </br>

Paper : https://arxiv.org/abs/1706.03762


## Extra Info
<pre>
1) Trainin Stratergy       : The whole network was fine tuned on the dataset.
2) Optimizer               : Adam optimizer was used with weight decay.
3) Learning Rate Scheduler : Linear decay scheduler.
4) Loss                    : Binary Cross-Entropy Loss.
5) Performance Metric      : Accuracy.
6) Epochs Trained          : 2
6) Performance             : 95% Accuracy
7) Training Time           : 52 minutes
</pre>

