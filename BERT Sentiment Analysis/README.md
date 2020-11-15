# IMDB MOVIE REVIEWS SENTIMENT ANALYSIS 

## Introduction

Sentiment Analysis is the process of determining whether a piece of writing is positive, negative or neutral. A sentiment analysis system for text analysis combines natural language processing (NLP) and machine learning techniques to assign weighted sentiment scores to the entities, topics, themes and categories within a sentence or phrase.

Sentiment analysis helps data analysts within large enterprises gauge public opinion, conduct nuanced market research, monitor brand and product reputation, and understand customer experiences. In addition, data analytics companies often integrate third-party sentiment analysis APIs into their own customer experience management, social media monitoring, or workforce analytics platform, in order to deliver useful insights to their own customers.

## Dataset Information 

The model was trained on IMDB movie reviews from kaggle. The dataset contains 50K movie reviews with 25k positive and 25k negetive movie reviews. </br>
Dataset : https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

## Usage 

1) Install the required libraries in requirement.txt
2) Download the dataset from url provided above and place it inside ``` input/ ``` folder.
3) Run ```python3 train.py``` and let the model train for 2-3 iterations.
4) To infer on the trained model run ```python3 predict.py```.


## Model Architecure 

I used a pretrained BERT architecture, and fine tuned it on 

