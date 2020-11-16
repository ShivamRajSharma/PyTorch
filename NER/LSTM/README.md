# Named Entity Recognition

<p align="center">
  <img src="https://cdn-images-1.medium.com/max/1000/1*EtZzTLreinuaZ9TtfuXhAw.png" height="150" />
</p>

Named-entity recognition (NER) (also known as (named) entity identification, entity chunking, and entity extraction) 
is a subtask of information extraction that seeks to locate and classify named entities mentioned in unstructured text into pre-defined 
categories such as person names, organizations, locations, medical codes, time expressions, quantities, monetary values, percentages, etc. </br>

In this repository we train a simple GRU based recurrent nueral network for Part-Of-Speech application from scratch, achieving good results even with a small network.

## Dataset Information 

The model was trained on NER dataset from kaggle. The dataset contains a total of 47k sentences with a total of 41 POS tags  . </br>
Dataset : https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus

## Usage 

1) Install the required libraries mentioned in requirement.txt.
2) Download the dataset from url provided above and place it inside ``` input/ ``` folder.
3) Run ```python3 train.py``` and let the model train for 2-3 iterations.
4) To infer on the trained model run ```python3 predict.py```.


## Model Architecure 
<p align="center">
  <img src="https://info.itemis.com/hubfs/Blog/DataScience/RNN-based-on-GRU-cells.jpg" height="350" />
</p>

A Unidirectional based simple GRU network with word to POS mapping.

## Extra Info
<pre>
1) Trainin Stratergy       : The whole network was trained from scratch.
2) Optimizer               : Adam optimizer.
4) Loss                    : Categorical Cross-Entropy Loss.
5) Regularization          : Dropout
6) Performance Metric      : Accuracy.
7) Epochs Trained          : 1
8) Performance             : 97.1% Accuracy.
9) Training Time           : 1 minute.
</pre>

## Further Improvement 
1) Larger Dataset.
2) CRF at the final layer.
3) Attention based network.
