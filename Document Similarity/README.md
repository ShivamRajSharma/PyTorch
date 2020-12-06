# Document Similarity

<p align="center">
  <img src="https://res.cloudinary.com/match2lists/image/upload/v1497274659/Match_600_gcdvaf.png" height="220" />
</p>

Document similarity, as the name suggests determines how similar are the two given documents. By “documents”, we mean a collection of strings. Measuring pairwise document similarity is an essential operation in various text mining tasks. </br> 

Most of the similarity measures judge the similarity between two documents by converting the text of each respective document as a vector, consisting of continous values that can be compared the other document vector. With the recent development of the deep learning techniques like RNN, LSTM have leveraged the preformance of mesuring the similarity between two documents by measuring the distance between vectors of respective document, resulting in a more accuracte document's extraction process that can be applied in a search engine.


## Dataset Information 
The model was trained on a subset of Quora Question Pairs from kaggle.  </br>
Dataset : https://www.kaggle.com/c/quora-question-pairs/data?select=train.csv.zip

## Usage 

1) Install the required libraries mentioned in requirement.txt.
2) Download the dataset from url provided above and place it inside ``` input/ ``` folder.
3) Run ```python3 train.py``` and let the model train for 2-3 iterations.
4) To infer on the trained model run ```python3 predict.py```.

## Results:
  <img src="https://github.com/ShivamRajSharma/PyTorch/blob/master/Document%20Similarity/output/output.png" />


## Model Architecure 

<p align="center">
  <img src="https://www.researchgate.net/profile/Tuan_Lai4/publication/336443055/figure/fig3/AS:812860531818507@1570812461956/QA-LSTM-with-attention-figure-adapted-from-Tan-et-al-2015.png" height="280" />
</p>

A Siamese LSTM based network was used with an "ALL vs ALL" loss instead of tripet loss in which all examples in a batch are compared with the rest of the examples in a batch based on the cosine similarity of a document vector. Hard negetive mining was used to find the difference between hard negetive question pairs (mean negetive and closest negtive mining). 


## Extra Info
<pre>
1) Trainin Stratergy       : Training the whole network from scratch.
2) Optimizer               : Adam optimizer was used with weight decay.
3) Learning Rate Scheduler : Exponential decay at Plateau.
4) Regularization          : Dropout, 
5) Loss                    : All vs All loss.
6) Performance Metric      : AUC Score.
7) Performance             : 0.68 AUC Score
7) Epochs Trained          : 5
8) Training Time           : 1 hour
</pre>

## Further Improvement 
1) Using Multiheaded attetion.
2) Using Pre-trained Encoder based transformer architecture.
3) Using Universal Sentence Encoder.
