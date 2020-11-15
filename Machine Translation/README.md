# Neural Machine Translation 

<p align="center">
  <img src="http://jalammar.github.io/images/t/The_transformer_encoders_decoders.png" height="300"/>
</p>

Neural machine translation (NMT) is an approach to machine translation that uses an artificial neural network to predict the likelihood of a sequence of words, typically modeling entire sentences in a single integrated model. An Encoder-Decoder based architecture is used to to to translate a sentence from one language to other. 
Here we translate German to English using an LSTM based network with a Teacher-Force ratio of 0.5 to further improve our model performance.</br>


## Dataset Information

The model was trained on a subset of WMT-2014 English-German Dataset. Preprocessing was carried out before training the model.</br>
Dataset :  https://nlp.stanford.edu/projects/nmt/</br>


## Usage

1) Install the required libraries mentioned in requirement.txt.
2) Download the dataset from url provided above and place it inside ``` input/ ``` folder.
3) Run ```python3 train.py``` and let the model train for 10 iterations.
4) To infer on the trained model run ```python3 predict.py```.</br>


## Model Architecture 

<p align="center">
  <img src="https://miro.medium.com/max/4000/0*UldQZaGB0w3omI3Y.png" height="300"/>
</p>

An Bi-Directional based Encoder-Decoder architecture with a default Teacher-Force ratio of 0.5. </br>


## Extra Info
<pre>
1) Training Stratergy      : Training the whole network from scratch.
2) Optimizer               : Adam optimizer was used with weight decay.
3) Learning Rate Scheduler : Exponential decay at Plateau.
4) Regularization          : Dropout, 
5) Loss                    : Categorical Cross-Entropy.
6) Performance Metric      : BLEU SCORE.
7) Performance             : 19.75
7) Epochs Trained          : .
8) Training Time           : .
9) Decoding                : Greedy</br>
</pre>


## Further Improvement
1) Attention Based architecture (Transformer/ Seq2Seq Attention)
2) Beam Search Decoding
