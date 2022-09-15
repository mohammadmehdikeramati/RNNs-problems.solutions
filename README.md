# Problems and Solutions (text classification approach)
In this repository, I will share with you some of the problems, I faced and my proposed solutions. It is worth mentioning that the codes are run on "Large Movie Review Dataset 1.0" data set, which is available freely on https://www.kaggle.com/datasets/macespinoza/large-movie-review-dataset-10. 
It will be my pleasure if you share your experinces with these problems and your ideas about my solutions.
## First problem: bias impact on LSTM and GRU performance
Based on "[An Empirical Exploration of Recurrent Network Architectures.pdf](https://github.com/mohammadmehdikeramati/Text-Classification/files/9563647/An.Empirical.Exploration.of.Recurrent.Network.Architectures.pdf)" paper, I came up with the fantastic impact of adding bias in improving convergence of LSTM and GRU. I tested this issue on a text classification code and uploaded its script as "Add Bias to Network".
## Second problem: add vectorization layer on LSTM and GRU performance
I was curious about the performance of an rnn architecture for text classification purposes when using a vectorization layer instead of implementing vectorization out of the network (it was used in "Add Bias to Network" script). My investigation's results show almost no difference between these two approaches. I tested this issue on a text classification code and uploaded its script as "Add Vectorization Layer ". 

## Third problem: using one hot encoding instead of embeding layer
In this investigation, I tried to use one hot encoding instead of an embeding layer to see its effect on the performance of our network. Even though I succeeded in encoding the integer output of the vectorization layer, I could not feed it to N.N architecture. Indeed, I faced this error:
```diff
-Exception encountered when calling layer "gru" (type GRU). Input 'b' of 'MatMul' Op has type float32 that does not match type int32 of argument 'a'. 
```
The script of this part is uploaded as "One Hot Encode".
## Fourth problem: a text classification project form the scratch
In this section, I made my real effort to do a text classification project form the scratch. Firstly, the imported data (it is accessable via https://www.kaggle.com/code/bindur/amazon-baby-sentiment/notebook) is preprocessed. In this stage respectively, the problem is turned into a binary classification, the rows contained "Nan" values are removed, the data is split to train (75%) and test (25%) and it is vectorized. Seconly, for classification purpose a three layer architecture consisted of an embeded , a LSTM and a fully connected layer is designed. Finally, the predicted data is compared with ground truth using classification_report. The script of this part is uploaded as "Project".
