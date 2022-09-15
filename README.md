# Problems and Solutions (text classification approach)
In this repository, I will share with you some of the problems, I faced and my proposed solutions. It is worth mentioning that the codes are run on "Large Movie Review Dataset 1.0" data set, which is available freely on https://www.kaggle.com/datasets/macespinoza/large-movie-review-dataset-10. 
It will be my pleasure if you share your experinces with these problems and your ideas about my solutions.
## First problem: bias impact on LSTM and GRU performance
Based on "[An Empirical Exploration of Recurrent Network Architectures.pdf](https://github.com/mohammadmehdikeramati/Text-Classification/files/9563647/An.Empirical.Exploration.of.Recurrent.Network.Architectures.pdf)" paper, I came up with the fantastic impact of adding bias in improving convergence of LSTM and GRU. I tested this issue on a text classification code and uploaded its script as "Add Bias to Network".
## Second problem: add vectorization layer on LSTM and GRU performance
I was curious about the performance of an rnn architecture for text classification purposes when using a vectorization layer instead of implementing vectorization out of the network (it was used in "Add Bias to Network" script). My investigation's results show almost no difference between these two approaches. I tested this issue on a text classification code and uploaded its script as "Add Vectorization Layer ". 


![pic1](https://user-images.githubusercontent.com/42337253/190309085-5230788d-30db-488f-8317-553b1168db20.PNG)
## Third problem: using one hot encoding instead of embeding layer
In this investigation, I tried to use one hot encoding instead of an embeding layer to see its effect on the performance of our network. Even though I succeeded in encoding the integer output of the vectorization layer, I could not feed it to N.N architecture.

![pic3](https://user-images.githubusercontent.com/42337253/190310122-6068f3cd-5829-4385-86c5-42a981ddb8ee.PNG)

![pic2](https://user-images.githubusercontent.com/42337253/190309604-1ab96a4b-a418-40e8-a9c0-cc7fda3bcc47.PNG)


Indeed, I faced this error:
```diff
-Exception encountered when calling layer "gru" (type GRU). Input 'b' of 'MatMul' Op has type float32 that does not match type int32 of argument 'a'. 
```
The script of this part is uploaded as "One Hot Encode".
## Fourth problem: a text classification project from scratch
In this section, I made my real effort to do a text classification project from scratch. Firstly, the imported data (it is accessible via https://www.kaggle.com/code/bindur/amazon-baby-sentiment/notebook) is preprocessed. In this stage, respectively, the problem is turned into a binary classification, the rows containing "Nan" values are removed, the data is split to train (75%) and test (25%) and, it is vectorized. Secondly, for classification purpose a three- layer architecture consisting of an embeded , an LSTM and, a fully connected layer is designed. Finally, the predicted data is compared with ground truth using classification_report. The script of this part is uploaded as "Project".

![pic4](https://user-images.githubusercontent.com/42337253/190312966-83e2102c-c68b-4059-a9f3-e8f24b6628a9.PNG)

