# Problems and Solutions (text classification approach)
In this repsitery, I am gonna share with you some of the problems, which faced and my proposed solutions. It is worth mentioning that the codes are run on "Large Movie Review Dataset 1.0" data set, which is available freely on https://www.kaggle.com/datasets/macespinoza/large-movie-review-dataset-10. 
It would be my pleasure if you share your experinces about these problems and your ideas about my solutions.
## First problem: bias impact on LSTM and GRU performance
Based on "[An Empirical Exploration of Recurrent Network Architectures.pdf](https://github.com/mohammadmehdikeramati/Text-Classification/files/9563647/An.Empirical.Exploration.of.Recurrent.Network.Architectures.pdf)" paper, I came up with the amazing impact of adding bias in improving convergence of LSTM and GRU. I tested this issue on a text classification code and uoloaded its script as "Add Bias to Network".
## Second problem: add vectorization layer on LSTM and GRU performance
I was curios about the performance of a rnn architeucture for text classification purpose, when using vectorization layer instead of implementing vectorization out of network (it used in "Add Bias to Network" script). My ivestigation's results shows there is almost no difference between these tow approaches. I tested this issue on a text classification code and uoloaded its script as "Add Vectorization Layer ".    
## Third problem: using one hot encoding instead of embeding layer
In this investigation, I tried to use one hot encoding instead of embeding layer to see its effect on performance of our network. Even though I succeded to encode the integer output of vectorization layer, I could not feed it to N.N architecture. Indeed, I faced this error `rgb(9, 105, 218)`"e" 


