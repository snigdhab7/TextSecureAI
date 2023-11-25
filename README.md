# CyberBullying Detection using SVM and Naive Bayes 🚫🔍
In the era of social media, the rise of hate speech has become a critical concern. This project focuses on automatically detecting cyberbullying on Twitter using Naive Bayes Classifier and Support Vector Machine (SVM). The project employs sentiment analysis for hate speech detection, comparing the effectiveness of Naive Bayes and SVM in classifying tweets.

## Abstract

With the increase usage, hate speech is currently of current interest in the domain of social media. The anonymity and flexibility afforded by the Internet has made it easy for people to communicate in an abusive and hateful manner. And as the amount of cyber-bullying is increasing, methods that automatically detect hate speech is becoming an urgent need. On Twitter, hateful tweets are the ones that contain abusive speech targeting selected users like (cyber-bullying, a politician, a celebrity, a product) or particular groups (a country, a religion, gender, an organization, etc.). Detecting such hateful comments is important for analysing public sentiment of an individual or group of users towards another set of groups, and for discouraging associated wrongful activities. The manual way of filtering out hateful tweets is not scalable and flexible, motivating researchers to identify automated ways. In this project, we focus on the problem of classifying a tweet as positive or negative text. The task is quite challenging due to the inherent complexity of the natural language constructs – different forms of hatred, different kinds of targets, different ways of representing the same meaning. Most of the earlier work revolves around using representation learning methods followed by a linear classifier. However, recently deep learning methods and SVM have shown accuracy improvements across amount of data. Hence, In order to reduce this problem, we propose a system that can detect cybercrimes from social media automatically using Naive Bayes Classifier and SVM.

### Text Classification Methods

- **Support Vector Machine (SVM):**
  - Utilizes a linear kernel for classifying texts as positive or negative.
  - Effectiveness measures: Precision, Recall, Accuracy, AUC, F-Measure.
    It has been chosen for the classiﬁcation within the experiments. The support-vector machines could even be a learning machine for two-group classiﬁcation problems introduced by . it's wont to classify the texts as positives or negatives. SVM works well for text classiﬁcation due toits advantages like its potential to handle large features.Another advantage is SVM is strong when there's a sparse set of examples and also because most of the matter are linearly separable . Support Vector Machine have shown promising leads to previous research in sentiment analysis
    Effectiveness Measures in SVM-
    
    Four effective measures that are utilized during this study are supported confusion matrix output, which are True Posi-tive (TP), False Positive(FP), True Negative (TN), and FalseNegative(FN)-
    
    - Precision(P) = TP/(TP+FP)
    - Recall(R) = TP/(TP+FN)
    - Accuracy(A) = (TP+TN)/(TP + TN + FP + FN)
    - AUC (Area under the (ROC) Curve) = 1/2.((T-P/(TP+FN))+(TN/(TN+FP))
    - F-Measure(Micro-averaging) = 2.(P.R)/(P+R)
      
    The text categorization effectiveness is typically measured using the F1, accuracy, and AUC . F1 measure could even be a combined effectiveness measure determined by precision and recall. the earth under the ROC curve (AUC) has become an honest measurement of performance of supervised classiﬁcation rules. However, the straightforward quite AUC is just applicable to the case of two classes.
    


- **Naive Bayes' Classifier:**
  - Based on Bayes' theorem with a strong independence assumption.
  - Requires minimal training data, making it efficient for text classification.
  - Evaluation includes accuracy and informative features.
 
      Bayesian network classifiers are a well-liked supervised classification paradigm. a documented Bayesian network classifier is that the Naïve Bayes’ classifier could even be a probabilistic classifier supported the Bayes’ theorem, considering Naïve (Strong) independence assumption.It was introduced under a special name into the text retrieval community and remains a popular(baseline) method for text categorizing, the problem of judging documents as belonging to a minimum of 1 category or the opposite with word frequencies because the feature. a plus of Naïve Bayes’ is that it only requires a little amount of coaching data to estimate the parameters necessary for classification. Abstractly, Naïve Bayes’ could even be a contingent probability model. Despite its simplicity and powerful assumptions, the naïve Bayes’ classifier has been proven to figure satisfactorily in many domains. Bayesian classification provides practical learning algorithms and prior knowledge and observed data are often combined. In Naïve Bayes’ technique, the essential idea to hunt out the probabilities of categories given a text document by using the joint probabilities words and categories. It is based on the thought of word independence. The starting point is that the Bayes’ theorem for conditional probability, stating that, for a given datum x and class C:<br/>
  P (C / x) = P(x/C)/P(x) <br/>
  Furthermore, by making the thought that for a data point x = {x1,x2,...xj}, the probability of every of its attributes occurring during a given class is independent, we can estimate the probability of x as follows:
  <br/>P(C/x)=P(C).∏P(xi/C)

## System Architecture

### Modules

1. **Feature Extractor:**
   - Reduces dimensionality using bag-of-words for further processing.
   - Converts raw tweets into feature vectors for SVM training.
  
      Feature Extraction is a process to reduce dimensionality of text by converting it into more manageable groups for further processing. It combines the variables into features, reducing the amount of data to be processed while still precisely describing original data. This reduces amount of redundant data. All the raw tweets in the csv file are converted into feature vectors to fit into the training dataset for SVM.

2. **Data Preparation:**
   - Tokenization, data cleaning, noise removal, and regular expressions.
   - Prepares data for SVM and Naive Bayes classification.

3. **Classifier Model:**
   - Trains SVM and Naive Bayes on the dataset.
   - Evaluates performance on the testing dataset.

       Classifier model is used to classify the text as either hate-speech or non hate-speech according to the percentage of embedded sentiment assigned to it by the sentiment analyzer. The classifier model is prepared by training SVM and Naive Bayes on training datasets and finally evaluating their performance on the testing dataset.

  ![image](https://user-images.githubusercontent.com/62890614/227709745-16ccc49e-ef54-4c77-bf4c-9f17439858a3.png)
  
  Fig1. Architecture Diagram
  
  
  This process can be explained with the help of an example like: -
  
  ![image](https://user-images.githubusercontent.com/62890614/227709809-e934f983-a416-43be-8fef-9bcb37037129.png)
  
  
  Fig2. Same Text Implementation
  
  
  Firstly, the data is tokenized, that is, it is split up into smaller parts called tokens. The punkt 
  tokenizer package is used, which splits the strings into sentences and words. Normalization 
  groups together words with same meaning, for example, “run”, “ran” and “runs”. This 
  reduces the redundancy in the training data. Lemmatization analyses the structure of the word 
  and converts it into normalized form. The tagging algorithm evaluates the context of the 
  words in the data. For example, NN for noun and VB for verb. In the figure, it can be noticed 
  that, “being” changes to “be”, which is its’ root word. Noise means unnecessary text. Noise 
  reduction is done via Regular Expressions to remove hyperlinks, social media handles, 
  punctuation and more. In this step, all the text is converted into lowercase. The common 
  words appearing in negative and positive texts according to the training data are identified 
  using the FreqDist class of NLTK. From the figure, it can be seen that smiling emoticon is 
  commonly present in positive sentences. The negative to positive ratios are assigned to the 
  words of the training dataset. The cleaned data are converted into a dictionary, which is 
  randomly shuffled into a dataset and split for training and testing.

## Data Flow Design

![Data Flow Diagram](https://user-images.githubusercontent.com/62890614/227709914-89ae3db5-f7e6-45b6-a709-80145f6fa68a.png)

  In Data flow Diagram it is explained how the raw data flows and eventually turns into the required information that we need for hate speech detection. First the raw data that is tweets are taken and its features are extracted which are basically the words without verbs etc. Then the rest of the data is cleaned. And when we get only cleaned and meaningful words from the text, then we apply classification algorithm to understand the polarity of the sentence and also whether it points towards cyberbullying or not.

## Class Diagram

![Class Diagram](https://user-images.githubusercontent.com/62890614/227709977-cbd3dfb6-7722-4bf4-ba36-42cb8a404744.png)

  The class diagram represents the entities and classes that are involved in Hate Speech Detection process. The classes have their properties and respective functions which are shown in the class diagram above.

## Results

![image](https://user-images.githubusercontent.com/62890614/227710194-e67abe18-42eb-4ab5-8a4d-b0aca759788e.png)

![image](https://user-images.githubusercontent.com/62890614/227710214-a643e4a8-05b0-4af5-81ef-8c4521ca820f.png)

![image](https://user-images.githubusercontent.com/62890614/227710245-fea2d433-8c65-45ec-ac36-38fdebf7edc9.png)

![image](https://user-images.githubusercontent.com/62890614/227710279-07459918-26b3-460e-8012-4f84ff974fc4.png)

![image](https://user-images.githubusercontent.com/62890614/227710304-09f58866-cc36-4d17-9eee-89be6144c438.png)

## More About The Process

  The methodology behind the annotation process is simplistic. Natural language processing (NLP) which converts all the texts into numerals and vectorize them which are easily understandable by the machine. To proceed with our motivation, data cleaning must be done. Feature Extraction is a process to reduce dimensionality of text by converting it into more manageable groups for further processing. It combines the variables into features, reducing the amount of data to be processed while still precisely describing original data. This reduces amount of redundant data. The method of feature extraction used in this project is Bag-of-words. It uses NLP to extract words and classify them by frequency of occurrence. This includes removing stop words, stemming, tokenization, etc. Once the data is clean, the method of classification to be used is sentiment analysis for hate speech detection. Sentiment analysis is the process of analysing text I order to determine its’ emotional tone. It will categorize a line as positive or negative using sentiment score which reflects the depth of emotions in the text. Sentiment Analysis is 24 mostly used as a classification tool that analyses the social media text and indicates if the emotion behind the text is positive or negative. After the data is separated based on sentiment, now we need to classify the data as hateful or not and for this task we need a classifier model .Classifier model is used to classify the text as ‘hate speech’ or ‘non-hate speech’ according to the percentage of embedded sentiment assigned to it by the sentiment analyser. The rule based algorithms’ aim is to identify and utilize a set of related rules that is capable of representing the information gained by the system. After this is done, we need a classifier model for classifying the data and for that we have used is Naïve Bayes Algorithm. This classifier was created using existing computing tools. We used to scikit-learn a python package than implements most of the machine learning algorithms including naïve Bayes algorithm and feature extraction techniques. The classifier model is created using the multinomial () function which is part the scikit-learn package. The package also contains functions like Count Vectorizer () for bag of words implementation and transforming documents to feature vectors. In addition to Naïve Bayes Algorithm, SVM is also used as a comparative analysis for greater accuracy of classification. SVM (Support Vector Machine) is a supervised machine learning algorithm used as a classifier. The values are plotted and a hyperplane is chosen in such a way that it maximizes the margin of the training data. Once the training is done, the input data is processed by the classifier as positive or negative indicating presence of cyber bullying or not. Both, SVM (Support Vector machine) and Naïve Bayes algorithm have been used to create a classification model. The test data is classified by both the models separately, and, on comparison of the results, a cumulative result is shown; whether text is ‘cyber bullying detected’ or ‘ cyber bullying not detected’.

## Conclusion & Future Work

  This project successfully created a hate speech classifier using Naive Bayes and SVM, achieving an average precision score of 95%. The system has room for improvement, especially in handling false positives, true negatives, and understanding nuances like sarcasm or regional languages.
  
  Despite this limitation, Naïve Bayes still performs well and can be enhanced by using more trigrams or longer word combinations.
  
  Although SVM is good for text classification but SVM algorithm is not suitable for large data sets and large texts .Also one major drawback that is faced using this system is that it does not read false positive and true negative comments. For example .if the text says ,”I am not very happy”, then the application might take the word “very” and “happy” and consider it to be positive which is wrong. Also, the application is unable to read GIFS and audio and videos .Another area where it can be improved in future s that it presently reads and analyses text which is written only in English language and people also sometimes use regional language which then changes the sentiment of the overall text sometimes. Also, the application is unable to read and understand any sarcastic comment. For example ,if the text says ,”oh my god ,you are so intelligent ,hahaha” ,the application would consider it as a positive comment .All these are areas which show a way for improvement in this application in future. This research will add to the body of knowledge in the field of curbing online hate, online hate speech monitoring, social media data mining and application of machine learning algorithms to solve real life problems.

The research contributes to curbing online hate, social media data mining, and applying machine learning to real-life problems.

Feel free to explore, contribute, and join the journey of making online spaces safer! 🌐🛡️
