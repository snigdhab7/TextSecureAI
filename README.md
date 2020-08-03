# CyberBullyingDetection-SVM-NB


<h3>ABSTRACT</h3>

<p>With the increase usage, hate speech is currently of current interest in the domain of social media. The anonymity and flexibility afforded by the Internet has made it easy for people to communicate in an abusive and hateful manner. And as the amount of cyber-bullying is increasing, methods that automatically detect hate speech is becoming an urgent need. 
On Twitter, hateful tweets are the ones that contain abusive speech targeting selected users like (cyber-bullying, a politician, a celebrity, a product) or particular groups (a country, a religion, gender, an organization, etc.). Detecting such hateful comments is important for analysing public sentiment of an individual or group of users towards another set of groups, and for discouraging associated wrongful activities. The manual way of filtering out hateful tweets is not scalable and flexible, motivating researchers to identify automated ways. In this project, we focus on the problem of classifying a tweet as positive or negative text. The task is quite challenging due to the inherent complexity of the natural language constructs – different forms of hatred, different kinds of targets, different ways of representing the same meaning. Most of the earlier work revolves around using representation learning methods followed by a linear classifier. However, recently deep learning methods and SVM have shown accuracy improvements across amount of data. Hence, In order to reduce this problem, we propose a system that can detect cybercrimes from social media automatically using Naive Bayes Classifier and SVM. 
</p>
<h4>Text Classiﬁcation Method Selection- </h4>
<ul>
<li><p>	<h5>Support Vector Machine(SVM)-</h5></br>It has been chosen for the classiﬁcation within the experiments. The support-vector machines could even be a learning machine for two-group classiﬁcation problems introduced by . it's wont to classify the texts as positives or negatives. SVM works well for text classiﬁcation due toits advantages like its potential to handle large features.Another advantage is SVM is strong when there's a sparse set of examples and also because most of the matter are linearly separable . Support Vector Machine have shown promising leads to previous research in sentiment analysis 
</br><h6>Effectiveness Measures in SVM-</h6></br>
Four effective measures that are utilized during this study are supported confusion matrix output, which are True Posi-tive (TP), False Positive(FP), True Negative (TN), and FalseNegative(FN)-
</br>•Precision(P) = TP/(TP+FP)</br>
•Recall(R) = TP/(TP+FN)</br>
•Accuracy(A) = (TP+TN)/(TP + TN + FP + FN)</br>
•AUC (Area under the (ROC) Curve) = 1/2.((T-P/(TP+FN))+(TN/(TN+FP))</br>
•F-Measure(Micro-averaging) = 2.(P.R)/(P+R)</br>
The text categorization effectiveness is typically measured using the F1, accuracy, and AUC . F1 measure could even be a combined effectiveness measure determined by precision and recall. the earth under the ROC curve (AUC) has become an honest measurement of performance of supervised classiﬁcation rules. However, the straightforward quite AUC is just applicable to the case of two classes..
</p></li>
<li>
<p><h5>Naïve Bayes’ Classifier-</h5></br>
Bayesian network classifiers are a well-liked supervised classification paradigm. a documented Bayesian network classifier is that the Naïve Bayes’ classifier could even be a probabilistic classifier supported the Bayes’ theorem, considering Naïve (Strong) independence assumption.It was introduced under a special name into the text retrieval community and remains a popular(baseline) method for text categorizing, the problem of judging documents as belonging to a minimum of 1 category or the opposite with word frequencies because the feature. a plus of Naïve Bayes’ is that it only requires a little amount of coaching data to estimate the parameters necessary for classification.
Abstractly, Naïve Bayes’ could even be a contingent probability 
model. Despite its simplicity and powerful assumptions, the
naïve Bayes’ classifier has been proven to figure satisfactorily in many domains. Bayesian classification
provides practical learning algorithms and prior knowledge and observed data are often combined. In Naïve Bayes’ technique, the essential idea to hunt out the
probabilities of categories given a text document by using the joint probabilities  words and categories. It is based on the thought of word independence. The
starting point is that the Bayes’ theorem for conditional probability, stating that, for a given datum x and class C:</br>
P (C / x) = P(x/C)/P(x) </br>
Furthermore, by making the thought that for a data point x = {x1,x2,...xj}, the probability of every of its attributes occurring during a given class is independent,
we can estimate the probability of x as follows:</br>
P(C/x)=P(C).∏P(xi/C)</br>
</li>
</ul>
