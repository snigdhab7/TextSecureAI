import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import time
from sklearn import svm
from sklearn.metrics import classification_report
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import twitter_samples, stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk import classify, NaiveBayesClassifier
import re, string, random
#NB- remove noise function
def remove_noise(tweet_tokens, stop_words = ()):

    cleaned_tokens = []

    #remove noise using regular expressions- like http links, userid tags,hashtags 
    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(@[A-Za-z0-9_]+)","", token)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens
#NB-get all words function
def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token

def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)
        
if __name__ == "__main__":
  #datasets used
   #SVM datasets used
    trainData = pd.read_csv("C:\\Users\\Snigdhabose\\Desktop\\major proj\\code\\randforeg1\\train.csv")
    testData = pd.read_csv("C:\\Users\\Snigdhabose\\Desktop\\major proj\\code\\randforeg1\\test.csv")
   #NB datasets used
    #500 pos 500 neg 20000 neutral
    positive_tweets = twitter_samples.strings('positive_tweets.json')
    negative_tweets = twitter_samples.strings('negative_tweets.json')
    text = twitter_samples.strings('tweets.20150430-223406.json')

    print("-------------SVM Classifier-----------------\n")
  #SVM- Create feature vectors
    vectorizer = TfidfVectorizer(min_df = 5,
                             max_df = 0.8,
                             sublinear_tf = True,
                             use_idf = True)
    train_vectors = vectorizer.fit_transform(trainData['Content'])
    test_vectors = vectorizer.transform(testData['Content'])
  #SVM- Perform classification with SVM, kernel=linear
    classifier_linear = svm.SVC(kernel='linear')
    t0 = time.time()
    classifier_linear.fit(train_vectors, trainData['Label'])
    t1 = time.time()
    prediction_linear = classifier_linear.predict(test_vectors)
    t2 = time.time()
    time_linear_train = t1-t0
    time_linear_predict = t2-t1
    
  #SVM- Classifier Report
    
    print("Sample Train Data for SVM Classifier:")
    print(trainData.sample(frac=1).head(5))
    print("\nSVM Classifier Report:-")
    print("Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict))
    report = classification_report(testData['Label'], prediction_linear, output_dict=True)
    print("*NOTE: F1 = 2 * (precision * recall) / (precision + recall)")
    print('positive: ', report['pos'])
    print('negative: ', report['neg'])
   
    print("\n-------------NB Classifier-----------------")
   #NB- tokenize, stop words 
    tweet_tokens = twitter_samples.tokenized('positive_tweets.json')[0]

    stop_words = stopwords.words('english')

    positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
    negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')

    positive_cleaned_tokens_list = []
    negative_cleaned_tokens_list = []

   #NB-pass tokenized n stop words to remove noise function
    for tokens in positive_tweet_tokens:
        positive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))
    for tokens in negative_tweet_tokens:
        negative_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

    positive_tokens_for_model = get_tweets_for_model(positive_cleaned_tokens_list)
    negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tokens_list)

    positive_dataset = [(tweet_dict, "Positive")
                         for tweet_dict in positive_tokens_for_model]
    negative_dataset = [(tweet_dict, "Negative")
                         for tweet_dict in negative_tokens_for_model]

    dataset = positive_dataset + negative_dataset
    random.shuffle(dataset)
    train_data = dataset[:7000]
    test_data = dataset[7000:]
  #NB- classifier
    classifier = NaiveBayesClassifier.train(train_data)
  #NB- classifier report
    print("Naive Bayes Classifier Report:-")
    print("Accuracy is:", classify.accuracy(classifier, test_data))
    print("**MOST COMMON INFORMATIVE FEATURES ARE:-**")
    print(classifier.show_most_informative_features(10))

#Both Classifiers Are ready for testing with new data sample
    print("\n-------------Testing the models-----------------")
    #test1
    print("SAMPLE TEXT1:")
    custom_tweet = "@shivangi234 is so fat. hahaha. She didn't even deserve the title of Miss glam2020. She is so ugly and fat too. All these shows are a scam."
    print(custom_tweet)
    #NB
    custom_tokens = remove_noise(word_tokenize(custom_tweet))
    nbresult=classifier.classify(dict([token, True] for token in custom_tokens))
    print("NB Classifier Result:",nbresult)
    #SVM
    review_vector = vectorizer.transform([custom_tweet]) # vectorizing
    if classifier_linear.predict(review_vector)=='neg':
        svmresult="Negative"
    else:
        svmresult="Positive"
    print("SVM Classifier Result:",svmresult)
    #Compare Results
    print("**RESULT**")
    if svmresult==nbresult:
        if svmresult=='Negative':
            print("CyberBullying is Detected (using SVM Classifier and naive Bayes Classifier)")
        else:
            print("CyberBullying is not Detected (using SVM Classifier and naive Bayes Classifier)")
    else:
        print("Unable to Detect")
    
   