#importing pandas
import pandas as pd
#reading the file
train = pd.read_csv("../data/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
#exploratory data anaysis
train.head()
#exploration again
print("Shape of the training data csv file:",train.shape)
#still exploring
print("Column names in train.csv file:",train.columns.values)


#using BeautifulSoup to clean data initially
from bs4 import BeautifulSoup
import re
#importing stopwords from nltk
print("Importing nltk.corpus.stopword . . . . . ")
from nltk.corpus import stopwords
#some stopwords in english language are
print("All stopwords in English:")
print(stopwords.words("english"))


# let's make a function from above code that can clean all the reviews
def review_to_words(raw_review):
    #remove html using BeautifulSoup
    review_text = BeautifulSoup(raw_review,"html.parser").get_text()
    #removing raw letters,numbers,punctuations
    letters_only = re.sub("[^a-zA-Z]"," ",review_text)
    #creating an array , resolving whitespaces
    words = letters_only.lower().split()
    #create an array of stopwords so that we don't have to access corpus to search for a stopword
    stop = set(stopwords.words("english"))
    #removing stopwords from the raw_review
    meaningful_words = [w for w in words if w not in stop]
    #return a string with only the words that are important
    return(" ".join(meaningful_words))


num_reviews = train.review.size
print("Total number of reviews :",num_reviews)
#storing all reviews at one place
print("Cleaning the whole data . . . . .")
clean_train_reviews = []
for i in range(num_reviews):
    clean_train_reviews.append(review_to_words(train.review[i]))
    if(i%5000==0):
        print("cleaning in process . . . . . . ")
        #included an extra line to visualize if our code is working properly or not
        print("cleaned ",i,"reviews till now")
print("cleaning complete")


print("Creating a bag of words . . . . . ")
#import CountVectorizer to create token counts of document
from sklearn.feature_extraction.text import CountVectorizer
#initializing the parameters as None so that we can write and manipulate the processing by our own
vectorizer = CountVectorizer(analyzer="word",
                            tokenizer=None,
                            preprocessor=None,
                            stop_words=None,
                            max_features=5000)
#train the classifer using fit_transform() method
train_data_features = vectorizer.fit_transform(clean_train_reviews)
#change the classifier into array
train_data_features = train_data_features.toarray()
print(train_data_features.shape)
#see all the features names
vocab = vectorizer.get_feature_names()
print("\nAll feature names:")
print(" , ".join(vocab[0:10])," . . . . "," , ".join(vocab[-10:]))

print("printing count of all the features . . . .")
import numpy as np
#frequency of each word is found using np.sum()
dist = np.sum(train_data_features,axis=0)
ct = 0
for tag,count in zip(vocab,dist):
    print(tag,":",count,end="\t\t")
startswith = []
for val in vocab:
    if(val[0] not in startswith):
        startswith.append(val[0])
print(startswith)
#counting the total numbers of words starting
counts = np.zeros((len(startswith)),dtype=np.int)
for val in vocab:
    index = startswith.index(val[0])
    counts[index] += 1


import matplotlib.pyplot as plt
plt.figure(1,figsize=(15,5))
plt.plot(counts)
nums = [i for i in range(26)]
plt.xticks(nums,startswith)
plt.grid()
plt.ylabel("frequency")
plt.show()

print("Training the classification models:")
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators = 100)
print("fitting RandomForest . . . ")
forest = forest.fit(train_data_features,train["sentiment"])
from sklearn.naive_bayes import MultinomialNB
naive = MultinomialNB()
print("fitting NaiveBayes . . . ")
naive.fit(train_data_features,train["sentiment"])
from sklearn.ensemble import AdaBoostClassifier
adaboost = AdaBoostClassifier(n_estimators = 100)
print("fitting AdaBoost . . . ")
adaboost.fit(train_data_features,train["sentiment"])
print("fitting complete.")

print("Reading test dataset. . . . ")
test = pd.read_csv("../data/testData.tsv",header=0,delimiter="\t",quoting=3)
print("Shape of test dataset: :",test.shape)
print(test.info())


num_reviews = len(test["review"])
clean_test_reviews = []
print("Cleaning and parsing the test dataset. . . . ")
for i in range(0,num_reviews):
    if((i+1)%3000 == 0):
        print(i+1," reviews processed . . .")
    clean_review = review_to_words(test["review"][i])
    clean_test_reviews.append(clean_review)
print("processing complete.")

print("test data analysis in process . . . . .")
test_data_features = vectorizer.fit_transform(clean_test_reviews)
test_data_features = test_data_features.toarray()
print("predicting using RandomForest . . . ")
result1 = forest.predict(test_data_features)
print("predicting using Naive Bayes . . . ")
result2 = naive.predict(test_data_features)
print("predicting using AdaBoost . . . ")
result3 = adaboost.predict(test_data_features)
print("Prediction complete.")
result = result1+result2+result3
for i in range(25000):
    if(result[i]==1):
        result[i]=0
    elif(result[i]==2):
        result[i]=1
    elif(result[i]==3):
        result[i]=1
output = pd.DataFrame(data = {"id":test["id"],"sentiment":result})


print("Writing output to file . . . . . ")
output.to_csv("../output/Bag_of_Words_model.csv", index=False, quoting=3)
print("Output written to file.")