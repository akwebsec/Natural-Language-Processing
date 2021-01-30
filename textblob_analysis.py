from textblob import TextBlob
import nltk
import numpy as np
from nltk.corpus import stopwords
import string

from nltk.classify.scikitlearn import SklearnClassifier
from nltk.classify import ClassifierI
from nltk.tokenize import word_tokenize
from nltk.corpus import movie_reviews
from sklearn.naive_bayes import MultinomialNB

with open('reviews.txt', 'r', encoding='utf-8') as coders:
    hacker = str(coders.read()).split()

with open('reviews.txt', 'r', encoding='utf-8') as coders:
    cracker = str(coders.read())

documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

np.random.shuffle(documents)

training_set = documents
testing_set = documents

all_words = []

for words_list, categ in documents:
    for w in words_list:
        all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

word_features = list(hacker)


# print(word_features)

def find_features(training_set):
    words = set(training_set)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features


featuresets = [(find_features(rev), category) for (rev, category) in documents]

np.random.shuffle(featuresets)

training_set = featuresets
testing_set = featuresets

MNB_classifier = SklearnClassifier(MultinomialNB())
classifier = nltk.NaiveBayesClassifier.train(training_set)

MNB_classifier.train(training_set)

stopwords_english = stopwords.words('english')


def bag_of_words(words):
    words_clean = []

    for word in words:
        word = word.lower()
        if word not in stopwords_english and word not in string.punctuation:
            words_clean.append(word)

    words_dictionary = dict([word, True] for word in words_clean)

    return words_dictionary


custom_review = cracker
custom_review_tokens = word_tokenize(custom_review)
custom_review_set = bag_of_words(custom_review_tokens)
prob_result = classifier.prob_classify(custom_review_set)


# print(prob_result.max())


# classifier.show_most_informative_features(5)

def percentage(part, whole):
    return 100 * float(part) / float(whole)


text = open('reviews.txt', encoding='utf-8').read()
lower_case_text = text.lower()

cleaned_text = lower_case_text.translate(str.maketrans('', '', string.punctuation))

blob = TextBlob(cleaned_text)

analysis = blob.sentiment.polarity

print(analysis)


if (analysis == 0):  # adding reaction of how people are reacting to find average later
    print("Neutral")
elif (analysis > 0 and analysis <= 0.3):
    print("Wpositive")
elif (analysis > 0.3 and analysis <= 0.6):
    print("Positive")
elif (analysis > 0.6 and analysis <= 1):
    print("Spositive")
elif (analysis > -0.3 and analysis <= 0):
    print("Wnegative")
elif (analysis > -0.6 and analysis <= -0.3):
    print("Negative")
elif (analysis > -1 and analysis <= -0.6):
    print("Snegative")
print("Data Accuracy Result : ", (nltk.classify.accuracy(MNB_classifier, testing_set)) * 100)
print(str("TextBlob Classifier Accuracy Result : "), (nltk.classify.accuracy(classifier, testing_set)) * 100)