import string
from collections import Counter
import nltk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
import numpy as np
from nltk.corpus import stopwords
import string

from nltk.classify.scikitlearn import SklearnClassifier
from nltk.classify import ClassifierI
from nltk.tokenize import word_tokenize
from nltk.corpus import movie_reviews
from sklearn.naive_bayes import MultinomialNB

text = open('read.txt', encoding='utf-8').read()
hacker = text.split()

lower_case = text.lower()
cleaned_text = lower_case.translate(str.maketrans('', '', string.punctuation))

# Using word_tokenize because it's faster than split()
tokenized_words = word_tokenize(cleaned_text, "english")

# Removing Stop Words
final_words = []
for word in tokenized_words:
    if word not in stopwords.words('english'):
        final_words.append(word)

# Lemmatization - From plural to single + Base form of a word (example better-> good)
lemma_words = []
for word in final_words:
    word = WordNetLemmatizer().lemmatize(word)
    lemma_words.append(word)

emotion_list = []
with open('emotions.txt', 'r') as file:
    for line in file:
        clear_line = line.replace("\n", '').replace(",", '').replace("'", '').strip()
        word, emotion = clear_line.split(':')

        if word in lemma_words:
            emotion_list.append(emotion)

w = Counter(emotion_list)


def sentiment_analyse(sentiment_text):
    score = SentimentIntensityAnalyzer().polarity_scores(sentiment_text)
    print(score)

    if score['neg'] > score['pos']:
        print("Negative Sentiment")
    elif score['neg'] < score['pos']:
        print("Positive Sentiment")
    else:
        print("Neutral Sentiment")


sentiment_analyse(cleaned_text)

fig, ax1 = plt.subplots()
ax1.bar(w.keys(), w.values())
fig.autofmt_xdate()
plt.savefig('graph.png')
plt.show()

with open('read.txt', 'r', encoding='utf-8') as coders:
    hacker = str(coders.read()).split()

with open('read.txt', 'r', encoding='utf-8') as coders:
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
print("Data Accuracy Result : ", (nltk.classify.accuracy(MNB_classifier, testing_set)) * 100)

print(str("NLTK Classifier Accuracy Result : "), (nltk.classify.accuracy(classifier, testing_set)) * 100)

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
