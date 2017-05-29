# this code runs a number of basic machine learning classifiers

import codecs, re
# preprocessing
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import LabelEncoder
# classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
# model pipeline stuff
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics.classification import accuracy_score
from sklearn.metrics import f1_score


# load the data to classify
# data should be in two txt/csv files (data and classes)
# each sample should be one line
print("Loading data...\n")

f_sents = codecs.open('datasets/met_corpus.txt', 'rb', encoding='utf8')
f_classes = codecs.open('datasets/met_labels.txt', 'rb', encoding='utf8')
sents = [sent.strip() for sent in f_sents.readlines()]
labels = [label.strip() for label in f_classes.readlines()]


# tokenizing:
print("Fitting tokenizer...\n")

# a custom tokenizing function for our data
# this code turns to lowercase, strips punctuation with re,
# splits to list, and stems words using the nltk snowball stemmer
# we can get fancy here with stopwords, punctuation, etc
def tokenize(sentence):
    stemmer = SnowballStemmer("english")
    sentence = sentence.lower()
    sentence = re.sub(r'[^\w\s]', '', sentence)
    wordlist = sentence.strip('\n').split(' ')
    result = [stemmer.stem(word) for word in wordlist]
    return result


# prepare classification labels
print("Preparing labels...\n")
encoder = LabelEncoder()
y = encoder.fit_transform(labels)


# generate new training and test data
print('generating training data...\n')
X_train, X_test, y_train, y_test = train_test_split(sents, y, test_size=0.2)


# check data
# print(X_train[0])
# print(y_train[0])
# print('')

tests = []
testacc = []
testf1s = []

#############################################
# TF.IDF with Random Forest (pipeline 0)
#############################################

name = 'TF.IDF with Random Forest'

vectorizer = CountVectorizer(tokenizer=tokenize)

pipeline0 = Pipeline([
    ('vect', vectorizer),
    ('tfidf', TfidfTransformer()),
    ('rfor', RandomForestClassifier(n_estimators=1000))
])

# Train
print("Fitting", name, "...\n")
pipeline0.fit(X_train, y_train)

# Test
pred_train = pipeline0.predict(X_train)
pred_test = pipeline0.predict(X_test)

# print evaluations
print("Train accuracy:", accuracy_score(y_train, pred_train))
print("Test  accuracy:", accuracy_score(y_test, pred_test))

print("Train F1-score:", f1_score(y_train, pred_train, average='macro'))
print("Test  F1-score:", f1_score(y_test, pred_test, average='macro'))

print('')

# display some tests
for idx, sent in enumerate(X_test[:10]):
    print(encoder.inverse_transform([y_test[idx]])[0], "|",
          encoder.inverse_transform(pipeline0.predict([sent])[0]),
          " : ", sent)

print('')
print('')

tests.append(name)
testacc.append(accuracy_score(pred_test, y_test))
testf1s.append(f1_score(y_test, pred_test, average='macro'))

#############################################
# TF.IDF with LR (pipeline 1)
#############################################

name = 'TF.IDF and Logistic Regression'

vectorizer = CountVectorizer(tokenizer=tokenize)

pipeline1 = Pipeline([
    ('vect', vectorizer),
    ('tfidf', TfidfTransformer()),
    ('clf', LogisticRegression())
])

# Train
print("Fitting", name, "...\n")
pipeline1.fit(X_train, y_train)

# Test
pred_train = pipeline1.predict(X_train)
pred_test = pipeline1.predict(X_test)

# print evaluations
print("Train accuracy:", accuracy_score(y_train, pred_train))
print("Test  accuracy:", accuracy_score(y_test, pred_test))

print("Train F1-score:", f1_score(y_train, pred_train, average='macro'))
print("Test  F1-score:", f1_score(y_test, pred_test, average='macro'))

print('')

# display some tests
for idx, sent in enumerate(X_test[:10]):
    print(encoder.inverse_transform([y_test[idx]])[0], "|",
          encoder.inverse_transform(pipeline1.predict([sent])[0]),
          " : ", sent)

print('')
print('')

tests.append(name)
testacc.append(accuracy_score(pred_test, y_test))
testf1s.append(f1_score(y_test, pred_test, average='macro'))

#############################################
# TF.IDF with LR, balanced classes (pipeline 2)
#############################################

name = 'TF.IDF and Logistic Regression (bw)'

vectorizer = CountVectorizer(tokenizer=tokenize)

pipeline2 = Pipeline([
    ('vect', vectorizer),
    ('tfidf', TfidfTransformer()),
    ('clf', LogisticRegression(class_weight='balanced'))
])

# Train
print("Fitting", name, "...\n")
pipeline2.fit(X_train, y_train)

# Test
pred_train = pipeline2.predict(X_train)
pred_test = pipeline2.predict(X_test)

# print evaluations
print("Train accuracy:", accuracy_score(y_train, pred_train))
print("Test  accuracy:", accuracy_score(y_test, pred_test))

print("Train F1-score:", f1_score(y_train, pred_train, average='macro'))
print("Test  F1-score:", f1_score(y_test, pred_test, average='macro'))

print('')

# display some tests
for idx, sent in enumerate(X_test[:10]):
    print(encoder.inverse_transform([y_test[idx]])[0], "|",
          encoder.inverse_transform(pipeline2.predict([sent])[0]),
          " : ", sent)

print('')
print('')

tests.append(name)
testacc.append(accuracy_score(pred_test, y_test))
testf1s.append(f1_score(y_test, pred_test, average='macro'))

#############################################
# TF.IDF with Linear SVM (pipeline 3)
#############################################

name = 'TF.IDF with Linear SVM'

vectorizer = CountVectorizer(tokenizer=tokenize)

linsvmclass = LinearSVC()

pipeline3 = Pipeline([
    ('vect', vectorizer),
    ('tfidf', TfidfTransformer()),
    ('svm', linsvmclass)
])

# Train
print("Fitting", name, "...\n")
pipeline3.fit(X_train, y_train)

# Test
pred_train = pipeline3.predict(X_train)
pred_test = pipeline3.predict(X_test)

# print evaluations
print("Train accuracy:", accuracy_score(y_train, pred_train))
print("Test  accuracy:", accuracy_score(y_test, pred_test))

print("Train F1-score:", f1_score(y_train, pred_train, average='macro'))
print("Test  F1-score:", f1_score(y_test, pred_test, average='macro'))

print('')

# display some tests
for idx, sent in enumerate(X_test[:10]):
    print(encoder.inverse_transform([y_test[idx]])[0], "|",
          encoder.inverse_transform(pipeline3.predict([sent])[0]),
          " : ", sent)

print('')
print('')

tests.append(name)
testacc.append(accuracy_score(pred_test, y_test))
testf1s.append(f1_score(y_test, pred_test, average='macro'))

#############################################
# TF.IDF with Linear SVM, balanced classes (pipeline 4)
#############################################

name = 'TF.IDF with Linear SVM (bw)'

vectorizer = CountVectorizer(tokenizer=tokenize)

linsvmclass = LinearSVC(class_weight='balanced')

pipeline4 = Pipeline([
    ('vect', vectorizer),
    ('tfidf', TfidfTransformer()),
    ('svm', linsvmclass)
])

# Train
print("Fitting", name, "...\n")
pipeline4.fit(X_train, y_train)

# Test
pred_train = pipeline4.predict(X_train)
pred_test = pipeline4.predict(X_test)

# print evaluations
print("Train accuracy:", accuracy_score(y_train, pred_train))
print("Test  accuracy:", accuracy_score(y_test, pred_test))

print("Train F1-score:", f1_score(y_train, pred_train, average='macro'))
print("Test  F1-score:", f1_score(y_test, pred_test, average='macro'))

print('')

# display some tests
for idx, sent in enumerate(X_test[:10]):
    print(encoder.inverse_transform([y_test[idx]])[0], "|",
          encoder.inverse_transform(pipeline4.predict([sent])[0]),
          " : ", sent)

print('')
print('')

tests.append(name)
testacc.append(accuracy_score(pred_test, y_test))
testf1s.append(f1_score(y_test, pred_test, average='macro'))

'''
# todo: not working due to tense vector crap
#############################################
# TF.IDF with Gradient Boosting (pipeline 5)
# takes long time to run, uncomment if curious
#############################################

from sklearn.base import TransformerMixin
from sklearn.ensemble import GradientBoostingClassifier

name = 'TF.IDF with Gradient Boosting'

vectorizer = CountVectorizer(tokenizer=tokenize)

gradboost = GradientBoostingClassifier()

#http://zacstewart.com/2014/08/05/pipelines-of-featureunions-of-pipelines.html
class DenseTransformer(TransformerMixin):

    def transform(self, X, y=None, **fit_params):
        return X.todense()

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def fit(self, X, y=None, **fit_params):
        return self

pipeline5 = Pipeline([
    ('vect', vectorizer),
    ('tfidf', TfidfTransformer()),
    ('to_dense', DenseTransformer()),
    ('svm', gradboost)
])

# Train
print("Fitting", name, "...\n")
pipeline5.fit(X_train, y_train)

# Test
pred_train = pipeline5.predict(X_train)
pred_test = pipeline5.predict(X_test)
print("Train accuracy:", accuracy_score(pred_train, y_train))
print("Test  accuracy:", accuracy_score(pred_test, y_test))
print('')

# display some tests
for idx, sent in enumerate(X_test[:10]):
    print(encoder.inverse_transform([y_test[idx]])[0], "|",
          encoder.inverse_transform(pipeline5.predict([sent])[0]),
          " : ", sent)

print('')
print('')

tests.append(name)
testacc.append(accuracy_score(pred_test, y_test))
'''

# print out stuff
for idx, test in enumerate(tests):
    print(test, "test accuracy:", testacc[idx], "f1_score:", testf1s[idx])