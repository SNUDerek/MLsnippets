# this is a simple cnn-rnn classifier
# it uses two files: see the 'load the data.' section
# one file for documents, one file for labels
# adjust the load the data section to use dataframe etc

import codecs, re
import numpy as np

# Scikit-Learn and NLTK for preprocessing ENGLISH data
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Keras for neural network classifier
from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.callbacks import EarlyStopping
from keras.models import save_model, load_model


# parameters
MAX_VOCAB = 18000    # limit total vocabulary
MAX_LENGTH = 100	 # maximum number of words per 'document' (sent etc)
EMBEDDING_SIZE = 64	 # embedding size (trainable randomized embeddings)
BATCH_SIZE = 32
MAX_EPOCHS = 25
DROPOUT_RATE = 0.4

stop_monitor = 'val_loss'  # variable for early stop: (default = val_loss)
stop_delta = 0.0   # minimum delta before early stop (default = 0)
stop_epochs = 2    # how many epochs to do after stop condition (default = 0)

# load the data.
print("Loading data...\n")

# todo : edit filenames here
f_sents = codecs.open('datasets/brown_sents.txt', 'rb', encoding='utf8')
f_classes = codecs.open('datasets/brown_topics.txt', 'rb', encoding='utf8')
sents = [sent.strip() for sent in f_sents.readlines()]
labels = [label.strip() for label in f_classes.readlines()]
# number of labels
num_labels = len(set(labels))

# we can create a custom tokenizer to clean and preprocess the data.


# we can use tokenizing function using sklearn, etc here
# so we can get fancy here with stopwords, etc
def tokenize(sentence):
    stemmer = SnowballStemmer("english")
    sentence = sentence.lower()
    sentence = re.sub(r'[^\w\s]', '', sentence)
    wordlist = sentence.strip('\n').split(' ')
    result = [stemmer.stem(word) for word in wordlist]
    return result


# prepare labels
print("Preparing labels...\n")
encoder = LabelEncoder()
labels = encoder.fit_transform(labels)


# generate new training data
print('generating training data...\n')
X_train, X_test, y_train, y_test = train_test_split(sents, labels, test_size=0.2)

train_sents = X_train
test_sents = X_test


# integer-index with dataset.py
print("integer-indexing input and output...\n")
cvectorizer = CountVectorizer(analyzer='char')
cvectorizer.fit(X_train)
X_vocab = cvectorizer.vocabulary_

vocab_size = len(X_vocab) + 1

X_train = index_sents(X_train, X_vocab, vocab_size)
X_test = index_sents(X_test, X_vocab, vocab_size)


# truncate and pad input sequences
X_train = sequence.pad_sequences(X_train, maxlen=MAX_LENGTH)
X_test = sequence.pad_sequences(X_test, maxlen=MAX_LENGTH)


# one-hot encoding for output
# https://www.reddit.com/r/MachineLearning/comments/31fk7i/converting_target_indices_to_onehotvector/
y_train = np.eye(num_labels)[y_train]
y_test = np.eye(num_labels)[y_test]


# check data
print(X_train[0])
print(y_train[0])
print('')

'''
for small sentences, a deep RNN without any convolutional layers works well
'''
# create the model
model = Sequential()

# embedding layer - initialized randomly and trained with model
model.add(Embedding(MAX_VOCAB, EMBEDDING_SIZE, input_length=MAX_LENGTH))

# convolution and max pooling layers
model.add(Convolution1D(nb_filter=32, filter_length=3, border_mode='same', activation='relu'))
model.add(MaxPooling1D(pool_length=2))
model.add(Dropout(DROPOUT_RATE))

# RNN (LSTM) layers - can copy the first one for even deeper NN
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(DROPOUT_RATE))
model.add(LSTM(128))

# dense layer that outputs class
# apparently adding another Dense layer also helps classification
model.add(Dense(num_labels*3, activation='relu'))
model.add(Dense(num_labels, activation='softmax')) # sigmoid ok
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

# callback for early stoppage
earlystop = EarlyStopping(monitor=stop_monitor,
                          min_delta=stop_delta,
                          patience=stop_epochs,
                          verbose=1,
                          mode='auto')

callbacks_list = [earlystop] # add model checkpointing etc here

model.fit(X_train, y_train,
          validation_data=(X_test, y_test),
          nb_epoch=MAX_EPOCHS,
          batch_size=BATCH_SIZE,
          callbacks=callbacks_list
          )

model.save('trained_model.h5')
print("saved model to disk")

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=1)
print("\n\nAccuracy: %.2f%%" % (scores[1]*100), '\n')

# Final evaluation of the model
preds = model.predict(X_test, verbose=1)

print('\n')
for idx, pred in enumerate(preds[:10]):
    #for ytest & pred, we need INDEX of MAXIMUM value, hence the mess
    print(encoder.inverse_transform([list(y_test[idx]).index(max(y_test[idx]))])[0], "|",
          encoder.inverse_transform([list(pred).index(max(pred))])[0], " : ",
          test_sents[idx])
