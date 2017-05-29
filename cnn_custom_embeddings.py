# this is a simple cnn classifier with embeddings
# it first uses gensim to create word embeddings
# then loads the pre-trained embeddings as first layer in NN

# on sample brown data: (embed size 96, dropout=0.5, train = 25 epochs)
# randomized,  trainable embeddings: 51.33%
# pre-trained, fixed     embeddings: 47.71%
# pre-trained, trainable embeddings: 56.56%

# dataset is probably too small to draw definitive conclusions, but it looks like
# starting with pretrained embeddings and allowing the network to further train them
# might be providing a little bit of a 'jump start' to performance

# tested with tensorflow-gpu==0.12 and Keras==1.2.2

# embedding references:
# http://ben.bolte.cc/blog/2016/keras-gensim-embeddings.html
# https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html

# network references:
# http://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/
# http://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/

import codecs, re
import numpy as np

# Scikit-Learn and NLTK for preprocessing
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

# embedding-specific imports
import multiprocessing
from gensim.models import Word2Vec
from embedding import create_embeddings, load_vocab, tokenize


# data and labels
sents_file = 'datasets/brown_sents.txt'
label_file = 'datasets/brown_topics.txt'


# embedding parameters
embeddings_path='model_embeddings.gensimmodel'
vocab_path='model_mapping.json'

do_embedding = True

EMBEDDING_SIZE = 96     # gensim embedding size
W2V_MINCOUNT = 2        # only consider words occuring this many times or more
W2V_BATCHWORDS = 10000  # words per training batch
W2V_ITERS = 25          # gensim training iterations


# network parameters
MAX_VOCAB = 18000
MAX_LENGTH = 50
BATCH_SIZE = 64
MAX_EPOCHS = 25
DROPOUT_RATE = 0.5

stop_monitor = 'val_loss'  # variable for early stop: (default = val_loss)
stop_delta = 0.0   # minimum delta before early stop (default = 0)
stop_epochs = 2    # how many epochs to do after stop condition (default = 0)


# create word2vec embeddings from text data
# get vocab as dict, gensim model
if do_embedding == True:
    print('Generating embeddings...\n')
    vocab, model = create_embeddings(sents_file,
                                     embeddings_path=embeddings_path,
                                     vocab_path=vocab_path,
                                     min_count=W2V_MINCOUNT,
                                     size=EMBEDDING_SIZE,
                                     sg=1,
                                     batch_words=W2V_BATCHWORDS,
                                     iter=W2V_ITERS,
                                     workers=multiprocessing.cpu_count())

else:

    print("Loading embeddings...\n")
    # load vocab todo: does this work???
    vocab, _ = load_vocab(vocab_path)
    # load embedding model todo: does this work???
    model = Word2Vec.load(embeddings_path)


# load the data.
print("Loading data...\n")

f_sents = codecs.open(sents_file, 'rb', encoding='utf8')
f_classes = codecs.open(label_file, 'rb', encoding='utf8')
sents = [sent.strip() for sent in f_sents.readlines()]
labels = [label.strip() for label in f_classes.readlines()]
# number of labels
num_labels = len(set(labels))

# fit vectorizers
print("Fitting tokenizer...\n")

# get count vectors
# https://github.com/fchollet/keras/issues/17
# MAX_VOCAB-1 because 0 for OOV/padding
sentvectorizer = CountVectorizer(tokenizer=tokenize, max_features=MAX_VOCAB-1)
sentvectorizer.fit(sents)

# prepare labels
print("Preparing labels...\n")
encoder = LabelEncoder()
labels = encoder.fit_transform(labels)

# generate new training data
print('generating training data...\n')
X_train, X_test, y_train, y_test = train_test_split(sents, labels, test_size=0.2)

train_sents = X_train
test_sents = X_test

X_train = sentvectorizer.transform(X_train)
# MAX_VOCAB-1 because 0 for OOV/padding
X_train =[row.indices + 1 for row in X_train]

X_test = sentvectorizer.transform(X_test)
# MAX_VOCAB-1 because 0 for OOV/padding
X_test =[row.indices + 1 for row in X_test]

# truncate and pad input sequences
X_train = sequence.pad_sequences(X_train, maxlen=MAX_LENGTH)
X_test = sequence.pad_sequences(X_test, maxlen=MAX_LENGTH)

# make embedding matrix:

# embedding matrix, filled with zeros
embedding_matrix = np.zeros((MAX_VOCAB, EMBEDDING_SIZE))

# for each word in the sentence vocabulary...
sentmapper = sentvectorizer.vocabulary_ # dictionary of ContVectorizer { vocab : index } entries
sentvocab = list(sentmapper.keys()) # input sentence vocab as list (from Countvectorizer)
                                    # can also use get_feature_names() here
bed_vocab = list(vocab.keys())      # embedding vect vocab as list (from gensim model)

# for each word in the data...
for word in sentvocab:
    # get the word vector from the embedding model
    # if it's there (check against vocab list)
    if word in  bed_vocab:
        # get the word vector
        word_vector = model[word]
        # slot it in at the proper index
        # add one to index to account for zeroes
        embedding_matrix[sentmapper[word] + 1] = word_vector
    # if word not there, it stays as zeros

# one-hot encoding for output
# https://www.reddit.com/r/MachineLearning/comments/31fk7i/converting_target_indices_to_onehotvector/
y_train = np.eye(num_labels)[y_train]
y_test = np.eye(num_labels)[y_test]

# check data
print("embedding data:")
print("should be all zeroes")
print(embedding_matrix[0])
print("should be non-zeroes")
print(embedding_matrix[2])
print("training data:")
print(X_train[0])
print("labels:")
print(y_train[0])
print(len(y_train[0]), "=", num_labels, "?")
print('')

# create the model
model = Sequential()

# embedding layer - initialized with gensim model weights
model.add(Embedding(MAX_VOCAB,
                    EMBEDDING_SIZE,
                    weights=[embedding_matrix], # 'list by convention' -Ben Bolte blog
                    input_length=MAX_LENGTH,
                    trainable=True))           # disallow them to be trained, can change this

# convolution and max pooling layers
model.add(Convolution1D(nb_filter=64, filter_length=5, border_mode='same', activation='relu'))
model.add(MaxPooling1D(pool_length=5))
model.add(Dropout(DROPOUT_RATE))
model.add(Convolution1D(nb_filter=64, filter_length=5, border_mode='same', activation='relu'))
model.add(MaxPooling1D(pool_length=5))
model.add(Dropout(DROPOUT_RATE))
model.add(LSTM(100))
model.add(Dropout(DROPOUT_RATE))
# dense layer(s) that outputs class
# apparently adding a second Dense layer helps performance
model.add(Dense(num_labels, activation='relu'))
model.add(Dense(num_labels, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

# callback for early stoppage
earlystop = EarlyStopping(monitor=stop_monitor,
                          min_delta=stop_delta,
                          patience=stop_epochs,
                          verbose=1,
                          mode='auto')

callbacks_list = [] # add earlystop, checkpointing etc here

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
