# LSTM many-to-many sequence labeling network in Keras + Tensorflow
# Chain CRF model uses Kera fork here: https://github.com/fchollet/keras/pull/4621
# install with pip install git+https://github.com/phipleg/keras.git@crf

import codecs
import feature
import tensorflow as tf
from keras import backend as K
import numpy as np

from keras.models import Sequential
from keras.layers.wrappers import Bidirectional
from keras.layers import Activation, TimeDistributed, Dense, Dropout, recurrent, Embedding, ChainCRF
from keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam

# for custom embeddings
from gensim.models import Word2Vec
import multiprocessing
from embedding import create_embeddings, load_vocab

sess = tf.Session()
K.set_session(sess)

# specify data here:
corpus = 'datasets/junhyuk/sample3'

# Parameters for the model and dataset
TRAIN_PERCENT = 0.8     # for train-test-split
VOCAB_SIZE = 8000       # max size of 'vocab' (# of characters, here)
EMBEDDING_SIZE = 256    # embedding size
MAX_SEQ_LENGTH = 50     # max sequence length in characters (for padding/truncating)
HIDDEN_SIZE = 192       # LSTM Nodes/Features/Dimension
BATCH_SIZE = 32
DROPOUTRATE = 0.33
LAYERS = 3              # bi-LSTM-RNN layers
MAX_EPOCHS = 25         # max iterations, early stop condition below

resume = False          # nothing for now

# Parameters for embedding
do_embedding = True     # else will load pre-trained embeddings (see embedding.py)
W2V_MINCOUNT = 2        # minimum word count for word2vec
W2V_SG = 1
W2V_BATCHWORDS = 10000
W2V_ITERS = 100

# parameters for model saving etc
model_filename = "models/trained_model.h5"
model_filepath = "models/TEMP_weights-{epoch:02d}-{val_acc:.2f}.hdf5"  # for temp models
model_logdir = "models/"  # for t-board log files (not working atm)
stop_monitor = 'val_acc'  # variable for early stop: (default = val_loss)
stop_delta = 0.00         # minimum delta before early stop (default = 0)
stop_epochs = 1           # how many epochs to do after stop condition (default = 0)


# create word2vec embeddings from text data
if do_embedding == True:
    print('Generating embeddings...\n')
    vocab, model = create_embeddings(corpus,
                                     min_count=W2V_MINCOUNT,
                                     size=EMBEDDING_SIZE,
                                     sg=W2V_SG,
                                     batch_words=W2V_BATCHWORDS,
                                     iter=W2V_ITERS,
                                     workers=multiprocessing.cpu_count())

else:

    print("Loading embeddings...\n")
    # load vocab todo: does this work???
    vocab = load_vocab('PUT/THE/SAVE/DIRECTROY/HERE.file')
    # load embedding model todo: does this work???
    model = Word2Vec.load('PUT/THE/SAVE/DIRECTROY/HERE.file')


print('Generating data...\n')

# read file
f_train = codecs.open(corpus, 'rb', encoding='utf-8')
X_data = f_train.readlines()

# todo: split long strings into shorter ones??
# todo: for now, just truncate
# http://stackoverflow.com/questions/13673060/split-string-into-strings-by-length
def splitter(data):
    results = []
    max_size = 100
    for x in data:
        if len(x) > max_size:
            chunks = len(x)
            for y in [ x[i:i + max_size] for i in range(0, chunks, max_size) ]:
                results.append(y)
        else:
            results.append(y)
    return y


# tokenize by syllable: "안녕하세요" >> ['안', '녕', '하', '세', '요']
X_lines = []
for string in X_data:
    X_lines.append(list(string.replace(' ', '').strip()))

# get output sequence: 2 for whitespace, 1 for no whitespace
y_data = feature.text2labels(corpus)

# split y_data, as it is returned as single sequence
y_indices = []
start = 0
for sentence in X_lines:
    length = len(sentence)
    y_indices.append(y_data[start:start + length])
    start += length

# testing
print("X_train", len(X_lines[2]), X_lines[2])
print("y_train", len(y_indices[2]), y_indices[2])

# todo: fixed the indexing
# turn syllable lists into space-separated strings for CountVectorizer
X_tokens = []
for data in X_lines:
    X_tokens.append(' '.join(data))
X_lines = X_tokens

# get index vectors
# https://github.com/fchollet/keras/issues/17
sentvectorizer = CountVectorizer(analyzer='char', max_features=VOCAB_SIZE-1)
sentvectorizer.fit(X_lines)


print("splitting into train and test...\n")

# todo: fixed the indexing
X_train, X_test, y_train, y_test = train_test_split(X_tokens, y_indices, test_size=0.2)

# save strings for analyzing results
X_train_strings = X_train
X_test_strings = X_test

# todo: fixed the indexing
print("indexing sentences...\n")
def index_sents(data, vectorizer):
    vocab = vectorizer.vocabulary_
    results = []
    for sent in data:
        this = []
        for word in sent:
            this.append(vocab[word]+1)
        results.append(this)
    return(results)

X_train = index_sents(X_train, sentvectorizer)
X_test = index_sents(X_test, sentvectorizer)


print('Truncating & zero-padding the data...\n')
# pad sequences with zeros at END of sequence (if too short)
# cut off sequences over MAX_SEQ_LENGTH
X_train = sequence.pad_sequences(X_train, truncating='post', padding='post', maxlen=MAX_SEQ_LENGTH)
X_test = sequence.pad_sequences(X_test, truncating='post', padding='post', maxlen=MAX_SEQ_LENGTH)
y_train = sequence.pad_sequences(y_train, truncating='post', padding='post', maxlen=MAX_SEQ_LENGTH)
y_test = sequence.pad_sequences(y_test, truncating='post', padding='post', maxlen=MAX_SEQ_LENGTH)


print("Creating embedding matrix from pretrained embeddings...\n")
# embedding matrix, filled with zeros
embedding_matrix = np.zeros((VOCAB_SIZE, EMBEDDING_SIZE))
# for each word in the sentence vocabulary...
sentmapper = sentvectorizer.vocabulary_ # dictionary of ContVectorizer vocab : index entries
sentvocab = list(sentmapper.keys()) # input sentence vocab as list (from Countvectorizer)
                                    # can also use get_feature_names() here
bed_vocab = list(vocab.keys())      # embedding vect vocab as list (from gensim model)
for word in sentvocab:
    # get the word vector from the embedding model
    # need a check?
    if word in  bed_vocab:
        # get word vector
        word_vector = model[word]
        # slot it in at the proper index
        # todo: add one to index to account for zeroes!
        embedding_matrix[sentmapper[word] + 1] = word_vector


print('one-hot encode the labels...\n')
# one-hot encoding for label sequences
# one-hot encoding
# see RNN_seq2seq_demo
def encode(head, maxlabels):
    X = np.zeros((len(head), maxlabels))
    for idx, word_int in enumerate(head):
        X[idx, word_int] = 1
    return X

y_train_enc = [[] for i in range(len(y_train))]
for i, sentence in enumerate(y_train):
    y_train_enc[i] = encode(sentence, maxlabels=3)

y_test_enc = [[] for i in range(len(y_test))]
for i, sentence in enumerate(y_test):
    y_test_enc[i] = encode(sentence, maxlabels=3)

y_train = np.asarray(y_train_enc)
y_test = np.asarray(y_test_enc)


print('Saving data...(DISABLED FOR NOW)\n')
# http://stackoverflow.com/questions/20928136/input-and-output-numpy-arrays-to-h5py
# np.save(input_filepath+'X_train.npy', X_train)
# np.save(input_filepath+'X_test.npy', X_test)
# np.save(input_filepath+'y_train.npy', y_train)
# np.save(input_filepath+'y_test.npy', y_test)
# np.save(input_filepath+'sent_dict.npy', sent_vocab)
# np.save(input_filepath+'head_dict.npy', head_vocab)
# load like this:
# read_dictionary = np.load('my_file.npy').item()


print('final shapes')
print("X_train", np.shape(X_train))
print("y_train", np.shape(y_train))
print("X_test ", np.shape(X_test))
print("y_test ", np.shape(y_test))
print('')

print('Building model...\n')

# try LSTM, GRU, SimpleRNN
RNN = recurrent.LSTM

model = Sequential()

# add pre-trained embedding layer (allow it to be trained)
# todo: zero-masking not working
model.add(Embedding(VOCAB_SIZE,
                    EMBEDDING_SIZE,
                    weights=[embedding_matrix], # 'list by convention' -Ben Bolte
                    input_length=MAX_SEQ_LENGTH,
                    #mask_zero=True,
                    trainable=True))

# use this for no pretrained embeddings:
# model.add(Embedding(VOCAB_SIZE, EMBEDDING_SIZE, input_length=MAX_SEQ_LENGTH, mask_zero=True))

# bi-LSTM layers here, with dropout
for _ in range(LAYERS):
    model.add(Bidirectional(RNN(HIDDEN_SIZE, return_sequences=True)))
    model.add(Dropout(DROPOUTRATE))

# For each of step of the sequence, decide which label should be chosen
model.add(TimeDistributed(Dense(3))) # for labels yes, no and 'other' (zero-padding)

# for non-CRF:
# model.add(Activation('softmax'))
# model.compile(loss='categorical_crossentropy', # non-crf default: categorical_crossentropy
#               optimizer='adam', # default: adam
#               metrics=['accuracy'])

# for Chain CRF:
crf = ChainCRF()
model.add(crf)

# todo: if loading a trained model, uncomment this and comment below
# model.load_weights('models/trained_model.h5')

model.compile(loss=crf.loss, # non-crf default: categorical_crossentropy
              optimizer='adam', # default: adam
              metrics=['accuracy'])

# callbacks for saving, early stoppage, tboard (todo)
checkpoint = ModelCheckpoint(model_filepath,
                             monitor=stop_monitor,
                             verbose=1,
                             save_best_only=True,
                             mode='max')
earlystop = EarlyStopping(monitor=stop_monitor,
                          min_delta=stop_delta,
                          patience=stop_epochs,
                          verbose=1,
                          mode='auto')

callbacks_list = [checkpoint, earlystop]

model.summary()

# todo: comment the below if you are loading a trained model:
# todo: ########### COMMENT FROM HERE ############
# Train the model each generation and show predictions against the validation dataset
print('Train model...')
print()
print('-' * 50)
model.fit(X_train, y_train,
          validation_data=(X_test, y_test),
          batch_size=BATCH_SIZE,
          epochs=MAX_EPOCHS,
          # verbose=1
          callbacks=callbacks_list
          )

model.save(model_filename)
print("saved model to disk")
# todo: ########### COMMENT TO HERE ############


scores = model.evaluate(X_test, y_test, verbose=0)
print('')
print('Eval model...')
print("Accuracy: %.2f%%" % (scores[1] * 100), '\n')
print('')
preds = model.predict(X_test)
for idx, pred in enumerate(preds[:200]):
    print('sent', X_test_strings[idx])
    p = [list(a).index(max(list(a))) for a in pred]
    print('pred', p)
    t = [list(b).index(max(list(b))) for b in y_test[idx]]
    print('true', t)
    print('')