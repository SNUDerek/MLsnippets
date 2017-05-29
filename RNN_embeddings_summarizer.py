# todo: BASIC SEQ2SEQ (done)
# https://github.com/fchollet/keras/blob/master/examples/addition_rnn.py
# transfering_hidden_states_in_sequence_to_sequence/
# https://github.com/fchollet/keras/issues/2654

# todo: EMBEDDINGS (done, needs validation)
# https://radimrehurek.com/gensim/models/word2vec.html
# http://ben.bolte.cc/blog/2016/keras-gensim-embeddings.html
# https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html

# todo: ATTENTION (doing)
# https://github.com/fchollet/keras/issues/4962
# https://gist.github.com/cbaziotis/6428df359af27d58078ca5ed9792bd6d
# https://gist.github.com/nigeljyng/37552fb4869a5e81338f82b338a304d3
# alternate implementations:
# https://github.com/fchollet/keras/issues/1472
# https://github.com/codekansas/keras-language-modeling/blob/master/attention_lstm.py
# https://gist.github.com/mbollmann/ccc735366221e4dba9f89d2aab86da1e
# http://ben.bolte.cc/blog/2016/keras-language-modeling.html
# https://github.com/shyamupa/snli-entailment

# todo: BEAM SEARCH DECODE

import tensorflow as tf
from keras import backend as K
import numpy as np
from gensim.models import Word2Vec
import multiprocessing

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from keras.engine import Layer
from keras.models import Sequential, Model
from keras.layers import Activation, TimeDistributed, Dense, Dropout, Merge, Permute, Lambda, LSTM, RepeatVector, recurrent, Embedding
from keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

import dataset
from embedding import create_embeddings, load_vocab

sess = tf.Session()
K.set_session(sess)


# Parameters for embedding
do_embedding = True # else will load from previous
W2V_MINCOUNT = 2
EMBEDDING_SIZE = 256
W2V_SG = 1
W2V_BATCHWORDS = 10000
W2V_ITERS = 50

# Parameters for the model and dataset
corpus = 'reuters/reuters_reversed_filtered.txt'
TRAIN_PERCENT = 0.75
VOCAB_SIZE = 15000
MAX_ART_LENGTH = 60
MAX_HDL_LENGTH = 12
INVERT = True # not being used at the moment
HIDDEN_SIZE = 256
BATCH_SIZE = 32
DROPOUTRATE = 0.25
LAYERS = 3
MAX_EPOCHS = 500

# for RNN vs dense middle layer
DENSE_MIDDLE = True

embeddings_path='temp_embeddings/embeddings.gensim'
vocab_path='temp_embeddings/mapping.json'

# try LSTM, GRU, SimpleRNN... AttentionLSTM?
RNN = recurrent.LSTM

resume = False


# parameters for model saving etc
model_filename = "reuters_reversed_128x3_DEBUG.h5"
input_filepath = "temp_input/"
model_filepath = "temp_models/weights-{epoch:02d}-{val_acc:.2f}.hdf5"  # for temp saving
model_logdir = "temp_logs/"
stop_monitor = 'val_loss'  # variable for early stop: (default = val_loss)
stop_delta = 0.01  # minimum delta before early stop (default = 0)
stop_epochs = 1  # how many epochs to do after stop condition (default = 0)

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
    vocab = load_vocab(vocab_path)
    # load embedding model todo: does this work???
    model = Word2Vec.load(embeddings_path)

'''
Given a set of characters:
+ Encode them to a one hot integer representation
+ Decode the one hot integer representation to their character output
+ Decode a vector of probabilities to their character output
'''

print('Generating data...\n')
# http://ben.bolte.cc/blog/2016/keras-gensim-embeddings.html

sents, heads = dataset.get_texts(corpus)

print('sents and heads')
print(sents[0])
print(heads[0])

# get count vectors
# https://github.com/fchollet/keras/issues/17
# todo: new: subtract one to index to account for zero (unseen)
sentvectorizer = CountVectorizer(max_features=VOCAB_SIZE-1)
headvectorizer = CountVectorizer(max_features=VOCAB_SIZE-1)

sent_vocab = sentvectorizer.fit(sents)
head_vocab = headvectorizer.fit(heads)

# todo: new: add one to index to account for zero
X_sents = sentvectorizer.transform(sents)
X_sents =[row.indices + 1 for row in X_sents]

y_heads = headvectorizer.transform(heads)
y_heads = [row.indices + 1 for row in y_heads]

# print('vectorized sents and heads')
# print(X_sents[0])
# print(y_heads[0])

X_train, X_test, y_train, y_test = train_test_split(X_sents, y_heads, test_size=1-TRAIN_PERCENT)

print('Indexing data...\n')
# pad sequences
X_train = sequence.pad_sequences(X_train, maxlen=MAX_ART_LENGTH, padding='pre')
X_test = sequence.pad_sequences(X_test, maxlen=MAX_ART_LENGTH, padding='pre')
y_train = sequence.pad_sequences(y_train, maxlen=MAX_HDL_LENGTH, padding='post')
y_test = sequence.pad_sequences(y_test, maxlen=MAX_HDL_LENGTH, padding='post')

# todo: put this embedding_matrix code in embedding.py
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

print("this should be all zeroes:")
print(embedding_matrix[0])
print('')

# one-hot encoding
# see RNN_seq2seq_demo
def encode(head, maxvocab):
    X = np.zeros((len(head), maxvocab))
    for idx, word_int in enumerate(head):
        X[idx, word_int] = 1
    return X

y_train_enc = np.zeros((len(X_train), MAX_HDL_LENGTH, VOCAB_SIZE), dtype=np.bool)
for i, sentence in enumerate(y_train):
    y_train_enc[i] = encode(sentence, maxvocab=VOCAB_SIZE)

y_test_enc = np.zeros((len(X_test), MAX_HDL_LENGTH, VOCAB_SIZE), dtype=np.bool)
for i, sentence in enumerate(y_test):
    y_test_enc[i] = encode(sentence, maxvocab=VOCAB_SIZE)

y_train_debug = y_train
y_train = y_train_enc
y_test = y_test_enc

print('Saving data...\n')
# http://stackoverflow.com/questions/20928136/input-and-output-numpy-arrays-to-h5py
np.save(input_filepath+'X_train.npy', X_train)
np.save(input_filepath+'X_test.npy', X_test)
np.save(input_filepath+'y_train.npy', y_train)
np.save(input_filepath+'y_test.npy', y_test)
np.save(input_filepath+'sent_dict.npy', sent_vocab)
np.save(input_filepath+'head_dict.npy', head_vocab)
# load like this:
# read_dictionary = np.load('my_file.npy').item()

print('final sents and heads')
print(np.shape(X_train), X_train[0])
print(np.shape(y_train_debug), y_train_debug[0])
print(np.shape(y_train), y_train[0])
print(np.shape(X_test), X_test[0])
print(np.shape(y_test), y_test[0])
print('')

print('Building model...\n')

model = Sequential()

# model.add(Embedding(VOCAB_SIZE, EMBEDDING_SIZE, input_length=MAX_ART_LENGTH, mask_zero=True))
# output: shape (None, MAX_ART_LENGTH, EMBEDDING_SIZE) where None = batch size
# input: shape (None, input_length=MAX_ART_LENGTH, input_dim=EMBEDDING SIZE)
model.add(Embedding(VOCAB_SIZE,
                    EMBEDDING_SIZE,
                    weights=[embedding_matrix], # 'list by convention' -Ben Bolte
                    input_length=MAX_ART_LENGTH,
                    mask_zero=True,
                    trainable=True))

for _ in range(LAYERS-1):
    model.add(RNN(HIDDEN_SIZE, return_sequences=True))
    model.add(Dropout(DROPOUTRATE))

# old code using simple Dense layer and/or straight RNN
# to use, change above to LAYERS - 1 and delete attention layer
# add dense layer here?? this is bilinear attention??
# https://www.reddit.com/r/MachineLearning/comments/3sexmi/transfering_hidden_states_in_sequence_to_sequence/
if DENSE_MIDDLE == True:
    model.add(RNN(HIDDEN_SIZE, return_sequences=False))
    model.add(Dropout(DROPOUTRATE))
    model.add(Dense(HIDDEN_SIZE))
    model.add(Activation('relu'))
else:
    model.add(RNN(HIDDEN_SIZE))
# copy embedding for each output timestep
model.add(RepeatVector(MAX_HDL_LENGTH))

# decoder
for _ in range(LAYERS):
    model.add(RNN(HIDDEN_SIZE, return_sequences=True))
    model.add(Dropout(DROPOUTRATE))

# For each of step of the output sequence, decide which word should be chosen
model.add(TimeDistributed(Dense(VOCAB_SIZE)))

model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', # default: categorical_crossentropy
              optimizer='adam', # default: adam
              metrics=['accuracy'])

# callbacks for saving, early stoppage, tensorboard
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
tboard = TensorBoard(log_dir=model_logdir,
                     histogram_freq=0,
                     write_graph=True,
                     write_images=False)
tboard.set_model(model)

callbacks_list = [checkpoint, earlystop, tboard]

model.summary()

# Train the model each generation and show predictions against the validation dataset
print('Train model...')
# for iteration in range(1, 2):
print()
print('-' * 50)
# print('Iteration', iteration)
model.fit(X_train, y_train,
          validation_data=(X_test, y_test),
          batch_size=BATCH_SIZE,
          nb_epoch=MAX_EPOCHS,
          verbose=1
          # callbacks=callbacks_list
          )

model.save(model_filename)
print("saved model to disk")

scores = model.evaluate(X_test, y_test, verbose=0)
print('')
print('Eval model...')
print("Accuracy: %.2f%%" % (scores[1] * 100), '\n')
