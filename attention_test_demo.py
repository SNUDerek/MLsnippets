import tensorflow as tf
from keras import backend as K
import numpy as np
from keras.models import Sequential
from keras.engine.training import _slice_arrays
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, recurrent
from keras.callbacks import EarlyStopping
sess = tf.Session()
K.set_session(sess)

from attention import Attention

# Parameters for the model and dataset
TRAINING_SIZE = 50000
VOCAB_SIZE = 12
DIGITS = 3
MAXLEN = DIGITS + 1 + DIGITS
INVERT = True
HIDDEN_SIZE = 200
BATCH_SIZE = 100
LAYERS = 2
MAX_EPOCHS = 5
# Try replacing with LSTM, GRU, or SimpleRNN
RNN = recurrent.LSTM

stop_monitor = 'val_acc'  # variable for early stop: (default = val_loss)
stop_delta = 0.0  # minimum delta before early stop (default = 0)
stop_epochs = 1    # how many epochs to do after stop condition (default = 0)

chars = '0123456789+ '

# class that creates one-hot vector representations of input sequences
class CharacterTable(object):
    '''
    Given a set of characters:
    + Encode them to a one hot integer representation
    + Decode the one hot integer representation to their character output
    + Decode a vector of probabilities to their character output
    '''
    def __init__(self, chars, maxlen):
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
        self.maxlen = maxlen

    def encode(self, C, maxlen=None):
        maxlen = maxlen if maxlen else self.maxlen
        X = np.zeros((maxlen, len(self.chars)))
        for i, c in enumerate(C):
            X[i, self.char_indices[c]] = 1
        return X

    def decode(self, X, calc_argmax=True):
        if calc_argmax:
            X = X.argmax(axis=-1)
        return ''.join(self.indices_char[x] for x in X)

# class to change colors of correct/incorrect answers
class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'

ctable = CharacterTable(chars, MAXLEN)

# data generation
questions = []
expected = []
seen = set()
print('Generating data...')
while len(questions) < TRAINING_SIZE:
    f = lambda: int(''.join(np.random.choice(list('0123456789')) for i in range(np.random.randint(1, DIGITS + 1))))
    a, b = f(), f()
    # Skip any addition questions we've already seen
    # Also skip any such that X+Y == Y+X (hence the sorting)
    key = tuple(sorted((a, b)))
    if key in seen:
        continue
    seen.add(key)
    # Pad the data with spaces such that it is always MAXLEN
    q = '{}+{}'.format(a, b)
    query = q + ' ' * (MAXLEN - len(q))
    ans = str(a + b)
    # Answers can be of maximum size DIGITS + 1
    ans += ' ' * (DIGITS + 1 - len(ans))
    if INVERT:
        query = query[::-1]
    questions.append(query)
    expected.append(ans)
print('Total addition questions:', len(questions))

# data vectorization
print('Vectorization...')
X = np.zeros((len(questions), MAXLEN, len(chars)), dtype=np.bool)
y = np.zeros((len(questions), DIGITS + 1, len(chars)), dtype=np.bool)
for i, sentence in enumerate(questions):
    X[i] = ctable.encode(sentence, maxlen=MAXLEN)
for i, sentence in enumerate(expected):
    y[i] = ctable.encode(sentence, maxlen=DIGITS + 1)

# Shuffle (X, y) in unison as the later parts of X will almost all be larger digits
indices = np.arange(len(y))
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

# Explicitly set apart 20% for validation data that we never train over
split_at = int(len(X) - len(X) / 20)
(X_train, X_val) = (_slice_arrays(X, 0, split_at), _slice_arrays(X, split_at))
(y_train, y_val) = (y[:split_at], y[split_at:])

print(X_train.shape)
print(y_train.shape)


print('Build model...')

# from https://gist.github.com/rouseguy/1122811f2375064d009dac797d59bae9
model = Sequential()
# "Encode" the input sequence using an RNN, producing an output of HIDDEN_SIZE
# note: in a situation where your input sequences have a variable length,
# use input_shape=(None, nb_feature).
model.add(RNN(HIDDEN_SIZE, return_sequences=True, input_shape=(MAXLEN, len(chars))))
# todo : for trying attention
model.add(RNN(HIDDEN_SIZE, return_sequences=True))
model.add(Attention())
# model.add(RNN(HIDDEN_SIZE))
# For the decoder's input, we repeat the encoded input for each time step
model.add(RepeatVector(DIGITS + 1))
# The decoder RNN could be multiple layers stacked or a single layer
for _ in range(LAYERS):
    model.add(RNN(HIDDEN_SIZE, return_sequences=True))

# For each of step of the output sequence, decide which character should be chosen
model.add(TimeDistributed(Dense(len(chars))))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', # default: categorical_crossentropy
              optimizer='adam', # default: adam, can try SGD, RMSprop
              metrics=['accuracy'])

model.summary()

# callback for early stoppage
earlystop = EarlyStopping(monitor=stop_monitor,
                          min_delta=stop_delta,
                          patience=stop_epochs,
                          verbose=1,
                          mode='auto')

callbacks_list = [earlystop]

# predict_classes replacement
def predict_classes(inpoot):
    # http://stackoverflow.com/questions/6771428/most-efficient-way-to-reverse-a-numpy-array
    # get maximums
    outpoot = np.argmax(inpoot, axis=2)[::-1]
    return(outpoot)

# Train the model each generation and show predictions against the validation dataset
model.fit(X_train, y_train,
          validation_data=(X_val, y_val),
          batch_size=BATCH_SIZE,
          nb_epoch=MAX_EPOCHS,
          # callbacks=callbacks_list,
          verbose=1)

###
# Select 10 samples from the validation set at random so we can visualize errors
# Final evaluation of the model
scores = model.evaluate(X_val, y_val, verbose=1)
print('')
print('Eval model...')
print("Accuracy: %.2f%%" % (scores[1] * 100), '\n')
for i in range(10):
    ind = np.random.randint(0, len(X_val))
    rowX, rowy = X_val[np.array([ind])], y_val[np.array([ind])]
    preds = model.predict(rowX, verbose=1)
    preds = predict_classes(preds)

    q = ctable.decode(rowX[0])
    correct = ctable.decode(rowy[0])
    guess = ctable.decode(preds[0], calc_argmax=False)
    print('Q', q[::-1] if INVERT else q)
    print('T', correct)
    print(colors.ok + '☑' + colors.close if correct == guess else colors.fail + '☒' + colors.close, guess)
    print('---')
