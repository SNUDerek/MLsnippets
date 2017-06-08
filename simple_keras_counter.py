from keras.models import Sequential
from keras.layers import Dense

input = [[0, 0, 0],
         [0, 0, 1],
         [0, 1, 0],
         [0, 1, 1],
         [1, 0, 0],
         [1, 0, 1],
         [1, 1, 0],
         [1, 1, 1]]

output = [[0, 0, 1],
          [0, 1, 0],
          [0, 1, 1],
          [1, 0, 0],
          [1, 0, 1],
          [1, 1, 0],
          [1, 1, 1],
          [0, 0, 0]]

model = Sequential()

# start with single Dense, add others
model.add(Dense(3, input_shape=(3,), activation='sigmoid'))
model.add(Dense(3, activation='sigmoid'))
model.add(Dense(3, activation='sigmoid'))

# start with MSE and SGD
model.compile(loss='binary_crossentropy', #mse or binary_crossentropy
              optimizer='adam') # adam, adadelta for good results

model.summary()

model.fit(input, output, epochs=10000, verbose=0)

preds = model.predict(input)

def decoder(vect):
    answer = []
    for item in vect.tolist():
        if item > 0.5:
            answer.append(1)
        else:
            answer.append(0)
    return answer

for idx in range(len(input)):
    print("input:", input[idx], "true:", output[idx], "pred:", decoder(preds[idx]))
