input = [[0, 0, 0], # 0
         [0, 0, 1], # 1
         [0, 1, 0], # 2
         [0, 1, 1], # 3
         [1, 0, 0], # 4
         [1, 0, 1], # 5
         [1, 1, 0], # 6
         [1, 1, 1]] # 7

output = [[0, 0, 1], # 1
          [0, 1, 0], # 2
          [0, 1, 1], # 3
          [1, 0, 0], # 4
          [1, 0, 1], # 5
          [1, 1, 0], # 6
          [1, 1, 1], # 7
          [0, 0, 0]] # 0


from keras.models import Sequential
from keras.layers import Dense

# make a model
model = Sequential()

# add a Dense layer with sigmoid activation
model.add(Dense(3, input_shape=(3,), activation='sigmoid'))
model.add(Dense(3, activation='sigmoid'))
model.add(Dense(3, activation='sigmoid'))

# specified loss and "minimization"
model.compile(loss='binary_crossentropy', optimizer='adam')

model.summary()

# training
model.fit(input, output, epochs=10000, verbose=0)

# prediction
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