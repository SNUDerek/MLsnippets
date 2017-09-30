import numpy as np
from mlpy import LinearRegression, ZeroRuleRegression, accuracy_score, train_test_split

# x = [1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 8, 9, 10]
# y = [2, 3, 4, 5, 5, 5, 6, 7, 8, 9, 9, 10, 11]
#
# zero_md = ZeroRuleRegression(mode='median')
#
# zero_mn = ZeroRuleRegression(mode='mean')
#
# zero_md.fit(x, y)
# zero_mn.fit(x, y)
#
# print(zero_md.predict(x, y))
# print(accuracy_score(zero_md.predict(x, y), y))
# print(zero_mn.predict(x, y))
# print(accuracy_score(zero_mn.predict(x, y), y))



# generate some dummy data

print("generating dummy data...\n")

feature1 = np.linspace(1.0, 10.0, 500)[:, np.newaxis]
feature2 = np.linspace(10.0, 1.0, 500)[:, np.newaxis]

y_data = np.sin(feature1) + 0.1*np.power(feature2,2) + 0.5*np.random.randn(500,1)

# normalize features
feature1 /= np.max(feature1)
feature2 /= np.max(feature2)

x_data = np.hstack((feature1, feature2))

X_train, X_test, y_train, y_test = train_test_split(x_data, y_data)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

print("fitting classifier...\n")

lrgs = LinearRegression(epochs=5000, lr=0.005)

lrgs.fit(X_train, y_train, verbose=True)

preds = lrgs.predict(X_test)

for i in range(len(preds)):
    print(preds[i], y_test[i])

print("train error", lrgs.error(X_train, y_train))
print("test  error", lrgs.error(X_test, y_test))


