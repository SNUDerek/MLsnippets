# # http://maciejjaskowski.github.io/2016/01/22/pandas-scikit-workflow.html

import csv as csv
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.metrics import accuracy_score
from sklearn_pandas import DataFrameMapper

# read in data
df_train = pd.read_csv('datasets/titanic/train.csv', header=0, index_col='PassengerId')
df_test = pd.read_csv('datasets/titanic/test.csv', header=0, index_col='PassengerId')
# concatenate the data
df = pd.concat([df_train, df_test], keys=["train", "test"])

# preprocessing
# new features from names
df['Title'] = df['Name'].apply(lambda c: c[c.index(',') + 2 : c.index('.')])
df['LastName'] = df['Name'].apply(lambda n: n[0:n.index(',')])
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# replace missing data with the mode of the column
df.loc[df['Embarked'].isnull(), 'Embarked'] = df['Embarked'].mode()[0]
df.loc[df['Fare'].isnull(), 'Fare'] = df['Fare'].mode()[0]

# features for family
df['FamilyID'] = df['LastName'] + ':' + df['FamilySize'].apply(str)
df.loc[df['FamilySize'] <= 2, 'FamilyID'] = 'Small_Family'

# replace missing age data with median by title

df['AgeOriginallyNaN'] = df['Age'].isnull().astype(int)
medians_by_title = pd.DataFrame(df.groupby('Title')['Age'].median()) \
  .rename(columns = {'Age': 'AgeFilledMedianByTitle'})

df = df.merge(medians_by_title, left_on = 'Title', right_index = True) \
  .sort_index(level = 0).sort_index(level = 1)

# reseparate using keys
df_train = df.ix['train']
df_test  = df.ix['test']

# use labelbinarizer to create dummy variables for categorical data
# these are one-hot ([0/1] if binary)
def featurize(features):
    transformations = [
                    ('Embarked', LabelBinarizer()),
                    ('Fare', None),
                    ('Parch', None),
                    ('Pclass', LabelBinarizer()),
                    ('Sex', LabelBinarizer()),
                    ('SibSp', None),
                    ('Title', LabelBinarizer()),
                    ('FamilySize', None),
                    ('FamilyID', LabelBinarizer()),
                    ('AgeOriginallyNaN', None),
                    ('AgeFilledMedianByTitle', None)]

    return DataFrameMapper(filter(lambda x: x[0] in df.columns, transformations))

# random forest classifier:

features = ['Sex', 'Title', 'FamilySize', 'AgeFilledMedianByTitle',
            'Embarked', 'Pclass', 'FamilyID', 'AgeOriginallyNaN']

pipeline = Pipeline([
                    ('featurize', featurize(features)),
                    ('forest', RandomForestClassifier())
                    ])

X_train = df_train[df_train.columns.drop('Survived')]
y_train = df_train['Survived']

model = pipeline.fit(X_train, y_train)

predictions = model.predict(df_test)

print("random forest with", features)
print("accuracy:", accuracy_score(df_test['Survived'], predictions))