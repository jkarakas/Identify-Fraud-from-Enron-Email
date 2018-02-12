# %load ud120-projects/final_project/poi_id.py
#!/usr/bin/python

import sys
import pickle
import pandas as pd
import numpy as np

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

# Task 1: Select what features you'll use.
# features_list is a list of strings, each of which is a feature name.
# The first feature must be "poi".

features_list = ['poi']  # You will need to use more features

# Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
  data_dict = pickle.load(data_file)

# Convert dict to a Pandas Dataframe for easier manipulation and visualization.
df = pd.DataFrame.from_records(data_dict).T

# Convert the numeric columns to float type (NAN will be recoginzed as nan)
numeric_cols = [col for col in df.columns if col != 'email_address']
df[numeric_cols] = df[numeric_cols].astype(float)

# Convert NaN to nan for string columns
df.loc[df.email_address == 'NaN', 'email_address'] = np.nan

# Task 2: Remove outliers
df = df[(df.index != 'TOTAL')
        & (df.index != 'THE TRAVEL AGENCY IN THE PARK')
        & (df.index != 'LOCKHART EUGENE E')]

# Task 3: Create new feature(s)

# fraction_from_poi
df['fraction_from_poi'] = df.from_poi_to_this_person / df.to_messages
df['fraction_to_poi'] = df.from_this_person_to_poi / df.from_messages

# fill nan wth 0s
df = df.fillna(0)

# Convert the Dataframe back to a dict
data_dict = df.T.to_dict()

# Store to my_dataset for easy export below.
my_dataset = data_dict

# get all the features
features_list = ['poi']
features_list = features_list + [col for col in df.columns if col not in ['poi', 'email_address']]

from sklearn import preprocessing

# Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

scaler = preprocessing.MinMaxScaler()
features = scaler.fit_transform(features)

# intelligently select features (univariate feature selection)
from sklearn.feature_selection import SelectKBest, chi2

selector = SelectKBest(chi2, k=10)
selector.fit(features, labels)
scores = zip(features_list[1:], selector.scores_)
scores_list = sorted(scores, key=lambda x: x[1], reverse=True)

# Find the best number of features
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn import tree

n_features = np.arange(1, len(features_list))

## DecisionTreeClassifier
# Create a pipeline with feature selection and classification
pipe = Pipeline([
    ('select_features', SelectKBest(chi2)),
    ('classify', tree.DecisionTreeClassifier())
])

param_grid = [
    {
        'select_features__k': n_features
    }
]

# Use GridSearchCV to automate the process of finding the optimal number of features
tree_clf = GridSearchCV(pipe, param_grid=param_grid, scoring='f1', cv=5)
tree_clf.fit(features, labels)

tree_clf.best_params_

##RandomForestClassifier
# Create a pipeline with feature selection and classification
pipe = Pipeline([
    ('select_features', SelectKBest(chi2)),
    ('classify', RandomForestClassifier(max_depth=None,
                                        min_samples_split=2))
])

param_grid = [
    {
        'select_features__k': n_features
    }
]

# Use GridSearchCV to automate the process of finding the optimal number of features
forest_clf = GridSearchCV(pipe, param_grid=param_grid, scoring='f1', cv=5)
forest_clf.fit(features, labels)

forest_clf.best_params_

# Task 4: Try a varity of classifiers
# Please name your classifier clf for easy export below.
# Note that if you want to do PCA or other multi-stage operations,
# you'll need to use Pipelines. For more info:
# http://scikit-learn.org/stable/modules/pipeline.html

# Use the 9 first features of the KBest selector
features_list = ['poi'] + [score[0] for score in scores_list][:9]

# Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

scaler = preprocessing.MinMaxScaler()
features = scaler.fit_transform(features)


from sklearn.cross_validation import train_test_split
from sklearn import cross_validation
from sklearn.metrics import accuracy_score, precision_score, recall_score


def cross_validate(clf, name):
  '''Receives a classifier and the name of the classifier
  and performs cross_validation scoring on the features and labbels
  in the global scope.
  Returns a pandas DataFrame with the results'''

  accuracy_scores = cross_validation.cross_val_score(
      clf, features, labels, cv=5, scoring='accuracy')

  precision_scores = cross_validation.cross_val_score(
      clf, features, labels, cv=5, scoring='precision')

  recall_scores = cross_validation.cross_val_score(
      clf, features, labels, cv=5, scoring='recall')

  f1_scores = cross_validation.cross_val_score(
      clf, features, labels, cv=5, scoring='f1')

  accuracy = '%0.2f (+/- %0.2f)' % (accuracy_scores.mean(),
                                    accuracy_scores.std() * 2)

  precision = '%0.2f (+/- %0.2f)' % (precision_scores.mean(),
                                     precision_scores.std() * 2)

  recall = '%0.2f (+/- %0.2f)' % (recall_scores.mean(),
                                  recall_scores.std() * 2)

  f1 = '%0.2f (+/- %0.2f)' % (f1_scores.mean(),
                              recall_scores.std() * 2)

  return pd.DataFrame(index=[name], data={'Accuracy': [accuracy],
                                          'Precision': [precision],
                                          'Recall': [recall],
                                          'F1': [f1]})


df = pd.DataFrame()
n_features = len(features_list) - 1

from sklearn.naive_bayes import GaussianNB
clf1 = GaussianNB()
df1 = cross_validate(clf1, 'GaussianNB')


clf2 = tree.DecisionTreeClassifier()
df2 = cross_validate(clf2, 'DecisionTreeClassifier')
df = df1.append(df2)

from sklearn.ensemble import RandomForestClassifier
clf3 = RandomForestClassifier(n_estimators=10,
                              max_features=n_features,
                              max_depth=None,
                              min_samples_split=2)
df3 = cross_validate(clf3, 'RandomForestClassifier')
df = df.append(df3)

from sklearn.linear_model import LogisticRegression
clf4 = LogisticRegression(C=1e5)
df4 = cross_validate(clf4, 'LogisticRegression')
df = df.append(df4)

from sklearn.cluster import KMeans
clf5 = KMeans(n_clusters=2, random_state=0)
df5 = cross_validate(clf5, 'KMeans')
df = df.append(df5)

# Task 5: Tune your classifier to achieve better than .3 precision and recall
# using our testing script. Check the tester.py script in the final project
# folder for details on the evaluation method, especially the test_classifier
# function. Because of the small size of the dataset, the script uses
# stratified shuffle split cross validation. For more info:
# http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

from sklearn.model_selection import GridSearchCV
clf = tree.DecisionTreeClassifier()
# Define the configuration of parameters to test with the
# Decision Tree Classifier
param_grid = {'criterion': ['gini', 'entropy'],
              'min_samples_split': [2, 4, 6, 8, 10, 20],
              'max_depth': [None, 5, 10, 15, 20],
              'max_features': [None, 'sqrt', 'log2', 'auto']}

# Use GridSearchCV to find the optimal hyperparameters for the classifier
tree_clf = GridSearchCV(clf, param_grid=param_grid, scoring='f1', cv=5)
tree_clf.fit(features, labels)
# Get the best algorithm hyperparameters for the Decision Tree
tree_clf.best_params_

# A Dataframe to use in the reporting
df_final = cross_validate(tree_clf.best_estimator_, 'Tuned DecisionTreeClassifier')

# Tune the parameters for RandomForestClassifier
clf3 = RandomForestClassifier(max_depth=None,
                              min_samples_split=2)

param_grid = {
    "n_estimators": [9, 18, 27, 36],
    "max_depth": [None, 1, 5, 10, 15],
    "min_samples_leaf": [1, 2, 4, 6]}

# Use GridSearchCV to find the optimal hyperparameters for the classifier
forest_clf = GridSearchCV(clf3, param_grid=param_grid, scoring='f1', cv=5)
forest_clf.fit(features, labels)
# Get the best algorithm hyperparameters for the Decision Tree
forest_clf.best_params_

df_final = df_final.append(cross_validate(forest_clf.best_estimator_, 'Tuned RandomForestClassifier'))

# Use the DecisionTreeClassifier for tester.py
clf = tree_clf.best_estimator_


# Task 6: Dump your classifier, dataset, and features_list so anyone can
# check your results. You do not need to change anything below, but make sure
# that the version of poi_id.py that you submit can be run on its own and
# generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
