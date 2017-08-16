#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



features_list = ['poi','salary', 'total_stock_value', 'total_payments','exercised_stock_options','bonus',\
                 'shared_receipt_with_poi','expenses', 'from_this_person_to_poi','from_poi_to_this_person',\
                 'other', 'from_messages', 'to_messages', 'deferral_payments', 'restricted_stock',\
                 'restricted_stock_deferred', 'loan_advances','director_fees','deferred_income','long_term_incentive'\
                 ]

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

my_dataset = data_dict

# There are 146 employees
print len(my_dataset)

# Print all features for each employee and the total number of features except poi
count = 0
for features, feature_values in data_dict['METTS MARK'].iteritems():
    count += 1
    print features
print 'count:', count

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = False, remove_all_zeroes = False)
labels, features = targetFeatureSplit(data)
print 'data type:', type(data)
print len(features)
print len(labels)
print sum(labels)

# In order to easily explore my_dataset, I convert my_dataset to Pandas DataFrame.
df = pd.DataFrame(
    data = data,
    index = my_dataset.keys(),
    columns = features_list)

# Explore whether there are features have many missing values.
print 'sum of zero of salary of employee:', sum(df['salary'] == 0)
print 'sum of zero of bonus of employee:', sum(df['bonus'] == 0)
print 'sum of zero of total_stock_value:', sum(df['total_stock_value'] == 0)
print 'sum of zero of exercised_stock_value:', sum(df['exercised_stock_options'] == 0)
print 'sum of zero of deferral_payments:', sum(df['deferral_payments'] == 0)
print 'sum of zero of restricted_stock:', sum(df['restricted_stock'] == 0)

### Task 2: Exploring outliers
# Define a function to find outliers which have largest residual by using regression method

# Moving all zeros employees
zero_employee = []
for i in range(0, len(df.index)):
     if df.sum(axis = 1)[i] == 0:
         zero_employee.append(df.index[i])
print 'number of df', len(df.index)
print zero_employee
df.drop(zero_employee, inplace=True)
print len(df.index)

print 'total:',df.loc['TOTAL','salary']
print 'total salary:',(sum(df['salary']) - df.loc['TOTAL', 'salary'])

## Visualize features for employees
## Using boxplot to explore outliers. Here, We don't simply delet those outliers or fill in with statistical values.
plt.boxplot(df['salary'])
plt.xlabel('salary')
#plt.show()
plt.boxplot(df['bonus'])
plt.xlabel('bonus')
#plt.show()
plt.boxplot(df['total_payments'])
plt.xlabel('total_payments')
#plt.show()
plt.boxplot(df['total_stock_value'])
plt.xlabel('total_stock_value')
#plt.show()
# Boxplots grouped by poi
fig, ax = plt.subplots(figsize = (4, 4))
df.boxplot(['salary', 'total_payments', 'bonus', 'total_stock_value'], 'poi', ax)
#plt.show()

# Figure out which un-poi employee has this extreme values
unpoi_with_extreme_values_index = df.idxmax(axis = 0)['salary']
print unpoi_with_extreme_values_index

# Remove 'TOTAL' from df (DataFrame)
df.drop(unpoi_with_extreme_values_index, inplace = True)
print len(df)

fig, ax = plt.subplots(figsize = (4, 4))
df.boxplot(['salary', 'total_payments', 'bonus', 'total_stock_value'], 'poi', ax)
#plt.show()

### Create new feature(s)
## Create a fraction of sum of salary and bonus in total_payments
df['fraction_sum_salary_and_bonus_in_total_payments']=sum(df['salary'] + df['bonus']) / df['total_payments']
df['fraction_sum_salary_and_bonus_in_total_payments'].replace(np.inf, np.nan,inplace=True)
df['fraction_sum_salary_and_bonus_in_total_payments'].fillna(0, inplace=True)
#print'fraction_sum_salary_and_bonus_in_total_payments:', df['fraction_sum_salary_and_bonus_in_total_payments']


fig, ax = plt.subplots(figsize = (4, 4))
df.boxplot(['fraction_sum_salary_and_bonus_in_total_payments'], 'poi', ax)
#plt.show()

print df.idxmax(axis = 0)['fraction_sum_salary_and_bonus_in_total_payments']

# Fraction of poi_to_this_person which means how many emails this person received are coming from poi
df['fraction_email_poi_to_this_person'] = df['from_poi_to_this_person']/df['to_messages']
df['fraction_email_poi_to_this_person'].fillna(0, inplace=True)

#print df['fraction_email_poi_to_this_person']

# Fraction of email_to_poi which means how many emails this person sent are sent to poi.
df['fraction_email_to_poi'] = df['from_this_person_to_poi']/df['from_messages']
df['fraction_email_to_poi'].fillna(0,inplace=True)
print df['fraction_email_to_poi']


# Convert DataFrame to Dictionary after creating and exploring the data
my_dataset = df.to_dict(orient = 'index')
print 'my_dataset', len(my_dataset)

# Explore new features if have impacts on original features
# split my_dataset on original features_list
data = featureFormat(my_dataset, features_list, sort_keys = False)
labels, features = targetFeatureSplit(data)

# Split my_dataset into training dataset and testing dataset
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
# Using SelectKBest to explore scores and pvalues on original features_list
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(f_classif, k = 10)
selector.fit(features_train, labels_train)
print 'selector.scores with original features_list:',selector.scores_
print 'selector.pvalues with original features_list:', selector.pvalues_

# Update the feature_list
features_list = features_list + ['fraction_sum_salary_and_bonus_in_total_payments', 'fraction_email_poi_to_this_person',\
                                 'fraction_email_to_poi']

# # Feature Selection_SelectKBest BEGIN

data = featureFormat(my_dataset, features_list, sort_keys = False)
labels, features = targetFeatureSplit(data)

# Split data into training dataset and testing dataset
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
print 'len(labels_train)',len(labels_train)
print 'len(data)', len(data)

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import preprocessing

from sklearn import tree
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# Individually use SelectKBest() ['salary', 'total_stock_value', 'exercised_stock_options', 'bonus', 'expenses',\
# 'from_this_person_to_poi', 'director_fees', 'deferred_income', 'long_term_incentive', 'fraction_email_to_poi'] \
# selecting 10 features out of features_list
selector = SelectKBest(f_classif, k = 10)
selector.fit(features_train, labels_train)
print 'selector.scores:',selector.scores_
print 'selector.pvalues:', selector.pvalues_
# Get how many features have pvalues are less than significant level 0.1
k = sum(selector.pvalues_ <= 0.1)
print 'best k features:', k
new_features = ['poi']
for i in range(0,len(features_list[1:])):
    if selector.pvalues_[i] <= 0.1:
        new_features.append(features_list[i+1])
print 'new_feature:', new_features

# Using new_features to split dataset
data = featureFormat(my_dataset, new_features, sort_keys = False, remove_all_zeroes = False)
labels, features = targetFeatureSplit(data)

# Split data into training dataset and testing dataset
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

print len(data)
print len(new_features)
print len(labels_train)
# Using pipeline to tune parameters
# KNN algorithms
pipe = Pipeline([
    ('scale', preprocessing.StandardScaler()),
    ('select', SelectKBest(f_classif)),
    ('classify', KNeighborsClassifier(weights = 'uniform'))])

param_grid = {
    'select__k':[8],
    'classify__n_neighbors':[4, 5, 6, 8, 10]
}
clf_knn = GridSearchCV(pipe, param_grid)
clf_knn.fit(features_train, labels_train)
print 'best_estimator of KNN', clf_knn.best_params_
pred = clf_knn.predict(features_test)
print 'KNN classification report:', classification_report(labels_test, pred)
print 'KNN confusion matrix:', confusion_matrix(labels_test, pred)
print 'accuracy_score of grid of KNN:', accuracy_score(pred, labels_test)

## Decision_Tree
pipe = Pipeline([
    ('select', SelectKBest(f_classif)),
    ('classify',tree.DecisionTreeClassifier(random_state = 5)),
])
param_grid = {
    'select__k':[8],
    'classify__min_samples_split':range(2, 23)
}
sss = StratifiedShuffleSplit(random_state=42)
clf_tree = GridSearchCV(pipe, param_grid, scoring = 'f1', cv=sss)
clf_tree.fit(features_train, labels_train)
print 'best_estimator of decision_tress', clf_tree.best_params_
pred = clf_tree.predict(features_test)
print 'decision_tress classification report:', classification_report(labels_test, pred)
print 'decision tress confusion matrix:', confusion_matrix(labels_test, pred)
print 'accuracy_score of grid of decision tree:', accuracy_score(pred, labels_test)

### Logistic_Regression

pipe_lr = Pipeline([
    ('scale', preprocessing.StandardScaler()),
    ('select', SelectKBest(f_classif)),
    ('classify',LogisticRegression())
])

param_grid = {
    'select__k':[8],
    'classify__C':[0.0000001, 0.00000001, 0.0000000001,0.000001, 0.001],
    'classify__tol':[0.0001,0.001,0.01,0.1, 1, 2]
}
clf_lr = GridSearchCV(pipe_lr, param_grid)
clf_lr.fit(features_train, labels_train)
print 'best_estimator of logisticregression:', clf_lr.best_params_
pred = clf_lr.predict(features_test)
print 'LogisticRegression classification report:', classification_report(labels_test, pred)
print 'LogisticRegression confusion matrix:', confusion_matrix(labels_test, pred)
print 'accuracy_score of grid of LogisticRegression:', accuracy_score(pred, labels_test)
print type(features_train)
# After I had trained three algorithms on training dataset, I ended up using LogisticRegression.
#clf = clf_lr.best_params_
pipe = Pipeline([
    ('scale', preprocessing.StandardScaler()),
    ('select', SelectKBest(f_classif,k=8)),
    ('classify',LogisticRegression(tol = .0001, C = 0.0000001, penalty = 'l2'))
])
clf = pipe.fit(features_train, labels_train)
pred = clf.predict(features_test)
print 'LogisticRegression classification report:', classification_report(labels_test, pred)
print 'LogisticRegression confusion matrix:', confusion_matrix(labels_test, pred)
print 'accuracy_score of grid of LogisticRegression:', accuracy_score(pred, labels_test)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)