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
df['fraction_sum_salary_bonus_in_total_payments']=sum(df['salary'] + df['bonus']) / df['total_payments']
df['fraction_sum_salary_bonus_in_total_payments'].replace(np.inf, np.nan,inplace=True)
df['fraction_sum_salary_bonus_in_total_payments'].fillna(0, inplace=True)
#print'fraction_sum_salary_bonus_in_total_payments:', df['fraction_sum_salary_bonus_in_total_payments']


#print df['salary_bonus_fraction']

fig, ax = plt.subplots(figsize = (4, 4))
df.boxplot(['fraction_sum_salary_bonus_in_total_payments'], 'poi', ax)
#plt.show()

print df.idxmax(axis = 0)['fraction_sum_salary_bonus_in_total_payments']

# Fraction of poi_to_this_person
df['fraction_email_poi_to_this_person'] = df['from_poi_to_this_person']/df['to_messages']
df['fraction_email_poi_to_this_person'].fillna(0, inplace=True)

#print df['fraction_email_poi_to_this_person']

# Fraction of email_to_poi
df['fraction_email_to_poi'] = df['from_this_person_to_poi']/df['from_messages']
df['fraction_email_to_poi'].fillna(0,inplace=True)
print df['fraction_email_to_poi']


# Convert DataFrame to Dictionary after creating and exploring the data
my_dataset = df.to_dict(orient = 'index')
print 'my_dataset', len(my_dataset)

# Update the feature_list
features_list = features_list + ['fraction_sum_salary_bonus_in_total_payments', 'fraction_email_poi_to_this_person',\
                                 'fraction_email_to_poi']

# # Feature Selection_SelectKBest BEGIN

data = featureFormat(my_dataset, features_list, sort_keys = False)
labels, features = targetFeatureSplit(data)

# Split data into training dataset and testing dataset
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

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
list_bool = selector.get_support()
new_features = []
for bool, feature in zip(list_bool, features_list[1:]):
    if bool:
        new_features.append(feature)

print 'new_features:', new_features

# Using pipeline to tune parameters
# KNN algorithms
pipe = Pipeline([
    ('scale', preprocessing.StandardScaler()),
    ('select', SelectKBest(f_classif)),
    ('classify', KNeighborsClassifier(weights = 'uniform'))])

param_grid = {
    'select__k':range(3, 23),
    'classify__n_neighbors':[4, 5, 6, 8, 10]
}
clf_knn = GridSearchCV(pipe, param_grid)
clf_knn.fit(features_train, labels_train)
print 'best_estimator of KNN', clf_knn.best_params_
pred = clf_knn.predict(features_test)
print 'KNN classification report:', classification_report(labels_test, pred)
print 'KNN confusion matrix:', confusion_matrix(labels_test, pred)
print 'accuracy_score of grid of KNN:', accuracy_score(pred, labels_test)

### Decision_Tree
pipe = Pipeline([
    ('select', SelectKBest(f_classif)),
    ('classify',tree.DecisionTreeClassifier(random_state = 5)),
])
param_grid = {
    'select__k':range(3, 23),
    'classify__min_samples_split':range(2, 23)
}
sss = StratifiedShuffleSplit()
clf_tree = GridSearchCV(pipe, param_grid, scoring = 'f1', cv=sss)
clf_tree.fit(features_train, labels_train)
print 'best_estimator of decision_tress', clf_tree.best_params_
pred = clf_tree.predict(features_test)
print 'decision_tress classification report:', classification_report(labels_test, pred)
print 'decision tress confusion matrix:', confusion_matrix(labels_test, pred)
print 'accuracy_score of grid of decision tree:', accuracy_score(pred, labels_test)
print 'total_number_of_poi_test:', sum(labels_test)

### Logistic_Regression
pipe_lr = Pipeline([
    ('scale', preprocessing.StandardScaler()),
    ('select', SelectKBest(f_classif)),
    ('classify',LogisticRegression())
])

param_grid = {
    'select__k':range(3, 23),
    'classify__C':[100, 10, 1, 0.01]
}
clf_lr = GridSearchCV(pipe_lr, param_grid)
clf_lr.fit(features_train, labels_train)
print 'best_estimator of logisticregression:', clf_lr.best_params_
pred = clf_lr.predict(features_test)
print 'LogisticRegression classification report:', classification_report(labels_test, pred)
print 'LogisticRegression confusion matrix:', confusion_matrix(labels_test, pred)
print 'accuracy_score of grid of LogisticRegression:', accuracy_score(pred, labels_test)
print 'total_number_of_poi_test:', sum(labels_test)
# #print 'clf_lr.best_score', clf_lr.cv_results_
# #print 'clf_lr.best_index:', clf_lr.best_index_
# support = clf_lr.named_steps['select'].get_support()
#
# new_features_lr = []
# for bool, feature in zip(support, features_list[1:]):
#     if bool:
#         new_features_lr.append(feature)
#
# print 'new_features:', new_features_lr

## Feature Selection_SelectKBest END

# #PCA BEGIN
# print the first people's information to check the df dataframe
# print df.iloc[0]
#
# features_list = ['poi','salary', 'total_stock_value', 'total_payments','exercised_stock_options','bonus',\
#                  'shared_receipt_with_poi','expenses', 'from_this_person_to_poi','from_poi_to_this_person',\
#                  'other', 'from_messages', 'to_messages', 'deferral_payments', 'restricted_stock',\
#                  'restricted_stock_deferred', 'loan_advances','director_fees','deferred_income','long_term_incentive'\
#                  ] # You will need to use more features
#
# data = featureFormat(my_dataset, features_list, sort_keys = True)
# labels, features = targetFeatureSplit(data)
#
# from sklearn.cross_validation import train_test_split
# features_train, features_test, labels_train, labels_test = \
#     train_test_split(features, labels, test_size=0.3, random_state=42)
#
# print 'type of features:', type(features)
#
# ### PCA
# from sklearn.decomposition import PCA
# from sklearn import preprocessing
# features_train_scaled = preprocessing.scale(features_train)
# features_test_scaled = preprocessing.scale(features_test)
# pca = PCA(n_components = 19)
# pca.fit(features_train_scaled)
# print 'pca:', pca
# print 'pca_explained_variance_ratio:',pca.explained_variance_ratio_
# first_pc = pca.components_[0]
# second_pc = pca.components_[1]
# print 'pc[0]:', first_pc
# print 'pc[1]:', second_pc
# # Cumulative Variance explains
# cumulative_variance_explained = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
# print 'cumulative_variace_explained:', cumulative_variance_explained
# plt.plot(cumulative_variance_explained)
# plt.show()

# #pca = PCA(n_components = 15)
# pca = PCA(n_components = 15)
#
# pca.fit(features_train_scaled)
# features_train_transformed = pca.fit_transform(features_train_scaled)
# features_test_transformed = pca.fit_transform(features_test_scaled)
#
# ### Using Pipeline
# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import GridSearchCV
# from sklearn import tree
# from sklearn import svm
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import accuracy_score
#
#
# #clf = tree.DecisionTreeClassifier()
# svr = svm.SVC()
# # #pipe = Pipeline(steps = [('pca', PCA()), ('clf', svm)])
# #
# #svm_parameters
# parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10], 'gamma':[0.2, 0.5, 0.8, 1, 10]}
# clf = GridSearchCV(svr, parameters)
# clf.fit(features_train_transformed, labels_train)
# print 'bets_estimator of SVM',clf.best_params_
# pred = clf.predict(features_test_transformed)
# print classification_report(labels_test, pred)
# print confusion_matrix(labels_test, pred)
# print 'accuracy_score of grid of SVM:', accuracy_score(pred, labels_test)

# # decision tree parameters
# parameters = {'min_samples_split':[2, 3, 4, 5, 6], 'random_state':[0, 4]}
# clf = GridSearchCV(clf, parameters)
# clf.fit(features_train_transformed, labels_train)
# print 'best_estimator of decision_tress', clf.best_params_
# pred = clf.predict(features_test_transformed)
# print 'decision_tress classification report:', classification_report(labels_test, pred)
# print 'decision tress confusion matrix:', confusion_matrix(labels_test, pred)
# print 'accuracy_score of grid of decision tree:', accuracy_score(pred, labels_test)
# print 'total_number_of_poi_test:', sum(labels_test)
#
# clf = tree.DecisionTreeClassifier(min_samples_split = 3, random_state = 0)
#
# clf.fit(features_train_transformed, labels_train)
# pred = clf.predict(features_test_transformed)
# pred = clf.predict(features_test_transformed)
# print 'decision_tress classification report:', classification_report(labels_test, pred)
# print 'decision tress confusion matrix:', confusion_matrix(labels_test, pred)
# print 'accuracy_score of grid of decision tree:', accuracy_score(pred, labels_test)
#
# # # KNN parameters
# # nbrs = KNeighborsClassifier(weights ='uniform')
# # parameters = {'n_neighbors':[4,5, 6, 8, 10]}
# # nbrs = GridSearchCV(nbrs, parameters)
# # nbrs.fit(features_train_transformed, labels_train)
# # print 'best_estimator of KNN:', nbrs.best_params_
# # pred = nbrs.predict(features_test_transformed)
# # print 'KNN classification report:', classification_report(labels_test, pred)
# # print 'KNN confusion matrix:', confusion_matrix(labels_test, pred)
# # print 'accuracy_score of grid of KNN:', accuracy_score(pred, labels_test)
#
# # # adaboost
# # bdt = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=1),
# #                          algorithm="SAMME")
# # parameters = {'n_estimators':[50, 100, 140, 200], 'random_state':[0, 4, 7, 9]}
# # bdt = GridSearchCV(bdt, parameters)
# # bdt.fit(features_train_transformed, labels_train)
# # print 'best_estimator of Adaboost:', bdt.best_params_
# # pred = bdt.predict(features_test_transformed)
# # print 'Adaboost classification report:', classification_report(labels_test, pred)
# # print 'Adaboost confusion matrix:', confusion_matrix(labels_test, pred)
# # print 'accuracy_score of grid of Adaboost:', accuracy_score(pred, labels_test)
#
# # # Randomforest
# # clf = RandomForestClassifier()
# # parameters = {'n_estimators':[10, 14, 18, 24], 'max_features':[15], 'min_samples_split':[2,4,5,7], 'random_state':[0,2,4]}
# # clf = GridSearchCV(clf, parameters)
# # clf.fit(features_train_transformed, labels_train)
# # print 'best_estimator of randomforest:', clf.best_params_
# # pred = clf.predict(features_test_transformed)
# # print 'randomforest classification_report:', classification_report(labels_test, pred)
# # print 'randomforest confusion_matrix:', confusion_matrix(labels_test, pred)
# # print 'accuracy_score of randomforest:', accuracy_score(pred, labels_test)
#
#
# #print features_train_transformed
#
# # GaussianNB() classifier on features_train_transformed
# # from sklearn.naive_bayes import GaussianNB
# #
# # clf = GaussianNB()
# # clf.fit(features_train_transformed, labels_train)
# # pred = clf.predict(features_test_transformed)
# # from sklearn.metrics import accuracy_score
# # print 'Naive Bayes accuracy after pca_transformed:', accuracy_score(pred, labels_test)
# # print classification_report(labels_test, pred)
# # print confusion_matrix(labels_test, pred)
#
# ### PCA END


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

features_list = ['poi','salary', 'total_stock_value', 'total_payments','exercised_stock_options','bonus',\
                 'shared_receipt_with_poi','expenses', 'from_this_person_to_poi','from_poi_to_this_person',\
                 'other', 'from_messages', 'to_messages']
# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
clf.fit(features_train, labels_train)
# Naive_bayes
from sklearn.metrics import accuracy_score
pred = clf.predict(features_test)

# accuracy_score = 0.113636363636 with original data
print 'Naive Bayes accuracy:', accuracy_score(pred, labels_test)

# KNN
from sklearn.neighbors import KNeighborsClassifier
nbrs = KNeighborsClassifier(n_neighbors = 2)
nbrs.fit(features_train, labels_train)
pred = nbrs.predict(features_test)

# accuracy_score = 0.886363636364 with original data
print 'KNN accuracy:', accuracy_score(pred, labels_test)

from sklearn.metrics import confusion_matrix
print confusion_matrix(labels_test, pred)

from sklearn.metrics import precision_score
print precision_score(labels_test, pred, average=None)

from sklearn.metrics import recall_score
print recall_score(labels_test, pred, average=None)




### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)