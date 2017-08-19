# Machine Learning Analysis on Enron Email Dataset

Objective:
Build a machine learning algorithm to detect employee who may have committed fraud based on financial and email datasets.

## 1.	Data Wrangling
### a).  load the dataset:
```
import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

my_dataset = data_dict
```
The dataset has 146 employees including 18 (POI) with 21 features. All features are numerical but ‘email_address’ (string) and ‘poi’ (bool). 
There are 146 employees
print len(my_dataset)

```
# Print all features for each employee and the total number of features
count = 0
for features, feature_values in data_dict['METTS MARK'].iteritems():
    count += 1
    print features #, 'corresponsive to ', type(features)
print 'count:', count
146
salary
to_messages
deferral_payments
total_payments
exercised_stock_options
bonus
restricted_stock
shared_receipt_with_poi
restricted_stock_deferred
total_stock_value
expenses
loan_advances
from_messages
other
from_this_person_to_poi
poi
director_fees
deferred_income
long_term_incentive
email_address
from_poi_to_this_person
count: 21
```

In order to clean data more easily, I converted the data to PandasDataFrame. The dataset is unbalanced since there are only 12.3% employees who have committed to fraud. 
```
features_list = ['poi','salary', 'total_stock_value', 'total_payments','exercised_stock_options','bonus',\
                 'shared_receipt_with_poi','expenses', 'from_this_person_to_poi','from_poi_to_this_person',\
                 'other', 'from_messages', 'to_messages', 'deferral_payments', 'restricted_stock',\
                 'restricted_stock_deferred', 'loan_advances','director_fees','deferred_income','long_term_incentive'\
                 ]
data = featureFormat(my_dataset, features_list, sort_keys = False, remove_all_zeroes = False)
labels, features = targetFeatureSplit(data)
print 'data type:', type(data)
print len(features)
print sum(labels)/len(labels)
data type: <type 'numpy.ndarray'>
146
0.123287671233
```
```
import numpy as np
import pandas as pd
# Convert dataset into PandaDataFrame
df = pd.DataFrame(
    data = data,
    index = my_dataset.keys(),
    columns = features_list)
```
### b). Exploring whether there are features have missing values.
```
# Explore whether there are features have many missing values.
print 'sum of zero of salary of employee:', sum(df['salary'] == 0)
print 'sum of zero of bonus of employee:', sum(df['bonus'] == 0)
print 'sum of zero of total_stock_value:', sum(df['total_stock_value'] == 0)
print 'sum of zero of exercised_stock_value:', sum(df['exercised_stock_options'] == 0)
print 'sum of zero of deferral_payments:', sum(df['deferral_payments'] == 0)
print 'sum of zero of restricted_stock:', sum(df['restricted_stock'] == 0)
sum of zero of salary of employee: 51
sum of zero of bonus of employee: 64
sum of zero of total_stock_value: 20
sum of zero of exercised_stock_value: 44
sum of zero of deferral_payments: 107
sum of zero of restricted_stock: 36
```

### c). Detecting if there is any employee with all zero values for all features. I wrote a for loop and found out there are two employees who have all zero values for all features. I have 144 observations in the dataset after I had deleted those two employees.
```
# Moving all zeros employees
zero_employee = []
for i in range(0, len(df.index)):
     if df.sum(axis = 1)[i] == 0:
         zero_employee.append(df.index[i])
print 'Total number of employees with original data:', len(df.index)
print zero_employee
df.drop(zero_employee, inplace=True)
print 'Total number of employees after deleted all zeros employee:',len(df.index)
Total number of employees with original data: 146
['CHAN RONNIE', 'LOCKHART EUGENE E']
Total number of employees after deleted all zeros employee: 144
```
### d). Visualizing main features such as ‘salary’, ‘bonus’, ‘total_payments’, ‘total_stock_value’ by using library ‘matplotlib.pyplot’. There is one employee (un-poi) who has extreme values for these four features. I found this extreme point is called ‘TOTAL’ which is obvious wrong, so I deleted this point.
```
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize = (4, 4))
df.boxplot(['salary', 'total_payments', 'bonus', 'total_stock_value'], 'poi', ax)
plt.show()
```

```
# Figure out which un-poi employee has this extreme values
unpoi_with_extreme_values_index = df.idxmax(axis = 0)['salary']
print unpoi_with_extreme_values_index
TOTAL
# Remove 'TOTAL' from df (DataFrame)
df.drop(unpoi_with_extreme_values_index, inplace = True)
print len(df)
143
```
### e). Creating new features. The first one is called ‘fraction_sum_salary_and_bonus_in_total_payments’ which means how many payments are coming from salary and bonus. The second one is called ‘fraction_email_poi_to_this_person’ which means how many emails this person received are coming from poi. The third one is called ‘fraction_email_to_poi’ which means how many emails this person sent are sent to poi.

```
# ‘fraction_sum_salary_and_bonus_in_total_payments’
df['fraction_sum_salary_and_bonus_in_total_payments']=sum(df['salary'] + df['bonus']) / df['total_payments']
df['fraction_sum_salary_and_bonus_in_total_payments'].replace(np.inf, np.nan,inplace=True)
df['fraction_sum_salary_and_bonus_in_total_payments'].fillna(0, inplace=True)

# ‘fraction_email_poi_to_this_person’
df['fraction_email_poi_to_this_person'] = df['from_poi_to_this_person']/df['to_messages']
df['fraction_email_poi_to_this_person'].fillna(0, inplace=True)

# ‘fraction_email_to_poi’
df['fraction_email_to_poi'] = df['from_this_person_to_poi']/df['from_messages']
df['fraction_email_to_poi'].fillna(0,inplace=True)
```
In order to continue feature engineering, I convert the dataset to dictionary.
```
# Convert DataFrame to Dictionary after creating and exploring the data
my_dataset = df.to_dict(orient = 'index')
print 'my_dataset', len(my_dataset)
```

## 2.	Feature Engineering 
### a). Before feature engineering, I split the dataset into training dataset (70%) and testing dataset (30%) in order to prevent overfitting.
```
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
```

### b.) Using SelectKBest method from sklearn to get scores and pvalues of each feature.
```
from sklearn.feature_selection import SelectKBest, f_classif selector = SelectKBest(f_classif, k = 10)
selector.fit(features_train, labels_train)
print 'selector.scores:',selector.scores_
print 'selector.pvalues:', selector.pvalues_
```
```
selector.scores: [  1.06124768e+01   9.62632217e+00   5.25866447e-01   1.03844772e+01
   6.02287938e+00   1.49284507e+00   6.80962317e+00   3.35387308e+00
   1.27598974e+00   6.79105633e-01   5.51858069e-02   8.00346695e-04
   1.21411726e+00   1.02280464e+00   9.44759847e-02              nan
   1.74674632e+00   4.50179539e+00   2.19811010e+00   3.01315076e-01
   1.04811536e+00   1.19242161e+01]
selector.pvalues: [  1.54395626e-03   2.50625925e-03   4.70077919e-01   1.72577581e-03
   1.58859410e-02   2.24706417e-01   1.04875897e-02   7.00851613e-02
   2.61403330e-01   4.11895996e-01   8.14763619e-01   9.77488069e-01
   2.73217039e-01   3.14345158e-01   7.59214405e-01              nan
   1.89363326e-01   3.63812831e-02   1.41388762e-01   5.84307672e-01
   3.08462053e-01   8.19735088e-04]
```
### c). Selecting significant features. I decided to get all features who have pvalues are less than significant level 0.1.
```
k = sum(selector.pvalues_ <= 0.1)
print 'best k features:', k
new_features = ['poi']
for i in range(0,len(features_list[1:])):
    if selector.pvalues_[i] <= 0.1:
        new_features.append(features_list[i+1])
print 'new_feature:', new_features
```
```
best k features: 8
new_feature: ['poi', 'salary', 'total_stock_value', 'exercised_stock_options', 'bonus', 'expenses', 'from_this_person_to_poi', 'deferred_income', 'fraction_email_to_poi']
```
### d). Splitting the dataset again with the best k features.
```
# Using new_features to split dataset
data = featureFormat(my_dataset, new_features, sort_keys = False, remove_all_zeroes = False)
labels, features = targetFeatureSplit(data)
```
```
# Split data into training dataset and testing dataset
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
```

## 3.	Training and Tuning Machine Learning algorithms
Before training my model, I need to explain why I split the dataset into training dataset (70%) and testing dataset (30%). The purpose of splitting is to prevent my training model from overfitting. If I do not do cross-validation, it is highly likely to overfit machine learning algorithms. Building a pipe line and grid search to tuning different algorithms with different parameters.

### 3.1	KNN (K-Nearest-Neighbors) 
I tuned ‘n_neighbors’ with 4, 5, 6, 8, and 10.
```
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
```
```
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
```
```
best_estimator of KNN {'select__k': 8, 'classify__n_neighbors': 8}
KNN classification report: precision    recall  f1-score   support

                     0.0       0.86      1.00      0.92        37
                     1.0       0.00      0.00      0.00         6

avg / total                    0.74      0.86      0.80        43

KNN confusion matrix: [[37  0]
                       [ 6  0]]
accuracy_score of grid of KNN: 0.860465116279
```

### 3.2	Decision Tree
I tuned ‘min_samples_split’ from 2 to 23.
```
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
```
```
best_estimator of decision_tress {'classify__min_samples_split': 2, 'select__k': 8}
decision_tress classification report: precision    recall  f1-score   support

                               0.0       0.90      0.73      0.81        37
                               1.0       0.23      0.50      0.32         6

avg / total                              0.81      0.70      0.74        43

decision tress confusion matrix: [[27 10]
                                  [ 3  3]]
accuracy_score of grid of decision tree: 0.697674418605
```
### 3.3	Logistic Regression
I tuned ‘C’ and ‘tol’.
```
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
```
```
best_estimator of logisticregression: {'classify__C': 1e-07, 'classify__tol': 0.0001, 'select__k': 8}
LogisticRegression classification report: precision    recall  f1-score   support

                                   0.0       0.94      0.81      0.87        37
                                   1.0       0.36      0.67      0.47         6
 
avg / total                                  0.86      0.79      0.81        43

LogisticRegression confusion matrix: [[30  7]
                                      [ 2  4]]
accuracy_score of grid of LogisticRegression: 0.790697674419
```
## 4.	Final Decision
Comparing those three algorithms, I ended up with Logistic Regression. 
```
Metrics         KNN        Decision Tree    LogisticRegression
precision       0.0             0.23             0.36
recall          0.0             0.5              0.67
f1-score        0.0             0.32             0.47
Accuracy  0.860465116279    0.697674418605   0.790697674419
```
```
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

dump_classifier_and_data(clf, my_dataset, features_list)
```
After I had chosen the final classification model (LogisticRegression), I ran the algorithm with the best parameters on the testing dataset without label. My algorithm achieved precision (0.32716) and recall (0.3535) are both higher than 0.3. 

