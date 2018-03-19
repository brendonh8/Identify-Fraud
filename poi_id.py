#!/usr/bin/python
import math
import sys
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pprint
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from tester import test_classifier, dump_classifier_and_data
from feature_format import featureFormat, targetFeatureSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing, tree, svm
from sklearn.metrics import recall_score, precision_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier

sys.path.append("../tools/")
##############################################################################
# Task 1: Select what features you'll use.
##############################################################################
target_label = 'poi'

email_features_list = [
    'to_messages',
    'email_address',
    'from_poi_to_this_person',
    'from_messages',
    'from_this_person_to_poi',
    'shared_receipt_with_poi']

financial_features_list = [
    'salary',
    'deferral_payments',
    'total_payments',
    'loan_advances',
    'bonus',
    'restricted_stock_deferred',
    'deferred_income',
    'total_stock_value',
    'expenses',
    'exercised_stock_options',
    'other',
    'long_term_incentive',
    'restricted_stock',
    'director_fees']

total_features_list = [target_label] + financial_features_list + email_features_list

# Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

poi_count = 0
for employee in data_dict:
    if data_dict[employee]['poi']:
        poi_count += 1
print("Number of Data Points: ", len(data_dict))
print("Number of POIs: ", poi_count)
print("Number of non POIs: ", len(data_dict) - poi_count)
print("Number of Features: ", len(total_features_list))

# Find how many missing values
missing_values = {}
for feature in total_features_list:
    missing_values[feature] = 0

for employee in data_dict:
    for feature in data_dict[employee]:
        if data_dict[employee][feature] == 'NaN':
            missing_values[feature] += 1

print 'Missing Values: '
pprint.pprint(missing_values)

df = pd.DataFrame.from_dict(data_dict, orient='index', dtype=np.float)
dfNull = df.isnull().sum()
dfNull.plot(kind='barh')
plt.xlabel('Quantity of NaN')
#plt.show()

##############################################################################
# Task 2: Remove outliers
##############################################################################


def plot_data(data_dict, features):
    '''
    Plot features denoting poi's by color
    '''
    data = featureFormat(data_dict, features)
    poi_colors = ['b', 'r']
    for point in data:
        matplotlib.pyplot.scatter(point[1], point[2], c=poi_colors[int(point[0])])

    matplotlib.pyplot.xlabel(features[1])
    matplotlib.pyplot.ylabel(features[2])
   # matplotlib.pyplot.show()


features = ['poi', 'salary', 'bonus']
# remove the outliers
data_dict.pop('TOTAL', 0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)
# plot_data(data_dict, features)

##############################################################################
# Task 3: Create new feature(s)
##############################################################################

# Store to my_dataset for easy export below.
my_dataset = data_dict


def compute_ratio(numerator, denominator):
    ratio = float(numerator) / float(denominator)
    ratio = ratio if not math.isnan(ratio) else 0
    # return 0 if num or den is NAN
    return ratio


new_features = ["ratio_from_poi", "ratio_to_poi", "ratio_exercised_stock_tot_stock_value"]

# Create new features in the dataset
for name in my_dataset:

    data_point = my_dataset[name]

    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]
    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]

    total_stock_value = data_point["total_stock_value"]
    exercised_stock_options = data_point["exercised_stock_options"]

    ratio_from_poi = compute_ratio(from_poi_to_this_person, to_messages)
    ratio_to_poi = compute_ratio(from_this_person_to_poi, from_messages)
    ratio_exercised_stock_tot_stock_value = compute_ratio(exercised_stock_options, total_stock_value)

    data_point["ratio_from_poi"] = ratio_from_poi
    data_point["ratio_to_poi"] = ratio_to_poi
    data_point["ratio_exercised_stock_tot_stock_value"] = ratio_exercised_stock_tot_stock_value


my_features_list = total_features_list + new_features
my_features_list.remove('email_address')

# Extract features and labels from dataset for local testing

data = featureFormat(my_dataset, my_features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)


def secondElem(elem):
    return elem[1]


# select the best features to use
kbest = SelectKBest(f_classif, k=10)
kbest.fit_transform(features, labels)
features_selected = [my_features_list[i+1] for i in kbest.get_support(indices=True)]
scores = zip(my_features_list[1:], kbest.scores_)
sorted_scores = sorted(scores, key=secondElem, reverse=True)
print 'Select KBest scores: '
pprint.pprint(sorted_scores)

print 'Features selected: '
print [target_label] + features_selected

features_list = ['poi']
features_list.extend(features_selected)

# dataset without new features
features_list.remove('ratio_to_poi')
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)
scaler = preprocessing.MinMaxScaler()
#features = scaler.fit_transform(features)

# dataset with new features
new_features_list = features_list + ['ratio_to_poi']
data = featureFormat(my_dataset, new_features_list, sort_keys=True)
new_labels, new_features = targetFeatureSplit(data)
#new_features = scaler.fit_transform(new_features)


##############################################################################
# Task 4: Try a varity of classifiers
##############################################################################

# Example starting point. Try investigating other evaluation techniques!
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)


def tune_params(grid_search, features, labels, params, iters=50):
    '''
    takes a grid search and dictionary of parameters for a specific
    algorith. Tunes the algorithm for best parameters and prints
    evaluation metrics(accuracy, precision, recall)
    '''
    acc = []
    pre = []
    recall = []
    for i in range(iters):
        features_train, features_test, labels_train, labels_test = \
            train_test_split(features, labels, test_size=0.3, random_state=i)
        grid_search.fit(features_train, labels_train)
        predict = grid_search.predict(features_test)

        acc = acc + [accuracy_score(labels_test, predict)]
        pre = pre + [precision_score(labels_test, predict)]
        recall = recall + [recall_score(labels_test, predict)]
    print "accuracy: {}".format(np.mean(acc))
    print "precision: {}".format(np.mean(pre))
    print "recall: {}".format(np.mean(recall))

    best_params = grid_search.best_estimator_.get_params()
    for param_name in params.keys():
        print("%s = %r, " % (param_name, best_params[param_name]))
##############################################################################
# Task 5: Tune your classifier to achieve better than .3 precision and recall
##############################################################################


# Naive Bayes
nb_clf = GaussianNB()
nb_param = {}
nb_grid_search = GridSearchCV(estimator=nb_clf, param_grid=nb_param)
print("Naive Bayes model evaluation")
#tune_params(nb_grid_search, features, labels, nb_param)
#tune_params(nb_grid_search, new_features, new_labels, nb_param)


# 2. Support Vector Machines
svm_clf = svm.SVC()
svm_param = {'kernel': ('linear', 'rbf', 'sigmoid'),
             'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
             'C': [0.1, 1, 10, 100, 1000]}
svm_grid_search = GridSearchCV(estimator=svm_clf, param_grid=svm_param)

print("SVM model evaluation")
#tune_params(svm_grid_search, features, labels, svm_param)
#tune_params(svm_grid_search, new_features, new_labels, svm_param)


# 3. Decision Tree
dt_clf = tree.DecisionTreeClassifier()
dt_param = {'criterion': ('gini', 'entropy'),
            'splitter': ('best', 'random'),
            'min_samples_split': range(2, 5)}
dt_grid_search = GridSearchCV(estimator=dt_clf, param_grid=dt_param)

print("Decision Tree model evaluation")
#tune_params(dt_grid_search, features, labels, dt_param)
#tune_params(dt_grid_search, new_features, new_labels, dt_param)

# 4. Random Forest
rf_clf = RandomForestClassifier(n_estimators=10)
rf_param = {}
rf_grid_search = GridSearchCV(estimator=rf_clf, param_grid=rf_param)

print("Random Forest model evaluation")
#tune_params(rf_grid_search, features, labels, rf_param)
#tune_params(rf_grid_search, new_features, new_labels, rf_param)


# Task 6: Dump your classifier, dataset, and features_list so anyone can
# check your results. You do not need to change anything below, but make sure
# that the version of poi_id.py that you submit can be run on its own and
# generates the necessary .pkl files for validating your results.
clf = dt_clf
features_list = new_features_list
dump_classifier_and_data(clf, my_dataset, features_list)
