# Random Forest Algorithm on Sonar Dataset
from random import seed
from random import randrange
from csv import reader
from math import sqrt
import sys
from pyspark import SparkContext
import random
import time
# Load a CSV file
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())

# Convert string column to integer
def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup

# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

# Split a dataset into k folds
def RDD_cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_indexed = dataset.zipWithIndex()
    fold_size = int(dataset.count() / n_folds)
    for i in range(n_folds):
        indexes = random.sample(range(0, dataset.count()), fold_size)
        fold = dataset_indexed.filter(lambda foo: foo[1] in indexes).map(lambda foo: foo[0])
        fold.cache()
        dataset_split.append(fold)
    return dataset_split


# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


# Calculate accuracy percentage
def RDD_accuracy_metric(actual, predicted):
    actual = actual.collect()
    predicted = predicted.collect()
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()

    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores


# Evaluate an algorithm using a cross validation split
def RDD_evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = RDD_cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sc.union(train_set)
        test_set = fold.map(lambda row: row[:-1] + [None])
        predicted = algorithm(train_set, test_set, *args)
        actual = fold.map(lambda row: row[-1])
        accuracy = RDD_accuracy_metric(actual, predicted)
        scores.append(accuracy)
        break
    return scores

# Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right

# Split a dataset based on an attribute and an attribute value
def RDD_test_split(index, value, dataset):
    start_time_left = time.time()
    left = dataset.filter(lambda x: x[index] <= value)
    left.cache()
    # print("size of left size ",left.count())
    # print("---Time for left split %s seconds ---" % (time.time() - start_time_left))
    start_time_right = time.time()
    right = dataset.filter(lambda x: x[index] > value)
    right.cache()
    # print("size of right size ", right.count())
    # print("---Time for right split %s seconds ---" % (time.time() - start_time_right))
    return left, right

# Calculate the Gini index for a split dataset
def gini_index(groups, classes):
    # count all samples at split point
    n_instances = float(sum([len(group) for group in groups]))
    # sum weighted Gini index for each group
    gini = 0.0
    for group in groups:
        size = float(len(group))
        # avoid divide by zero
        if size == 0:
            continue
        score = 0.0
        # score the group based on the score for each class
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p
        # weight the group score by its relative size
        gini += (1.0 - score) * (size / n_instances)
    return gini

# Calculate the Gini index for a split dataset
def RDD_gini_index(groups, classes):
    # count all samples at split point
    n_instances = float(sum([group.count() for group in groups]))
    # sum weighted Gini index for each group
    gini = 0.0

    for group in groups:
        start_time_group = time.time()
        size = float(group.count())
        # avoid divide by zero
        if size == 0:
            continue
        score = 0.0
        # score the group based on the score for each class
        for class_val in classes:
            p = group.map(lambda row: row[-1]).collect()
            p = p.count(class_val) / size
            score += p * p
        # weight the group score by its relative size
        gini += (1.0 - score) * (size / n_instances)
        print("---Time group gini score %s seconds ---" % (time.time() - start_time_group))
    return gini

# Select the best split point for a dataset
def get_split(dataset, n_features):
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    features = list()
    while len(features) < n_features:
        index = randrange(len(dataset[0])-1)
        if index not in features:
            features.append(index)
    for index in features:
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {'index':b_index, 'value':b_value, 'groups':b_groups}


# Select the best split point for a dataset
def RDD_get_split(dataset, n_features, name):
    class_values = list(set(dataset.map(lambda row: row[-1]).collect()))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    features = list()
    while len(features) < n_features:
        index = randrange(len(dataset.first())-1)
        if index not in features:
            features.append(index)

    list_dataset = dataset.collect()
    inx = n_features
    for index in features:
        idx = 0
        for row in list_dataset:
            print(inx,idx)

            idx += 1
            start_time_RDD_test_split = time.time()

            groups = RDD_test_split(index, row[index], dataset)
            print("---Function RDD_test_split %s seconds ---" % (time.time() - start_time_RDD_test_split),name)
            start_time_RDD_gini_index = time.time()
            gini = RDD_gini_index(groups, class_values)
            print("---Function RDD_gini_index %s seconds ---" % (time.time() - start_time_RDD_gini_index),name)
            print("gini score of current split ", gini)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
        inx -= 1
    return {'index':b_index, 'value':b_value, 'groups':b_groups}


# Create a terminal node value
def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)

# Create a terminal node value
def RDD_to_terminal(group):
    outcomes = group.map(lambda row: row[-1])
    outcomes = outcomes.collect()
    return max(set(outcomes), key=outcomes.count)

# Create child splits for a node or make terminal
def split(node, max_depth, min_size, n_features, depth):
    left, right = node['groups']
    del(node['groups'])
    # check for a no split
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    # check for max depth
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    # process left child
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left, n_features)
        split(node['left'], max_depth, min_size, n_features, depth+1)
    # process right child
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right, n_features)
        split(node['right'], max_depth, min_size, n_features, depth+1)


# Create child splits for a node or make terminal
def RDD_split(node, max_depth, min_size, n_features, depth):
    left, right = node['groups']
    del (node['groups'])
    left_count = left.count()
    right_count = right.count()
    # check for a no split
    if left_count == 0 or right_count == 0:
        print("Stopping due to zero count",left_count,right_count)
        node['left'] = node['right'] = RDD_to_terminal( sc.union([left,right]) )
        return
    # check for max depth
    if depth >= max_depth:
        print("Stopping due to max dept")
        node['left'], node['right'] = RDD_to_terminal(left), RDD_to_terminal(right)
        return
    # process left child
    if left_count <= min_size:
        node['left'] = RDD_to_terminal(left)
    else:
        node['left'] = RDD_get_split(left, n_features,'left')
        RDD_split(node['left'], max_depth, min_size, n_features, depth + 1)
    # process right child
    if right_count <= min_size:
        node['right'] = RDD_to_terminal(right)
    else:
        node['right'] = RDD_get_split(right, n_features,'right')
        RDD_split(node['right'], max_depth, min_size, n_features, depth + 1)

# Build a decision tree
def build_tree(train, max_depth, min_size, n_features):
    root = get_split(train, n_features)
    split(root, max_depth, min_size, n_features, 1)
    return root


# Build a decision tree
def RDD_build_tree(train, max_depth, min_size, n_features):
    root = RDD_get_split(train, n_features,'root')
    print("After root in RDD build tree**************************")
    RDD_split(root, max_depth, min_size, n_features, 1)
    return root

# Make a prediction with a decision tree
def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']

# Create a random subsample from the dataset with replacement
def subsample(dataset, ratio):
    sample = list()
    n_sample = round(len(dataset) * ratio)
    while len(sample) < n_sample:
        index = randrange(len(dataset))
        sample.append(dataset[index])
    return sample


# Create a random subsample from the dataset with replacement
def RDD_subsample(dataset, ratio):
    dataset_count = dataset.count()
    n_sample = int(round(dataset_count * ratio))
    dataset_indexed = dataset.zipWithIndex()
    indexes = random.sample(range(0, dataset_count), n_sample)
    sample = dataset_indexed.filter(lambda foo: foo[1] in indexes)
    sample = sample.map(lambda row: row[0])
    sample.cache()
    return sample

# Make a prediction with a list of bagged trees
def bagging_predict(trees, row):
    predictions = [predict(tree, row) for tree in trees]
    return max(set(predictions), key=predictions.count)

# Random Forest Algorithm
def random_forest(train, test, max_depth, min_size, sample_size, n_trees, n_features):
    trees = list()
    for i in range(n_trees):
        sample = subsample(train, sample_size)
        tree = build_tree(sample, max_depth, min_size, n_features)
        trees.append(tree)
    predictions = [bagging_predict(trees, row) for row in test]
    return(predictions)

# Random Forest Algorithm
def RDD_random_forest(train, test, max_depth, min_size, sample_size, n_trees, n_features):
    trees = list()

    sample = RDD_subsample(train, sample_size)
    tree = RDD_build_tree(sample, max_depth, min_size, n_features)
    trees.append(tree)

    # for i in range(n_trees):
    #     print("tree : ",i)
    #     sample = RDD_subsample(train, sample_size)
    #     tree = RDD_build_tree(sample, max_depth, min_size, n_features)
    #     trees.append(tree)
    predictions = test.map(lambda row: bagging_predict(trees, row))
    return(predictions)

def label_to_int(row):
    row[-1] = lookup[row[-1]]
    return row

# Test the random forest algorithm
seed(2)
# load and prepare data
filename = 'sonar.all-data.csv'
dataset = load_csv(filename)

# Prepare data in RDD
sc = SparkContext(appName="RandomForest")
inputData = sc.textFile(sys.argv[1])
inputData = inputData.map(lambda line: line.split(',')).map(lambda row: [float(x) for x in row[:-1]] + [row[-1]])
inputData.cache()
label = set(inputData.map(lambda row: row[-1]).collect())
lookup = {}
for i, value in enumerate(label):
    lookup[value] = i


data = inputData.collect()
indexs = random.sample(range(0,len(data)),10)
data2 = []
for i in indexs:
    data2.append(data[i])
inputData = sc.parallelize(data2)

inputData = inputData.map(label_to_int)
inputData.cache()
# convert string attributes to integers
for i in range(0, len(dataset[0])-1):
    str_column_to_float(dataset, i)
# convert class column to integers
str_column_to_int(dataset, len(dataset[0])-1)
dataset = dataset[:3000]
# evaluate algorithm
n_folds = 5
max_depth = 3
min_size = 1
sample_size = 1.0

# n_features = int(sqrt(len(dataset[0])-1))
n_features = 3
for n_trees in [1]:
    # scores = evaluate_algorithm(dataset, random_forest, n_folds, max_depth, min_size, sample_size, n_trees, n_features)
    scores_RDD = RDD_evaluate_algorithm(inputData, RDD_random_forest, n_folds, max_depth, min_size, sample_size, n_trees, n_features)
    print('Trees: %d' % n_trees)
    # print('Scores: %s' % scores)
    print('Scores_RDD: %s' % scores_RDD)
    # print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
    print('Mean Accuracy_RDD: %.3f%%' % (sum(scores_RDD) / float(len(scores_RDD))))