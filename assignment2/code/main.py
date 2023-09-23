import copy
import numpy as np
import pandas as pd
import math
import time
import random
from  sklearn.model_selection import KFold 
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import confusion_matrix

df=pd.read_csv("Employee-Attrition.csv")
dataset=df.copy()


# Equal-Width Discretization
def equal_width_discretization(arr, bins=3):
    # max and min values
    arr_max = max(arr)
    arr_min = min(arr)
    # weight
    w = int((arr_max - arr_min) / bins)
    bin_slices = []
    # declaring intervals
    for i in range(0, bins + 1):
        if i == bins:
            bin_slices.append(arr_max)
        else:
            bin_slices.append(arr_min + w * i)

    res = [0] * len(arr)

    # discretization of dataset column according to intervals
    for i in range(len(arr)):
        for j in range(len(bin_slices) - 1):
            if arr[i] >= bin_slices[j] and arr[i] <= bin_slices[j + 1]:
                res[i] = str(bin_slices[j]) + " - " + str(bin_slices[j + 1])

    return res


#discretization of dataset
columns_for_discretization=list(dataset.select_dtypes(include='number').columns)
for col in columns_for_discretization:
    dataset[col]=equal_width_discretization(dataset[col].to_numpy().tolist(),3)

#shuffling the dataset
dataset= dataset.sample(frac = 1)

#all Attributes Unique Values after Equal-Width Discretization
columnsUniqueValues={}
for i in dataset.columns:
    columnsUniqueValues[i]=list(dataset[i].unique())

#data exploring
dataset.head()


# total entropy of given dataset
def total_entropy(train_set, target_attribute="Attrition"):
    classes = train_set[target_attribute].unique()  # "yes " or "no"
    entropy = 0.0
    for c in classes:
        fraction = train_set[target_attribute].value_counts()[c] / len(train_set[target_attribute])
        entropy += -fraction * np.log2(fraction)

    return entropy


# attribute entropy of given dataset
def get_attribute_entropy(train_set, attribute, target_attribute="Attrition"):
    classes = train_set[target_attribute].unique()  # "yes " or "no"
    variables = train_set[attribute].unique()  # attribute classes
    total_entropy = 0.0
    # for each value
    for var in variables:
        entropy = 0.0
        total_count = len(train_set[attribute][train_set[attribute] == var])
        # entropy of each attribute value
        for c in classes:
            value_count = len(train_set[attribute][train_set[attribute] == var][train_set[target_attribute] == c])
            if total_count != 0 and value_count != 0:
                fraction = value_count / total_count
                entropy += -fraction * np.log2(fraction)
        total_fraction = total_count / len(train_set)
        total_entropy += total_fraction * entropy

    return total_entropy


# information gain is simply total_entropy-entropyofattribute
def get_information_gain(train_set, attribute, target_attribute="Attrition"):
    return total_entropy(train_set, target_attribute) - get_attribute_entropy(train_set, attribute, target_attribute)


# finds the attribute that has highest information gain
def find_max_attribute(train_set, attribute_list, target_attribute="Attrition"):
    ig_list = []
    for attribute in attribute_list:
        ig = get_information_gain(train_set, attribute, target_attribute)
        ig_list.append(ig)
    i = np.argmax(ig_list)  # index of attribute which has maximum information gain value

    return attribute_list[i]


# Tree Node for ID3 algroithm and Decision Tree
class Node:
    def __init__(self):
        self.childs = []
        self.value = ""
        self.isLeaf = False
        self.pred = ""
        self.parent = None


def ID3(train_set, columnsAttributeValues, attribute_list, target_attribute="Attrition"):
    # create Root of Tree
    root = Node()
    # find the attribute that has maximum information gain
    max_attribute = find_max_attribute(train_set, attribute_list, target_attribute)
    root.value = max_attribute

    # for each value of max attribute
    for val in columnsAttributeValues[max_attribute]:
        # obtaining subdataset of max attribute value
        sub_train_set = train_set[train_set[max_attribute] == val]
        # Yes and No values
        target_vals = list(sub_train_set[target_attribute].unique())
        # creating new Node for child node
        newNode = Node()
        newNode.value = val
        newNode.parent = root

        # if subdataset is empty
        if len(sub_train_set) == 0:

            # find most occured value in dataset and declare Node as leaf
            newNode.isLeaf = True
            target_vals = list(train_set[target_attribute].unique())

            maxPred = ""
            maxPredCount = 0
            for tval in target_vals:
                cnt = len(train_set[train_set[target_attribute] == tval])
                if cnt > maxPredCount:
                    maxPredCount = cnt
                    maxPred = tval
            newNode.pred = maxPred
            # add Node to the root
            root.childs.append(newNode)

        # if subdataset contains only "No" or "Yes" as Target Attribute
        elif len(target_vals) == 1:
            # declare Node as leaf
            newNode.isLeaf = True
            newNode.pred = target_vals[0]
            root.childs.append(newNode)

        # else recursively call id3 algorithm
        else:
            # subdataset attributes list
            sub_attribute_list = attribute_list.copy()
            sub_attribute_list.remove(max_attribute)
            # getting child recursively and add it to the root
            child = ID3(sub_train_set, columnsAttributeValues, sub_attribute_list, target_attribute)
            child.parent = newNode
            newNode.childs.append(child)
            root.childs.append(newNode)

    # finally return root
    return root


# predicting the singe row with Decision tree
def predict(root, test):
    # for each child
    for child in root.childs:

        if child.value == test[root.value]:
            # if child is leaf then return its pred value
            if child.isLeaf:

                return child.pred
                # else recursively call predict
            else:
                return predict(child.childs[0], test)


def printTree(root, depth=0):
    for i in range(depth):
        print("      ", end="")

    if len(root.childs) == 1 or root.isLeaf:
        print(">" + root.value, end="")
    else:
        print("*" + root.value, end="")
    if root.isLeaf:
        print("->", root.pred)
    print()
    for child in root.childs:
        printTree(child, depth + 1)


# train and test datas
x = dataset.drop(columns=['Attrition']).to_numpy()
y = dataset['Attrition'].to_numpy()

columns = list(dataset.drop(columns=['Attrition']).columns)
columns.append('Attrition')

# Kfold-cross-validation
kf = KFold(n_splits=5)
kf.get_n_splits(x)

allFoldResults = {"Fold 1": [], "Fold 2": [], "Fold 3": [], "Fold 4": [], "Fold 5": []}
fnum = 1
# best tree to print
bestTree = None
bestTreeAccuracy = 0
for train_index, test_index in kf.split(x):
    # declaring x_test and y_test  as pd.Dataframe
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    train_data = pd.DataFrame(np.concatenate((x_train, np.asarray([y_train]).T), axis=1), columns=columns)
    x_test = pd.DataFrame(x_test, columns=list(dataset.drop(columns=['Attrition']).columns))

    st = time.time()
    attribute_list = list(train_data.drop(columns=["Attrition"]).columns)

    # call ID3 algoritm for training
    tree = ID3(train_data, columnsUniqueValues, attribute_list, "Attrition")
    tb = time.time() - st

    print("Training time for Fold {} is : {} sn".format(fnum, round(tb, 3)))

    # Prediction Time
    preds = []
    # for each row in x_test ,predict its value
    for index, row in x_test.iterrows():
        p = predict(tree, row)
        preds.append(p)

    # displaying accuracy,precision,recall and f1score metrics
    print("Fold {} Results \n  Accuracy :{} \n Precision :{} \n  Recall :{} \n  F1Score :{} \n\n".format(fnum,
                                                                                                         accuracy_score(
                                                                                                             y_test,
                                                                                                             preds),
                                                                                                         precision_score(
                                                                                                             y_test,
                                                                                                             preds,
                                                                                                             average='weighted'),
                                                                                                         recall_score(
                                                                                                             y_test,
                                                                                                             preds,
                                                                                                             average='weighted'),
                                                                                                         f1_score(
                                                                                                             y_test,
                                                                                                             preds,
                                                                                                             average='weighted')))

    # find best tree to print
    if accuracy_score(y_test, preds) > bestTreeAccuracy:
        bestTree = tree
        bestTreeAccuracy = accuracy_score(y_test, preds)

    allFoldResults[("Fold " + str(fnum))].append(accuracy_score(y_test, preds))
    allFoldResults[("Fold " + str(fnum))].append(precision_score(y_test, preds, average='weighted'))
    allFoldResults[("Fold " + str(fnum))].append(recall_score(y_test, preds, average='weighted'))
    allFoldResults[("Fold " + str(fnum))].append(f1_score(y_test, preds, average='weighted'))

    fnum += 1

printTree(bestTree)

allFolds=pd.DataFrame.from_dict(allFoldResults,orient='index',
                       columns=['Accuracy', 'Precision', 'Recall', 'F1Score'])
allFolds.loc['Avrg. OF Folds'] = allFolds.mean()

#prints table as Jupter Markdown
#print(allFolds.to_markdown())

# return if node is leaf
def isLeaf(node):
    return node.isLeaf


# return if node is twig
def isTwig(node):
    for child in node.childs:
        if not isLeaf(child):
            return False
    return True


# get all twigs
def getAllTwigs(node, twlist):
    # if node is twig ,add to twigList and return
    if isTwig(node):
        twlist.append(node)
        return
    else:
        # else ,recursively call it for each child
        for child in node.childs:
            if len(child.childs) > 0:
                getAllTwigs(child.childs[0], twlist)
    return twlist


# method for obtaining subdataset of twig
# recursion from twig to root
def getTwigDataset(root, dataset, twig):
    if root == twig:
        return dataset
    else:
        if twig is not None:
            return getTwigDataset(root, dataset[dataset[twig.parent.parent.value] == twig.parent.value],
                                  twig.parent.parent)


def prunTree(train_data, y_validation, y_test, targetAttribute="Attrition"):
    st = time.time()

    attribute_list = list(train_data.drop(columns=["Attrition"]).columns)

    # obtain tree
    tree = ID3(train_data, columnsUniqueValues, attribute_list, "Attrition")
    treeBeforePrun = copy.deepcopy(tree)

    # calculate accuracy on test data before prunning
    preds = []
    for index, row in x_test.iterrows():
        p = predict(tree, row)
        preds.append(p)

    accuracyWithoutPrunning = accuracy_score(y_test, preds)
    print("Accuracy Without Prunning -Test Set : {}".format(accuracyWithoutPrunning))

    print("\nPrunning Process begins...\n")
    # calculate accuracy on validation data before prunning
    preds = []
    for index, row in x_validation.iterrows():
        p = predict(tree, row)
        preds.append(p)
    accuracyBeforePrunning = accuracy_score(y_validation, preds)

    prunCount = 0
    while True:

        print("Accuracy Before Prunning -Validation Set: {}".format(accuracyBeforePrunning))

        minInfoGainTwig = ""
        minInfoGain = 10
        # all twigs
        twigList = getAllTwigs(tree, [])
        # find the twig that has least information gain
        for twig in twigList:
            subdataForTwig = getTwigDataset(tree, dataset, twig)
            ig = get_information_gain(subdataForTwig, twig.value)
            if ig < minInfoGain:
                minInfoGain = ig
                minInfoGainTwig = twig

        prunSet = []
        # remove from twiglist
        twigList.remove(minInfoGainTwig)

        # get predictions of twig
        for child in minInfoGainTwig.childs:
            prunSet.append(child.pred)

        # create new leaf node
        newNode = Node()
        newNode.isLeaf = True
        newNode.value = minInfoGainTwig.parent.value
        # declare its prediction value as max of child values
        newNode.pred = max(set(prunSet), key=prunSet.count)

        # prunning the tree
        prunIndex = 0  # this value is used later on reversing prunning
        for pchild in range(len(minInfoGainTwig.parent.parent.childs)):
            if minInfoGainTwig.parent.parent.childs[pchild].value == minInfoGainTwig.parent.value:
                minInfoGainTwig.parent.parent.childs[pchild] = newNode
                prunIndex = pchild
                break

        # calculate accuracy after prunn on validation data
        preds = []
        for index, row in x_validation.iterrows():
            p = predict(tree, row)
            preds.append(p)

        accuracyAfterPrunning = accuracy_score(y_validation, preds)
        print("Accuracy After Prunning -Validation Set: {}".format(accuracyAfterPrunning))

        # if accuracy incerased or remained same, do the same operations again
        if (accuracyAfterPrunning >= accuracyBeforePrunning):
            accuracyBeforePrunning = accuracyAfterPrunning
            prunCount += 1

        # if accuracy is decreased
        else:
            # reverse last prunning
            print("Accuracy decreased, reversing last prunn..")
            minInfoGainTwig.parent.parent.childs[prunIndex] = minInfoGainTwig.parent
            break

    print("Prunned {} times.".format(prunCount))
    tb = time.time() - st
    print("Prunning Process completed in {} seconds".format(tb))

    # calculate accuracy on test data after prunning is completed
    preds = []
    for index, row in x_test.iterrows():
        p = predict(tree, row)
        preds.append(p)

    accuracyWithPrunning = accuracy_score(y_test, preds)

    print("\nAccuracy With Prunning -Test Set: {}".format(accuracyWithPrunning))
    # return accuracy before and after prunning on test data
    return accuracyWithoutPrunning, accuracyWithPrunning, treeBeforePrun, tree


# for declaring best tree in 5-fold that has highest accuracy gain after prunning
bestTreeDifOfPrun = 0
bestTreeBeforePrun = None
bestTreeAfterPrun = None

fnum = 1
resultOfPrunning = {"Fold 1": [], "Fold 2": [], "Fold 3": [], "Fold 4": [], "Fold 5": []}
for train_index, test_index in kf.split(x):
    # declaring x_test and y_test  as pd.Dataframe
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    # train data -%80 of dataset
    train_data = pd.DataFrame(np.concatenate((x_train, np.asarray([y_train]).T), axis=1), columns=columns)
    # test_data -%20 of dataset
    x_test = pd.DataFrame(x_test, columns=list(dataset.drop(columns=['Attrition']).columns))

    attribute_list = list(train_data.drop(columns=["Attrition"]).columns)

    # validation data -%25 of train data (%20 of total dataset)
    validation_data = train_data[882:1176]
    # train data decreased to %60 of dataset
    train_data = train_data[:882]
    # y_validation and y_test
    y_validation = validation_data["Attrition"]
    x_validation = pd.DataFrame(validation_data, columns=list(dataset.drop(columns=['Attrition']).columns))

    print("\nFold {} -Prunning on Validation Data....".format(fnum))

    # calling prunning tree, aap is accuracy before prunning and aap accuracy after prunning
    abp, aap, treeBeforePrun, treeAfterPrun = prunTree(train_data, y_validation, y_test, targetAttribute="Attrition")

    # find the best tree that highest accuracy change
    if (aap - abp) > bestTreeDifOfPrun:
        bestTreeDifOfPrun = aap - abp
        bestTreeBeforePrun = treeBeforePrun
        bestTreeAfterPrun = treeAfterPrun

    resultOfPrunning["Fold " + str(fnum)] = [abp, aap]
    fnum += 1


#before prunning
printTree(bestTreeBeforePrun)

#after prunning
printTree(bestTreeAfterPrun)

allFoldsforPrunning=pd.DataFrame.from_dict(resultOfPrunning,orient='index',
                       columns=['Accuracy Before Prunning','Accuracy After Prunning'])
allFoldsforPrunning.loc['Avrg. OF Folds'] = allFoldsforPrunning.mean()
#printing Table as Juptre Markdown
#print(allFoldsforPrunning.to_markdown())