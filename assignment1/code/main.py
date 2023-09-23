#BBM409 Assignment 1
#b21946242 Numan Kafadar
#b21946198 Umut Güngör

#importing necessary libraries
import numpy as np
import pandas as pd
import math
import time
from random import randint


# fischer-yates algorithm to shuffle the dataset
def shuffle(arr):
    n = len(arr)
    arr = arr.tolist()
    for i in range(n - 1, 0, -1):
        j = randint(0, i + 1)
        arr[i], arr[j] = arr[j], arr[i]
    return np.asarray(arr)


# normalization method for training dataset
def normalization(data_set):
    # geting min and max values of each columns
    min_set = data_set.min(axis=0)
    max_set = data_set.max(axis=0)
    num_rows, num_cols = data_set.shape

    new_data_set = np.zeros((num_rows, num_cols)).astype(dtype="float64")

    # min-max normalization
    for i in range(num_rows):
        for j in range(num_cols):
            a1 = data_set[i, j] - min_set[j]
            a2 = max_set[j] - min_set[j]
            new_data_set[i, j] = a1 / a2

    return new_data_set


# splitting data into 5 parts using kFold
def kFold(data, labels, k=5):
    size = len(data)
    n = int(size / k)  # length of test samples
    train_test_data = []

    for i in range(k):
        x_train = np.concatenate([data[:(i * n)], data[((i + 1) * n):]])
        x_test = np.concatenate([labels[:(i * n)], labels[((i + 1) * n):]])
        y_train = data[(i * n):((i + 1) * n)]
        y_test = labels[(i * n):((i + 1) * n)]
        train_test_data.append([x_train, x_test, y_train, y_test])

    return train_test_data


# euclidean distance algorithm using numpy functions
def fast_euclidean_distance(x, y):
    return np.power(np.sum(np.power((x - y), 2)), 0.5, dtype="float64")


# calculating distance arrays for eacht test sample before KNN prediction
def calculate_distances(train_data, test_data, labels):
    result = []
    for test_vector in test_data:
        nearest_neighbors = []
        idx = 0
        for row in train_data:
            dist = fast_euclidean_distance(test_vector, row)
            nearest_neighbors.append((dist, labels[idx]))
            idx += 1

        nearest_neighbors.sort(key=lambda t: t[0])  # sort the distances
        result.append(nearest_neighbors)

    return result


# KNN algorithm for prediction
def kNN(distances, k, kNNType="Normal"):
    predictions = []
    # weighted KNN
    if kNNType == "Weighted":
        for d in distances:
            # get first k elements of the distance array
            results = [row[1] for row in d[:k]]
            dists = [row[0] for row in d[:k]]
            neigs = {}

            for i in range(len(results)):
                # for each distance value, get its weight as 1/d
                weight = 1.0 / dists[i]
                if results[i] in neigs:
                    neigs[results[i]] += weight
                else:
                    neigs[results[i]] = weight
            predictions.append(max(neigs, key=neigs.get))

    # normal KNN
    else:
        for d in distances:
            # get first k elements of the distance array
            results = [row[1] for row in d[:k]]
            pred = max(set(results), key=results.count)
            predictions.append(pred)

    return predictions


# calculating accuracy, precision and recall metrics using confusion matrix
def get_statics(actualVals, predVals, classes):
    # confusion matrix
    conf_matrix = pd.DataFrame(
        np.zeros((len(classes), len(classes)), dtype=int),
        index=classes, columns=classes)
    for i, j in zip(actualVals, predVals):
        conf_matrix.loc[i, j] += 1

    # getting precision and recall using confusion matrix
    precision, recall = calculate_metrics(conf_matrix.to_numpy())

    # calculating accuracy
    correct = 0
    for i in range(len(actualVals)):
        if actualVals[i] == predVals[i]:
            correct += 1

    accuracy = correct / float(len(actualVals))

    return accuracy * 100, precision * 100, recall * 100


# a method for getting precision and recall metrics using confusion matrix
def calculate_metrics(metrics_matrix):
    precision = 0.0
    recall = 0.0
    for i in range(16):
        precision += metrics_matrix[i, i] / np.sum(metrics_matrix[i])
        recall += metrics_matrix[i, i] / np.sum(metrics_matrix[:, i])

    precision /= 16
    recall /= 16

    return precision, recall


# Mean Absolute Error calculation method
def mae(actuals, preds):
    dif = 0.0
    for i in range(len(actuals)):
        dif += abs(actuals[i] - preds[i])
    return dif / len(actuals)


# a function to get misclassified items to comment why they are misclassified
"""
def get_misclassified_items(test_data,actuals,preds):
    res=[]
    for i in range(len(actuals)):
        if actuals[i] != preds[i]:
            res.append([test_data[i],actuals[i],preds[i]])
    return res
"""
#data exploring
df = pd.read_csv("subset_16P.csv", encoding="cp1252")
df.head()

arr = np.array(df)

#shuffle the numpy array
arr=shuffle(arr)

#splitting labels and features
x = arr[:, 1:-1].astype(dtype="int32")
y = arr[:, -1:]
y = y.reshape(len(y),)

#split the data into 5 fold
folds=kFold(x,y)

#First , getting all the distance arrays to process KNN
foldsResult=[]
fnum=1
print("Non-Normalized Data ,Getting Distance arrays...")
for fold in folds:
    st=time.time()
    dist=calculate_distances(fold[0],fold[2],fold[1])
    et=time.time()
    passed=et-st
    print("For fold {} , prediction time = {} ".format(fnum,passed))
    foldsResult.append(dist)
    fnum+=1

# class list for part 1
classes = df["Personality"].unique()


# Average of 5-fold Cross Validation Results
def get_Avrgs(arr, n):
    res = []
    # calculating each folds average results
    for i in range(5):
        tmp = [0.0] * n
        for k in range(n):
            for j in range(5):
                tmp[k] += arr[j][i][k]
        res.append(tmp)

    for i in range(5):
        for j in range(n):
            res[i][j] = round(res[i][j] / 5, 3)
    rN = []
    rW = []

    # getting values in table format for jupyter markdown
    for i in range(5):
        rN.append("Accuracy : %" + str(res[i][0]) + "<br> Precision : %" + str(res[i][1]) +
                  "<br> Recall : %" + str(res[i][2]))
        rW.append("Accuracy : %" + str(res[i][3]) + "<br> Precision : %" + str(res[i][4]) +
                  "<br> Recall : %" + str(res[i][5]))
    # return normal averages and weighted averages of each folds
    return rN, rW


# processing KNN for each k value and each fold
# this function simply returns pandas Dataframe of average,precision and recall metrics table
# for both normal and weighted KNN
def get_KNN_Table_pt1(fold_result, folds):
    kArray = [1, 3, 5, 7, 9]
    fnum = 1
    avrgNonNormalizd = []
    resaN = []
    resaW = []
    for dists in fold_result:
        tmp = []
        aN = []
        aW = []
        for k in kArray:
            predsNormal = kNN(dists, k)
            predsWeighted = kNN(dists, k, kNNType="Weighted")
            accuracyN, precisionN, recallN = get_statics(folds[fnum - 1][3], predsNormal, classes)
            accuracyW, precisionW, recallW = get_statics(folds[fnum - 1][3], predsWeighted, classes)
            aN.append("Accuracy : %" + str(round(accuracyN, 3)) + "<br> Precision : %" + str(round(precisionN, 3)) +
                      "<br> Recall : %" + str(round(recallN, 3)))
            aW.append("Accuracy : %" + str(round(accuracyW, 3)) + "<br> Precision : %" + str(round(precisionW, 3)) +
                      "<br> Recall : %" + str(round(recallW, 3)))
            tmp.append([accuracyN, precisionN, recallN, accuracyW, precisionW, recallW])
        avrgNonNormalizd.append(tmp)
        fnum += 1
        resaN.append(aN)
        resaW.append(aW)

    avrN, avrW = get_Avrgs(avrgNonNormalizd.copy(), 6)
    resaN.append(avrN)
    resaW.append(avrW)

    # normal KNN Results as Dataframe
    df1 = pd.DataFrame(resaN, index=["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5", "Average Of The Folds"],
                       columns=["k=1", "k=3", "k=5", "k=7", "k=9"])

    # Weighted KNN Results as Dataframe
    df2 = pd.DataFrame(resaW, index=["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5", "Average Of The Folds"],
                       columns=["k=1", "k=3", "k=5", "k=7", "k=9"])
    return df1, df2


# getting results of KNN for Non-Normalized Data
df_normal, df_weighted = get_KNN_Table_pt1(foldsResult, folds)

#printing Non-Normalized and Non-Weighted KNN for report
#print(df_normal.to_markdown())

#printing Non-Normalized and Weighted KNN for report
#print(df_weighted.to_markdown())

normalized_x=normalization(x) #normalize the train set
foldsNormalized=kFold(normalized_x,y)
#First , getting all the distance arrays to process KNN
foldsResultNormalized=[]
fnum=1
print("Normalized Data ,Getting Distance arrays...")
for fold in foldsNormalized:
    st=time.time()
    dist=calculate_distances(fold[0],fold[2],fold[1])
    et=time.time()
    passed=et-st
    print("For fold {} , prediction time = {} ".format(fnum,passed))
    foldsResultNormalized.append(dist)
    fnum+=1


#getting results of KNN for Normalized Data
normalized_df_normal,normalized_df_weighted=get_KNN_Table_pt1(foldsResultNormalized,foldsNormalized)

#printing Normalized and Non-Weighted KNN for report as markdown table
#print(normalized_df_normal.to_markdown())

#printing Normalized and Weighted KNN for report as markdown table
#print(normalized_df_weighted.to_markdown())

#data exploring
df2=pd.read_csv("energy_efficiency_data.csv")
df2.head()

arr_2 = np.array(df2).astype(dtype="float64")

arr_2=shuffle(arr_2) #shuffling the array
x_values = arr_2[:, :-2]

#normalized feature values
normalized_x_values=normalization(x_values)

y_heating_load = arr_2[:, -2:-1]
y_cooling_load = arr_2[:, -1:]
y_heating_load = y_heating_load.reshape(len(y_heating_load),)
y_cooling_load = y_cooling_load.reshape(len(y_cooling_load),)

#5-fold cross validation
folds_heating=kFold(x_values,y_heating_load)
folds_cooling=kFold(x_values,y_cooling_load)
folds_heating_normalized=kFold(normalized_x_values,y_heating_load)
folds_cooling_normalized=kFold(normalized_x_values,y_cooling_load)


# KNN regression algorithm for part 2
def kNN_for_Regression(distances, k, kNNType="Normal"):
    predictions = []
    # weighted KNN
    if kNNType == "Weighted":

        for d in distances:
            # get first k elements of the distance array
            results = [row[1] for row in d[:k]]
            dists = [row[0] for row in d[:k]]
            neigs = {}

            pred = 0.0
            weight_sum = 0.0  # sum of k weightes
            for i in range(len(results)):
                # for each distance value, get its weight as 1/d
                weight = 1.0 / dists[i]
                weight_sum += weight
                # for each closest sample, sum up pred value with weight * samples label value
                pred += weight * results[i]

                # our prediction
            pred = pred / weight_sum
            predictions.append(pred)

    # normal KNN
    else:
        for d in distances:
            # get first k elements of the distance array
            results = [row[1] for row in d[:k]]
            # getting average of the closest sample labels
            pred = float(sum(results) / k)
            predictions.append(pred)

    return predictions


#calculating distance arrays for Heating Load
folds_heating_result=[]
fnum=1
print("Non-normalized Data ,Getting Distance Arrays......")
for fold in folds_heating:
    st=time.time()
    dist=calculate_distances(fold[0],fold[2],fold[1])
    et=time.time()
    passed=et-st
    print("For fold {} , prediction time = {} ".format(fnum,passed))
    folds_heating_result.append(dist)
    fnum+=1

folds_heating_result_normalized=[]
fnum=1
print("Normalized Data ,Getting Distance Arrays...")
for fold in folds_heating_normalized:
    st=time.time()
    dist=calculate_distances(fold[0],fold[2],fold[1])
    et=time.time()
    passed=et-st
    print("For fold {} , prediction time = {} ".format(fnum,passed))
    folds_heating_result_normalized.append(dist)
    fnum+=1


#calculating distance arrays for Cooling Load
folds_cooling_result=[]
fnum=1
print("Non-normalized Data Prediction...")
for fold in folds_cooling:
    st=time.time()
    dist=calculate_distances(fold[0],fold[2],fold[1])
    et=time.time()
    passed=et-st
    print("For fold {} , prediction time = {} ".format(fnum,passed))
    folds_cooling_result.append(dist)
    fnum+=1

folds_cooling_result_normalized=[]
fnum=1
print("Non-normalized Data Prediction...")
for fold in folds_cooling_normalized:
    st=time.time()
    dist=calculate_distances(fold[0],fold[2],fold[1])
    et=time.time()
    passed=et-st
    print("For fold {} , prediction time = {} ".format(fnum,passed))
    folds_cooling_result_normalized.append(dist)
    fnum+=1


# a function that returns average of MAE values for each k value
def get_Avrgs_For_Mae(arr, n):
    res = []
    for i in range(5):
        tmp = [0.0] * n
        for k in range(n):
            for j in range(5):
                tmp[k] += arr[j][i][k]
        res.append(tmp)

    for i in range(5):
        for j in range(n):
            res[i][j] = "MAE :%" + str(round(res[i][j] / 5, 3))

    return [i[0] for i in res], [i[1] for i in res]


# processing KNN for each k value and each fold
# this function simply returns pandas Dataframe of MAE metrics table
# for both normal and weighted KNN
def get_KNN_Table_pt2(fold_result, folds):
    kArray = [1, 3, 5, 7, 9]
    fnum = 1
    avrgArray = []
    resNormal = []
    resWeighted = []
    for dists in fold_result:
        tmp = []
        tmpN = []
        tmpW = []
        for k in kArray:
            predsNormal = kNN_for_Regression(dists, k)
            predsWeighted = kNN_for_Regression(dists, k, kNNType="Weighted")

            maeNormal = mae(predsNormal, folds[fnum - 1][3])
            maeWeighted = mae(predsWeighted, folds[fnum - 1][3])
            tmp.append([maeNormal, maeWeighted])
            tmpN.append("MAE: %" + str(round(maeNormal, 3)))
            tmpW.append("MAE: %" + str(round(maeWeighted, 3)))

        avrgArray.append(tmp)
        fnum += 1
        resNormal.append(tmpN)
        resWeighted.append(tmpW)
    avrN, avrW = get_Avrgs_For_Mae(avrgArray, 2)
    resNormal.append(avrN)
    resWeighted.append(avrW)

    # normal KNN Results as Dataframe
    df1 = pd.DataFrame(resNormal, index=["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5", "Average Of The Folds"],
                       columns=["k=1", "k=3", "k=5", "k=7", "k=9"])

    # weighted KNN Results as Dataframe
    df2 = pd.DataFrame(resWeighted, index=["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5", "Average Of The Folds"],
                       columns=["k=1", "k=3", "k=5", "k=7", "k=9"])
    return df1, df2

#KNN processing for each combinations
dt_heating_normal,dt_heating_weighted=get_KNN_Table_pt2(folds_heating_result,folds_heating)

normalized_heating_normal, normalized_heating_weighted = get_KNN_Table_pt2(folds_heating_result_normalized,folds_heating_normalized)

dt_cooling_normal,dt_cooling_weighted=get_KNN_Table_pt2(folds_cooling_result,folds_cooling)

normalized_cooling_normal, normalized_cooling_weighted = get_KNN_Table_pt2(folds_cooling_result_normalized,folds_cooling_normalized)


#printing Heating Load KNN results for report as markdown table
"""
print("Table of Heating Load For Non-normalized and Non-Weighted Data\n")
print(dt_heating_normal.to_markdown()+"\n")

print("Table of Heating Load For Non-normalized and Weighted Data\n")
print(dt_heating_weighted.to_markdown()+"\n")

print("Table of Heating Load For Normalized and Non-Weighted Data\n")
print(normalized_heating_normal.to_markdown()+"\n")

print("Table of Heating Load For Normalized and Weighted Data\n")
print(normalized_heating_weighted.to_markdown()+"\n")
"""

#printing Cooling Load KNN results for report as markdown table
"""
print("Table of Cooling Load For Non-normalized and Non-Weighted Data\n")
print(dt_cooling_normal.to_markdown()+"\n")

print("Table of Cooling Load For Non-normalized and Weighted Data\n")
print(dt_cooling_weighted.to_markdown()+"\n")

print("Table of Cooling Load For Normalized and Non-Weighted Data\n")
print(normalized_cooling_normal.to_markdown()+"\n")

print("Table of Cooling Load For Normalized and Weighted Data\n")
print(normalized_cooling_weighted.to_markdown()+"\n")
"""